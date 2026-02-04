"""
개선된 학습 파이프라인 (Phase 1 + Phase 2)

Phase 1:
- LightGBM 하이퍼파라미터 튜닝 (Optuna)
- 피처 선택 (Feature Importance)
- 클래스 불균형 처리 (scale_pos_weight)

Phase 2:
- Meta-labeling (2차 필터 모델)
- Triple Barrier 최적화
- CatBoost 앙상블
"""
from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score

from packages.common.db import execute, fetch_all
from services.registry.storage import upload_json, upload_model
from services.training.splits import SplitConfig, walk_forward_splits

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

Timeframe = Literal["1m", "15m", "1h"]


@dataclass
class ImprovedTrainConfig:
    """개선된 학습 설정"""
    label_spec_hash: str
    feature_schema_version: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    targets: Tuple[str, ...] = ("er_long", "q05_long", "e_mae_long", "e_hold_long")
    algo: str = "lgbm"
    purge_bars: int = 0
    embargo_pct: float = 0.0
    label_spec: Dict[str, Any] | None = None
    timeframe: Timeframe = "1m"
    use_multi_tf: bool = False

    # Phase 1: 하이퍼파라미터 튜닝
    use_optuna: bool = True
    optuna_trials: int = 50

    # Phase 1: 피처 선택
    use_feature_selection: bool = True
    feature_importance_threshold: float = 0.001  # 중요도 하위 % 제거

    # Phase 1: 클래스 불균형
    handle_imbalance: bool = True

    # Phase 2: Meta-labeling
    use_meta_labeling: bool = True
    meta_threshold: float = 0.5  # 메타 모델 신뢰도 임계값

    # Phase 2: CatBoost 앙상블
    use_catboost_ensemble: bool = True
    ensemble_weights: Tuple[float, float] = (0.6, 0.4)  # (lgbm, catboost)

    # 병렬 학습 설정
    parallel_workers: int = 2  # 동시 학습 타겟 수 (1=순차, 2+=병렬)


def _trade_metrics(returns: pd.Series) -> Dict[str, float]:
    """거래 지표 계산"""
    if len(returns) == 0:
        return {"profit_factor": 0, "max_drawdown": 0, "expectancy": 0, "tail_loss": 0, "turnover": 0}

    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    profit_factor = float(pos / abs(neg)) if neg != 0 else float("inf")
    cum = returns.cumsum()
    peak = cum.cummax()
    drawdown = (cum - peak).min()
    tail = returns.nsmallest(max(1, int(len(returns) * 0.01))).mean()
    return {
        "profit_factor": profit_factor,
        "max_drawdown": float(drawdown),
        "expectancy": float(returns.mean()),
        "tail_loss": float(tail),
        "turnover": float(len(returns)),
        "win_rate": float(len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0,
    }


def _normalize_features(df: pd.DataFrame, prefix: str = "f") -> Tuple[pd.DataFrame, list[str]]:
    """피처 정규화"""
    def _parse_features(val: Any) -> Any:
        if isinstance(val, (bytes, bytearray)):
            val = val.decode()
        if isinstance(val, str):
            return json.loads(val)
        return val

    df["features"] = df["features"].apply(_parse_features)
    feats = pd.json_normalize(df["features"])
    feats.columns = [f"{prefix}_{c}" for c in feats.columns]

    for col in feats.columns:
        feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0)

    return pd.concat([df.drop(columns=["features"]), feats], axis=1), feats.columns.tolist()


def _load_multi_tf_dataset(
    label_spec_hash: str,
    feature_schema_version: int,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: List[str] | None = None,
) -> pd.DataFrame:
    """멀티 타임프레임 데이터셋 로드"""
    import gc

    if symbols:
        symbols_list = symbols
    else:
        symbols_rows = fetch_all(
            "SELECT DISTINCT symbol FROM instruments WHERE status = 'active'"
        )
        symbols_list = [r[0] for r in symbols_rows]

    all_dfs = []
    all_feature_cols = None

    for symbol in symbols_list:
        # 1m features + labels (long + short)
        query_1m = """
            SELECT f.symbol, f.ts, f.features, f.schema_version, f.atr, f.funding_z, f.btc_regime,
                   ll.ret_net as ret_net_long, ll.mae as mae_long, ll.time_to_event_min as time_to_event_min_long, ll.y as y_long,
                   ls.ret_net as ret_net_short, ls.mae as mae_short, ls.time_to_event_min as time_to_event_min_short, ls.y as y_short
            FROM features_1m f
            JOIN labels_long_1m ll
              ON f.symbol = ll.symbol AND f.ts = ll.ts
            JOIN labels_short_1m ls
              ON f.symbol = ls.symbol AND f.ts = ls.ts
            WHERE f.schema_version = %s AND ll.spec_hash = %s AND ls.spec_hash = %s AND f.symbol = %s
        """
        params: list = [feature_schema_version, label_spec_hash, label_spec_hash, symbol]
        if start_date:
            query_1m += " AND f.ts >= %s"
            params.append(start_date)
        if end_date:
            query_1m += " AND f.ts <= %s"
            params.append(end_date)
        query_1m += " ORDER BY f.ts"

        rows_1m = fetch_all(query_1m, tuple(params))
        if not rows_1m:
            continue

        df_1m = pd.DataFrame(
            rows_1m,
            columns=[
                "symbol", "ts", "features", "schema_version", "atr", "funding_z", "btc_regime",
                "ret_net_long", "mae_long", "time_to_event_min_long", "y_long",
                "ret_net_short", "mae_short", "time_to_event_min_short", "y_short",
            ],
        )
        df_1m["ts"] = pd.to_datetime(df_1m["ts"], utc=True)
        df_1m, cols_1m = _normalize_features(df_1m, prefix="f_1m")

        # 15m features
        query_15m = """SELECT ts, features FROM features_15m WHERE symbol = %s"""
        params_15m = [symbol]
        if start_date:
            query_15m += " AND ts >= %s"
            params_15m.append(start_date)
        if end_date:
            query_15m += " AND ts <= %s"
            params_15m.append(end_date)
        query_15m += " ORDER BY ts"

        rows_15m = fetch_all(query_15m, tuple(params_15m))
        if rows_15m:
            df_15m = pd.DataFrame(rows_15m, columns=["ts", "features"])
            df_15m["ts"] = pd.to_datetime(df_15m["ts"], utc=True)
            df_15m, cols_15m = _normalize_features(df_15m, prefix="f_15m")
            df_15m = df_15m.set_index("ts")

            df_1m = df_1m.set_index("ts")
            for col in cols_15m:
                df_1m[col] = df_15m[col].reindex(df_1m.index, method="ffill").values
            df_1m = df_1m.reset_index()
        else:
            cols_15m = []

        # 1h features
        query_1h = """SELECT ts, features FROM features_1h WHERE symbol = %s"""
        params_1h = [symbol]
        if start_date:
            query_1h += " AND ts >= %s"
            params_1h.append(start_date)
        if end_date:
            query_1h += " AND ts <= %s"
            params_1h.append(end_date)
        query_1h += " ORDER BY ts"

        rows_1h = fetch_all(query_1h, tuple(params_1h))
        if rows_1h:
            df_1h = pd.DataFrame(rows_1h, columns=["ts", "features"])
            df_1h["ts"] = pd.to_datetime(df_1h["ts"], utc=True)
            df_1h, cols_1h = _normalize_features(df_1h, prefix="f_1h")
            df_1h = df_1h.set_index("ts")

            if "ts" in df_1m.columns:
                df_1m = df_1m.set_index("ts")
            for col in cols_1h:
                df_1m[col] = df_1h[col].reindex(df_1m.index, method="ffill").values
            df_1m = df_1m.reset_index()
        else:
            cols_1h = []

        all_cols = cols_1m + cols_15m + cols_1h
        if all_feature_cols is None:
            all_feature_cols = all_cols

        df_1m["feature_cols"] = [all_cols] * len(df_1m)
        all_dfs.append(df_1m)

        del rows_1m
        gc.collect()

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    result = result.sort_values("ts")
    return result


# =============================================================================
# Phase 1: Optuna 하이퍼파라미터 튜닝
# =============================================================================

def optimize_lgbm_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    objective: str = "regression",
    n_trials: int = 50,
    alpha: float | None = None,
) -> Dict[str, Any]:
    """Optuna를 사용한 LightGBM 하이퍼파라미터 최적화"""
    if not OPTUNA_AVAILABLE:
        print("  Optuna not available, using default params")
        return _get_default_lgbm_params(objective, alpha)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective_func(trial):
        params = {
            "objective": objective,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbose": -1,
        }
        if objective == "quantile" and alpha is not None:
            params["alpha"] = alpha

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, pred)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["objective"] = objective
    best_params["verbose"] = -1
    if objective == "quantile" and alpha is not None:
        best_params["alpha"] = alpha

    print(f"  Best params (RMSE={study.best_value:.4f}): lr={best_params['learning_rate']:.4f}, "
          f"leaves={best_params['num_leaves']}, depth={best_params['max_depth']}")

    return best_params


def _get_default_lgbm_params(objective: str, alpha: float | None = None) -> Dict[str, Any]:
    """기본 LightGBM 파라미터"""
    params = {
        "objective": objective,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha
    return params


# =============================================================================
# Phase 1: 피처 선택
# =============================================================================

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.001,
) -> List[str]:
    """Feature importance 기반 피처 선택"""
    # Quick LightGBM for feature importance
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=32,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    })
    importances["importance_pct"] = importances["importance"] / importances["importance"].sum()
    importances = importances.sort_values("importance", ascending=False)

    # 중요도 합계가 threshold 이하인 피처 제거
    cumsum = 0
    selected = []
    for _, row in importances.iterrows():
        if cumsum < (1 - threshold):
            selected.append(row["feature"])
            cumsum += row["importance_pct"]
        else:
            break

    # 최소 10개 피처는 유지
    if len(selected) < 10:
        selected = importances.head(10)["feature"].tolist()

    return selected


# =============================================================================
# Phase 2: Meta-labeling
# =============================================================================

def train_meta_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,  # 0/1 binary labels (TP hit or not)
    X_val: pd.DataFrame,
    y_val: pd.Series,
    primary_pred_train: np.ndarray,  # 1차 모델 예측값
    primary_pred_val: np.ndarray,
) -> lgb.LGBMClassifier:
    """Meta-labeling 모델 학습"""
    # 1차 모델 예측을 피처로 추가
    X_train_meta = X_train.copy()
    X_train_meta["primary_pred"] = primary_pred_train

    X_val_meta = X_val.copy()
    X_val_meta["primary_pred"] = primary_pred_val

    # 클래스 가중치 계산
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=32,
        max_depth=6,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )

    model.fit(
        X_train_meta, y_train,
        eval_set=[(X_val_meta, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # 성능 출력
    pred_proba = model.predict_proba(X_val_meta)[:, 1]
    pred_binary = (pred_proba > 0.5).astype(int)
    acc = accuracy_score(y_val, pred_binary)
    f1 = f1_score(y_val, pred_binary, zero_division=0)
    print(f"  Meta-model: Accuracy={acc:.3f}, F1={f1:.3f}")

    return model


# =============================================================================
# Phase 2: CatBoost 앙상블
# =============================================================================

def train_catboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    objective: str = "RMSE",
) -> Optional[CatBoostRegressor]:
    """CatBoost 회귀 모델 학습"""
    if not CATBOOST_AVAILABLE:
        print("  CatBoost not available")
        return None

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function=objective,
        verbose=0,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    print(f"  CatBoost RMSE: {rmse:.4f}")

    return model


def ensemble_predict(
    models: Dict[str, Any],
    X: pd.DataFrame,
    weights: Tuple[float, float] = (0.6, 0.4),
) -> np.ndarray:
    """앙상블 예측"""
    lgbm_model = models.get("lgbm")
    catboost_model = models.get("catboost")

    if lgbm_model is None:
        return np.zeros(len(X))

    lgbm_pred = lgbm_model.predict(X)

    if catboost_model is None:
        return lgbm_pred

    catboost_pred = catboost_model.predict(X)
    return weights[0] * lgbm_pred + weights[1] * catboost_pred


# =============================================================================
# 통합 학습 함수
# =============================================================================

def run_improved_training(
    cfg: ImprovedTrainConfig,
    symbols: List[str] | None = None,
) -> Dict[str, Any]:
    """개선된 학습 파이프라인 실행"""
    print("=" * 60)
    print("개선된 학습 파이프라인 시작")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/7] 데이터 로드 중...")
    if cfg.use_multi_tf:
        df = _load_multi_tf_dataset(
            cfg.label_spec_hash,
            cfg.feature_schema_version,
            start_date=cfg.train_start,
            end_date=cfg.val_end,
            symbols=symbols,
        )
    else:
        from services.training.train import _load_dataset
        df = _load_dataset(
            cfg.label_spec_hash,
            cfg.feature_schema_version,
            start_date=cfg.train_start,
            end_date=cfg.val_end,
            timeframe=cfg.timeframe,
            symbols=symbols,
        )

    if df.empty:
        return {"status": "empty", "message": "No training data"}

    print(f"  → {len(df):,}개 샘플 로드")

    # Train/Val 분할
    split_cfg = SplitConfig(purge_bars=cfg.purge_bars, embargo_pct=cfg.embargo_pct)
    splits = list(walk_forward_splits(df["ts"], split_cfg))
    if not splits:
        return {"status": "empty", "message": "Insufficient split data"}
    train_idx, val_idx, test_idx = splits[-1]
    train_df = df[df["ts"].isin(train_idx)]
    val_df = df[df["ts"].isin(val_idx)]

    if train_df.empty or val_df.empty:
        return {"status": "empty", "message": "Insufficient split data"}

    print(f"  → Train: {len(train_df):,}, Val: {len(val_df):,}")

    feature_cols = train_df.iloc[0]["feature_cols"]
    X_train = train_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    X_val = val_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Phase 1: 피처 선택
    if cfg.use_feature_selection:
        print("\n[2/7] 피처 선택 중...")
        y_for_selection = train_df["ret_net_long"].fillna(0.0)
        selected_features = select_features(X_train, y_for_selection, cfg.feature_importance_threshold)
        print(f"  → {len(feature_cols)} → {len(selected_features)} 피처 선택")
        feature_cols = selected_features
        X_train = X_train[feature_cols]
        X_val = X_val[feature_cols]
    else:
        print("\n[2/7] 피처 선택 스킵")

    # 타겟 준비 (롱 + 숏)
    target_map = {
        "er_long": "ret_net_long",
        "q05_long": "ret_net_long",
        "e_mae_long": "mae_long",
        "e_hold_long": "time_to_event_min_long",
        "er_short": "ret_net_short",
        "q05_short": "ret_net_short",
        "e_mae_short": "mae_short",
        "e_hold_short": "time_to_event_min_short",
    }

    metrics: Dict[str, Any] = {}
    models: Dict[str, Dict[str, Any]] = {}

    # 단일 타겟 학습 함수 (병렬 처리용)
    def _train_single_target(target: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """단일 타겟 학습 - 병렬 처리를 위해 분리"""
        raw_target = target_map.get(target)
        if raw_target is None:
            return target, {}, {}

        objective = "quantile" if target.startswith("q05") else "regression"
        alpha = 0.05 if objective == "quantile" else None

        y_train_target = train_df[raw_target].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        y_val_target = val_df[raw_target].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Optuna 튜닝
        if cfg.use_optuna:
            best_params = optimize_lgbm_params(
                X_train, y_train_target, X_val, y_val_target,
                objective=objective, n_trials=cfg.optuna_trials, alpha=alpha
            )
        else:
            best_params = _get_default_lgbm_params(objective, alpha)

        # LightGBM 학습
        lgbm_model = lgb.LGBMRegressor(**best_params)
        lgbm_model.fit(
            X_train, y_train_target,
            eval_set=[(X_val, y_val_target)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        lgbm_pred = lgbm_model.predict(X_val)
        lgbm_rmse = float(np.sqrt(mean_squared_error(y_val_target, lgbm_pred)))

        target_models = {"lgbm": lgbm_model}
        final_pred = lgbm_pred

        # CatBoost 앙상블
        if cfg.use_catboost_ensemble:
            catboost_model = train_catboost_regressor(X_train, y_train_target, X_val, y_val_target)
            if catboost_model:
                target_models["catboost"] = catboost_model
                ensemble_pred = ensemble_predict(target_models, X_val, cfg.ensemble_weights)
                ensemble_rmse = float(np.sqrt(mean_squared_error(y_val_target, ensemble_pred)))
                final_pred = ensemble_pred

        target_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val_target, final_pred))),
            "mae": float(mean_absolute_error(y_val_target, final_pred)),
            "best_iteration": lgbm_model.best_iteration_ if hasattr(lgbm_model, 'best_iteration_') else None,
            "params": best_params,
            "lgbm_rmse": lgbm_rmse,
        }

        return target, target_models, target_metrics

    # Phase 1: Optuna 하이퍼파라미터 튜닝 + 학습
    parallel_workers = cfg.parallel_workers if cfg.parallel_workers > 1 else 1
    print(f"\n[3/7] LightGBM 학습 (Optuna 튜닝, {parallel_workers}개 병렬)...")

    valid_targets = [t for t in cfg.targets if target_map.get(t) is not None]

    if parallel_workers > 1:
        # 병렬 학습
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(_train_single_target, t): t for t in valid_targets}
            for future in as_completed(futures):
                target = futures[future]
                try:
                    target_name, target_models, target_metrics = future.result()
                    if target_models:
                        models[target_name] = target_models
                        metrics[target_name] = target_metrics
                        lgbm_rmse = target_metrics.get("lgbm_rmse", target_metrics["rmse"])
                        ensemble_rmse = target_metrics["rmse"]
                        print(f"\n  --- {target_name} ---")
                        print(f"  Best params (RMSE={lgbm_rmse:.4f}): lr={target_metrics['params'].get('learning_rate', 0):.4f}, "
                              f"leaves={target_metrics['params'].get('num_leaves', 0)}, "
                              f"depth={target_metrics['params'].get('max_depth', 0)}")
                        print(f"  LightGBM RMSE: {lgbm_rmse:.4f}")
                        if cfg.use_catboost_ensemble and "catboost" in target_models:
                            cat_pred = target_models["catboost"].predict(X_val)
                            cat_rmse = float(np.sqrt(mean_squared_error(val_df[target_map[target_name]].fillna(0.0), cat_pred)))
                            print(f"  CatBoost RMSE: {cat_rmse:.4f}")
                            print(f"  Ensemble RMSE: {ensemble_rmse:.4f}")
                except Exception as e:
                    print(f"\n  --- {target} --- ERROR: {e}")
    else:
        # 순차 학습 (기존 방식)
        for target in valid_targets:
            print(f"\n  --- {target} ---")
            target_name, target_models, target_metrics = _train_single_target(target)
            if target_models:
                models[target_name] = target_models
                metrics[target_name] = target_metrics
                lgbm_rmse = target_metrics.get("lgbm_rmse", target_metrics["rmse"])
                print(f"  Best params (RMSE={lgbm_rmse:.4f}): lr={target_metrics['params'].get('learning_rate', 0):.4f}, "
                      f"leaves={target_metrics['params'].get('num_leaves', 0)}, "
                      f"depth={target_metrics['params'].get('max_depth', 0)}")
                print(f"  LightGBM RMSE: {lgbm_rmse:.4f}")
                if cfg.use_catboost_ensemble and "catboost" in target_models:
                    cat_pred = target_models["catboost"].predict(X_val)
                    cat_rmse = float(np.sqrt(mean_squared_error(val_df[target_map[target_name]].fillna(0.0), cat_pred)))
                    print(f"  CatBoost RMSE: {cat_rmse:.4f}")
                    print(f"  Ensemble RMSE: {target_metrics['rmse']:.4f}")

    # Phase 2: Meta-labeling
    meta_model = None
    if cfg.use_meta_labeling and "er_long" in models:
        print("\n[4/7] Meta-labeling 모델 학습...")

        # y 라벨 (TP 도달 여부) - 롱 포지션 기준
        y_train_binary = (train_df["y_long"] == 1).astype(int)
        y_val_binary = (val_df["y_long"] == 1).astype(int)

        # 1차 모델 예측
        er_model = models["er_long"]["lgbm"]
        primary_pred_train = er_model.predict(X_train)
        primary_pred_val = er_model.predict(X_val)

        meta_model = train_meta_model(
            X_train, y_train_binary,
            X_val, y_val_binary,
            primary_pred_train, primary_pred_val
        )
        models["meta"] = meta_model
    else:
        print("\n[4/7] Meta-labeling 스킵")

    # 필터링된 거래 지표 계산
    print("\n[5/7] 거래 지표 계산...")
    val_df = val_df.copy()

    # 롱 기본 지표
    base_metrics_long = _trade_metrics(val_df["ret_net_long"].fillna(0.0))
    print(f"  Baseline (Long): PF={base_metrics_long['profit_factor']:.2f}, "
          f"Expectancy={base_metrics_long['expectancy']*100:.2f}%, "
          f"Trades={int(base_metrics_long['turnover']):,}")

    # 숏 기본 지표
    base_metrics_short = _trade_metrics(val_df["ret_net_short"].fillna(0.0))
    print(f"  Baseline (Short): PF={base_metrics_short['profit_factor']:.2f}, "
          f"Expectancy={base_metrics_short['expectancy']*100:.2f}%, "
          f"Trades={int(base_metrics_short['turnover']):,}")

    # er_long > 0 필터
    if "er_long" in models:
        er_pred_long = ensemble_predict(models["er_long"], X_val, cfg.ensemble_weights) \
            if cfg.use_catboost_ensemble else models["er_long"]["lgbm"].predict(X_val)
        val_df["er_pred_long"] = er_pred_long

        # er_long > 0 필터
        filtered_er_long = val_df[val_df["er_pred_long"] > 0]
        er_metrics_long = _trade_metrics(filtered_er_long["ret_net_long"].fillna(0.0))
        print(f"  Long er>0: PF={er_metrics_long['profit_factor']:.2f}, "
              f"Expectancy={er_metrics_long['expectancy']*100:.2f}%, "
              f"Trades={int(er_metrics_long['turnover']):,}")

        # er_long > 0.001 필터
        filtered_er_001_long = val_df[val_df["er_pred_long"] > 0.001]
        er_001_metrics_long = _trade_metrics(filtered_er_001_long["ret_net_long"].fillna(0.0))
        print(f"  Long er>0.001: PF={er_001_metrics_long['profit_factor']:.2f}, "
              f"Expectancy={er_001_metrics_long['expectancy']*100:.2f}%, "
              f"Trades={int(er_001_metrics_long['turnover']):,}")

        metrics["filtered_er0_long"] = er_metrics_long
        metrics["filtered_er001_long"] = er_001_metrics_long

    # er_short > 0 필터
    if "er_short" in models:
        er_pred_short = ensemble_predict(models["er_short"], X_val, cfg.ensemble_weights) \
            if cfg.use_catboost_ensemble else models["er_short"]["lgbm"].predict(X_val)
        val_df["er_pred_short"] = er_pred_short

        # er_short > 0 필터
        filtered_er_short = val_df[val_df["er_pred_short"] > 0]
        er_metrics_short = _trade_metrics(filtered_er_short["ret_net_short"].fillna(0.0))
        print(f"  Short er>0: PF={er_metrics_short['profit_factor']:.2f}, "
              f"Expectancy={er_metrics_short['expectancy']*100:.2f}%, "
              f"Trades={int(er_metrics_short['turnover']):,}")

        # er_short > 0.001 필터
        filtered_er_001_short = val_df[val_df["er_pred_short"] > 0.001]
        er_001_metrics_short = _trade_metrics(filtered_er_001_short["ret_net_short"].fillna(0.0))
        print(f"  Short er>0.001: PF={er_001_metrics_short['profit_factor']:.2f}, "
              f"Expectancy={er_001_metrics_short['expectancy']*100:.2f}%, "
              f"Trades={int(er_001_metrics_short['turnover']):,}")

        metrics["filtered_er0_short"] = er_metrics_short
        metrics["filtered_er001_short"] = er_001_metrics_short

    # 롱+숏 결합 지표 (둘 다 있는 경우)
    if "er_long" in models and "er_short" in models:
        # 롱: er_long > 0.001, 숏: er_short > 0.001
        combined_long = val_df[val_df["er_pred_long"] > 0.001]["ret_net_long"].fillna(0.0)
        combined_short = val_df[val_df["er_pred_short"] > 0.001]["ret_net_short"].fillna(0.0)
        combined_returns = pd.concat([combined_long, combined_short])
        combined_metrics = _trade_metrics(combined_returns)
        print(f"  Combined er>0.001: PF={combined_metrics['profit_factor']:.2f}, "
              f"Expectancy={combined_metrics['expectancy']*100:.2f}%, "
              f"Trades={int(combined_metrics['turnover']):,}")
        metrics["filtered_combined"] = combined_metrics

    # Meta-labeling 필터 (롱만)
    if meta_model is not None and "er_long" in models:
        X_val_meta = X_val.copy()
        X_val_meta["primary_pred"] = val_df["er_pred_long"]
        meta_proba = meta_model.predict_proba(X_val_meta)[:, 1]
        val_df["meta_proba"] = meta_proba

        # er_long > 0 + meta > threshold
        filtered_meta = val_df[(val_df["er_pred_long"] > 0) & (val_df["meta_proba"] > cfg.meta_threshold)]
        meta_metrics = _trade_metrics(filtered_meta["ret_net_long"].fillna(0.0))
        print(f"  Long er>0 + meta>{cfg.meta_threshold}: PF={meta_metrics['profit_factor']:.2f}, "
              f"Expectancy={meta_metrics['expectancy']*100:.2f}%, "
              f"Trades={int(meta_metrics['turnover']):,}")

        metrics["filtered_meta"] = meta_metrics

    metrics["trade_long"] = base_metrics_long
    metrics["trade_short"] = base_metrics_short
    metrics["trade"] = base_metrics_long  # 호환성 유지

    # 심볼별 지표 (롱 기준)
    print("\n[6/7] 심볼별 지표...")
    symbol_metrics: Dict[str, Any] = {}
    for symbol, group in val_df.groupby("symbol"):
        if group.empty:
            continue
        sm = _trade_metrics(group["ret_net_long"].fillna(0.0))
        symbol_metrics[str(symbol)] = sm
        pf_str = f"{sm['profit_factor']:.2f}" if sm['profit_factor'] != float('inf') else "inf"
        print(f"  {symbol}: PF={pf_str}, Expectancy={sm['expectancy']*100:.2f}%")
    metrics["by_symbol"] = symbol_metrics

    # 모델 저장
    print("\n[7/7] 모델 저장...")
    model_id = uuid.uuid4()
    artifact_name = f"models/{model_id}.pkl"
    artifact_uri = upload_model(models, artifact_name)

    report = {
        "model_id": str(model_id),
        "targets": cfg.targets,
        "feature_schema_version": cfg.feature_schema_version,
        "label_spec_hash": cfg.label_spec_hash,
        "timeframe": cfg.timeframe,
        "use_multi_tf": cfg.use_multi_tf,
        "improvements": {
            "optuna": cfg.use_optuna,
            "feature_selection": cfg.use_feature_selection,
            "meta_labeling": cfg.use_meta_labeling,
            "catboost_ensemble": cfg.use_catboost_ensemble,
        },
        "metrics": metrics,
        "feature_count": len(feature_cols),
        "selected_features": feature_cols[:20],  # Top 20
    }

    report_uri = upload_json(report, f"reports/{model_id}.json")

    execute(
        """
        INSERT INTO models (model_id, algo, feature_schema_version, label_spec_hash, train_start, train_end, metrics, artifact_uri, is_production)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            str(model_id),
            "lgbm_improved",
            cfg.feature_schema_version,
            cfg.label_spec_hash,
            cfg.train_start,
            cfg.train_end,
            json.dumps(metrics),
            artifact_uri,
            False,
        ),
    )

    print(f"\n  → Model ID: {model_id}")
    print(f"  → Artifact: {artifact_uri}")

    return {
        "status": "ok",
        "model_id": str(model_id),
        "metrics": metrics,
        "artifact_uri": artifact_uri,
        "report_uri": report_uri,
        "report": report,
    }
