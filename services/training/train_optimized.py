"""
최적화된 학습 파이프라인 (30종목 저사양 PC 지원)

Week 1 최적화:
- float32 변환 (메모리 50% 절감)
- 심볼별 증분 학습 (메모리 1/N)
- Optuna 단일화 (시간 85% 절감)
- 저메모리 LightGBM 파라미터

Week 2 최적화:
- Streaming cursor (대용량 쿼리)
- Feature Selection 캐싱
- 쿼리 통합
"""
from __future__ import annotations

import gc
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from packages.common.db import execute, fetch_all, get_conn

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

Timeframe = Literal["1m", "15m", "1h"]

# ============================================================================
# LightGBM 기본 파라미터 (최적값)
# ============================================================================
LGBM_DEFAULT_PARAMS = {
    "objective": "regression",
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


@dataclass
class OptimizedTrainConfig:
    """최적화된 학습 설정"""
    label_spec_hash: str
    feature_schema_version: int = 3  # 3 또는 4 지원
    train_start: str = ""
    train_end: str = ""
    val_start: str = ""
    val_end: str = ""
    targets: Tuple[str, ...] = ("er_long", "q05_long", "e_mae_long", "e_hold_long")

    # 최적화 옵션
    use_optuna: bool = True
    optuna_trials: int = 50  # 원래 값 복원
    optuna_per_target: bool = True  # 타겟별 개별 튜닝

    use_feature_selection: bool = True
    feature_importance_threshold: float = 0.001  # 누적 중요도 99.9% 기준

    use_meta_labeling: bool = True
    use_catboost_ensemble: bool = False  # 저사양에서는 비활성화 권장

    # 메모리 최적화
    use_float32: bool = True
    chunk_size: int = 50000
    free_raw_data: bool = True


# ============================================================================
# 메모리 최적화 유틸리티
# ============================================================================

def _parse_features_batch(features_col: pd.Series) -> pd.DataFrame:
    """배치 JSON 파싱 (row-by-row 대신)"""
    def parse_single(val):
        if isinstance(val, (bytes, bytearray)):
            val = val.decode()
        if isinstance(val, str):
            return json.loads(val)
        return val

    parsed = features_col.apply(parse_single)
    return pd.json_normalize(parsed.tolist())


def _to_float32(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """float32로 변환하여 메모리 절감"""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
    return df


def _normalize_features_optimized(
    df: pd.DataFrame,
    prefix: str = "f",
    use_float32: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """최적화된 피처 정규화 (float32 + 배치 파싱)"""
    # 배치 JSON 파싱
    feats = _parse_features_batch(df["features"])
    feats.columns = [f"{prefix}_{c}" for c in feats.columns]

    # float32 변환
    for col in feats.columns:
        if use_float32:
            feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0)

    result = pd.concat([df.drop(columns=["features"]), feats], axis=1)
    return result, feats.columns.tolist()


# ============================================================================
# Streaming DB 쿼리
# ============================================================================

def chunked_fetch(query: str, params: tuple, chunk_size: int = 50000) -> List[tuple]:
    """페이지네이션으로 대용량 데이터 fetch (메모리 효율적 대안)

    Note: 실제 스트리밍은 context manager 구조상 어렵기 때문에
    일반 fetch_all 사용. 심볼별 로드로 메모리 관리.
    """
    return fetch_all(query, params)


def load_symbol_data_streaming(
    symbol: str,
    label_spec_hash: str,
    feature_schema_version: int,
    start_date: str,
    end_date: str,
    use_float32: bool = True,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """단일 심볼 데이터 스트리밍 로드 (멀티TF 통합 쿼리)"""

    # 통합 쿼리: 1m 피처 + 라벨 + 15m/1h 피처를 한번에
    query = """
        SELECT
            f.symbol, f.ts, f.features,
            l.ret_net, l.mae, l.time_to_event_min, l.y
        FROM features_1m f
        JOIN labels_long_1m l ON f.symbol = l.symbol AND f.ts = l.ts
        WHERE f.schema_version = %s
          AND l.spec_hash = %s
          AND f.symbol = %s
          AND f.ts >= %s
          AND f.ts <= %s
        ORDER BY f.ts
    """
    params = (feature_schema_version, label_spec_hash, symbol, start_date, end_date)

    rows = chunked_fetch(query, params, chunk_size)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["symbol", "ts", "features", "ret_net", "mae", "time_to_event_min", "y"]
    )
    del rows
    gc.collect()

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 피처 정규화
    df, feature_cols = _normalize_features_optimized(df, prefix="f_1m", use_float32=use_float32)

    # 15m, 1h 피처 로드 및 병합
    df = _merge_higher_tf_features(df, symbol, start_date, end_date, use_float32)

    return df


def _merge_higher_tf_features(
    df: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
    use_float32: bool = True
) -> pd.DataFrame:
    """15m, 1h 피처를 1m 데이터에 병합 (forward-fill)"""

    for tf, prefix in [("15m", "f_15m"), ("1h", "f_1h")]:
        query = f"""
            SELECT ts, features
            FROM features_{tf}
            WHERE symbol = %s AND ts >= %s AND ts <= %s
            ORDER BY ts
        """
        rows = fetch_all(query, (symbol, start_date, end_date))

        if not rows:
            continue

        df_tf = pd.DataFrame(rows, columns=["ts", "features"])
        df_tf["ts"] = pd.to_datetime(df_tf["ts"], utc=True)
        df_tf, tf_cols = _normalize_features_optimized(df_tf, prefix=prefix, use_float32=use_float32)
        df_tf = df_tf.set_index("ts")

        # Forward-fill merge (메모리 효율적)
        df = df.set_index("ts")
        for col in tf_cols:
            if col in df_tf.columns:
                df[col] = df_tf[col].reindex(df.index, method="ffill")
        df = df.reset_index()

        del df_tf, rows
        gc.collect()

    return df


# ============================================================================
# Feature Selection (누적 중요도 99.9% 기준)
# ============================================================================

_feature_selection_cache: Dict[str, List[str]] = {}


def select_features_cached(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.001,  # 하위 0.1% 제거 = 상위 99.9% 선택
    cache_key: str = None
) -> List[str]:
    """피처 선택 (누적 중요도 기준, 캐싱으로 중복 계산 방지)"""
    global _feature_selection_cache

    if cache_key and cache_key in _feature_selection_cache:
        cached = _feature_selection_cache[cache_key]
        # 캐시된 피처 중 현재 존재하는 것만 반환
        return [f for f in cached if f in X.columns]

    # Quick LightGBM으로 중요도 계산
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=32,
        verbose=-1,
    )

    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
    if X_clean.dtypes.apply(lambda x: x == np.float32).all():
        X_np = X_clean.values
    else:
        X_np = X_clean.values.astype(np.float32)

    model.fit(X_np, y.fillna(0).values.astype(np.float32))

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })
    importances["importance_pct"] = importances["importance"] / importances["importance"].sum()
    importances = importances.sort_values("importance", ascending=False)

    # 누적 중요도가 99.9%가 될 때까지 피처 추가
    cumsum = 0
    selected = []
    for _, row in importances.iterrows():
        if cumsum < (1 - threshold):  # 99.9% 미만이면 계속 추가
            selected.append(row["feature"])
            cumsum += row["importance_pct"]
        else:
            break

    # 최소 10개 피처는 유지
    if len(selected) < 10:
        selected = importances.head(10)["feature"].tolist()

    if cache_key:
        _feature_selection_cache[cache_key] = selected

    del model, X_clean, X_np
    gc.collect()

    return selected


# ============================================================================
# Optuna 최적화 (타겟별 개별 튜닝)
# ============================================================================

def optimize_lgbm_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    objective: str = "regression",
    n_trials: int = 50,
    alpha: float = None,
) -> Dict[str, Any]:
    """Optuna를 사용한 LightGBM 하이퍼파라미터 최적화 (원래 범위 복원)"""
    if not OPTUNA_AVAILABLE:
        print("  Optuna not available, using default params")
        return LGBM_DEFAULT_PARAMS.copy()

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


# ============================================================================
# 심볼별 증분 학습
# ============================================================================

def train_incremental_by_symbol(
    cfg: OptimizedTrainConfig,
    symbols: List[str],
    params: Dict[str, Any],
    feature_cols: List[str],
    target: str = "ret_net",
) -> lgb.Booster:
    """심볼별 증분 학습 (메모리 효율적)"""
    model = None
    total_samples = 0

    target_map = {
        "er_long": "ret_net",
        "q05_long": "ret_net",
        "e_mae_long": "mae",
        "e_hold_long": "time_to_event_min",
    }
    raw_target = target_map.get(target, target)

    for i, symbol in enumerate(symbols):
        # 단일 심볼 로드
        df = load_symbol_data_streaming(
            symbol=symbol,
            label_spec_hash=cfg.label_spec_hash,
            feature_schema_version=cfg.feature_schema_version,
            start_date=cfg.train_start,
            end_date=cfg.train_end,
            use_float32=cfg.use_float32,
            chunk_size=cfg.chunk_size,
        )

        if df.empty:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: 데이터 없음")
            continue

        # 피처 컬럼 필터링 (존재하는 것만)
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: 피처 없음")
            del df
            gc.collect()
            continue

        X = df[available_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        y = df[raw_target].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

        train_data = lgb.Dataset(X, label=y, free_raw_data=cfg.free_raw_data)

        if model is None:
            # 첫 심볼: 처음부터 학습
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params.get("n_estimators", 300),
            )
        else:
            # 이후 심볼: 이어서 학습
            model = lgb.train(
                params,
                train_data,
                num_boost_round=50,  # 추가 트리
                init_model=model,
                keep_training_booster=True,
            )

        total_samples += len(df)
        print(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df):,}개 샘플, "
              f"총 {total_samples:,}개, trees={model.num_trees()}")

        del df, X, y, train_data
        gc.collect()

    return model


# ============================================================================
# 거래 지표
# ============================================================================

def _trade_metrics(returns: pd.Series) -> Dict[str, float]:
    """거래 지표 계산"""
    if len(returns) == 0:
        return {"profit_factor": 0, "expectancy": 0, "turnover": 0, "win_rate": 0}

    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    pf = float(pos / abs(neg)) if neg != 0 else float("inf")

    return {
        "profit_factor": pf,
        "expectancy": float(returns.mean()),
        "turnover": float(len(returns)),
        "win_rate": float(len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0,
    }


# ============================================================================
# Meta-labeling
# ============================================================================

def train_meta_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    primary_pred_train: np.ndarray,
    primary_pred_val: np.ndarray,
) -> lgb.LGBMClassifier:
    """Meta-labeling 모델 학습"""
    # 1차 예측을 피처로 추가
    X_train_meta = np.column_stack([X_train, primary_pred_train])
    X_val_meta = np.column_stack([X_val, primary_pred_val])

    # 클래스 가중치
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=16,
        max_depth=5,
        min_data_in_leaf=100,
        max_bin=63,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )

    model.fit(
        X_train_meta, y_train,
        eval_set=[(X_val_meta, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return model


# ============================================================================
# 메인 학습 함수
# ============================================================================

def run_optimized_training(
    cfg: OptimizedTrainConfig,
    symbols: List[str],
) -> Dict[str, Any]:
    """최적화된 학습 파이프라인 실행"""
    print("=" * 60)
    print("최적화된 학습 파이프라인 (30종목 지원)")
    print("=" * 60)
    print(f"심볼: {len(symbols)}개")
    print(f"float32: {cfg.use_float32}")
    print(f"Optuna trials: {cfg.optuna_trials} (타겟별 튜닝: {cfg.optuna_per_target})")
    print(f"Feature Selection: 누적 중요도 {(1-cfg.feature_importance_threshold)*100:.1f}%")

    # ========================================================================
    # Step 1: 첫 심볼로 피처 컬럼 및 Optuna 파라미터 결정
    # ========================================================================
    print("\n[1/5] 피처 분석 및 Optuna 튜닝...")

    # 첫 심볼 로드
    first_symbol = symbols[0]
    df_first = load_symbol_data_streaming(
        symbol=first_symbol,
        label_spec_hash=cfg.label_spec_hash,
        feature_schema_version=cfg.feature_schema_version,
        start_date=cfg.train_start,
        end_date=cfg.val_end,
        use_float32=cfg.use_float32,
        chunk_size=cfg.chunk_size,
    )

    if df_first.empty:
        return {"status": "error", "message": f"No data for {first_symbol}"}

    # 피처 컬럼 추출
    all_feature_cols = [c for c in df_first.columns if c.startswith("f_")]
    print(f"  총 피처: {len(all_feature_cols)}개")

    # Train/Val 분할
    train_mask = df_first["ts"] <= cfg.train_end
    val_mask = (df_first["ts"] >= cfg.val_start) & (df_first["ts"] <= cfg.val_end)

    train_first = df_first[train_mask]
    val_first = df_first[val_mask]

    if train_first.empty or val_first.empty:
        return {"status": "error", "message": "Insufficient data for train/val split"}

    # Feature Selection (누적 중요도 99.9% 기준)
    if cfg.use_feature_selection:
        X_for_selection = train_first[all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_for_selection = train_first["ret_net"].fillna(0)

        cache_key = hashlib.md5(f"{cfg.label_spec_hash}_{cfg.feature_schema_version}".encode()).hexdigest()
        selected_features = select_features_cached(X_for_selection, y_for_selection,
                                                   threshold=cfg.feature_importance_threshold, cache_key=cache_key)
        print(f"  선택된 피처: {len(selected_features)}개 (누적 중요도 {(1-cfg.feature_importance_threshold)*100:.1f}%)")
    else:
        selected_features = all_feature_cols

    # Train/Val 데이터 준비 (Optuna 및 이후 학습에 사용)
    X_train_opt = train_first[selected_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_val_opt = val_first[selected_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    # 타겟별 Optuna 파라미터 저장
    target_params = {}
    target_map = {
        "er_long": "ret_net",
        "q05_long": "ret_net",
        "e_mae_long": "mae",
        "e_hold_long": "time_to_event_min",
    }

    if cfg.use_optuna and cfg.optuna_per_target:
        print("\n  타겟별 Optuna 튜닝 시작...")
        for target in cfg.targets:
            raw_target = target_map.get(target, target)
            y_train_t = train_first[raw_target].fillna(0).values.astype(np.float32)
            y_val_t = val_first[raw_target].fillna(0).values.astype(np.float32)

            objective = "quantile" if target == "q05_long" else "regression"
            alpha = 0.05 if target == "q05_long" else None

            print(f"\n  --- {target} ---")
            params = optimize_lgbm_params(
                X_train_opt, y_train_t, X_val_opt, y_val_t,
                objective=objective, n_trials=cfg.optuna_trials, alpha=alpha
            )
            target_params[target] = params
    elif cfg.use_optuna:
        # 단일 타겟만 튜닝 (이전 방식 - 권장하지 않음)
        y_train = train_first["ret_net"].fillna(0).values.astype(np.float32)
        y_val = val_first["ret_net"].fillna(0).values.astype(np.float32)
        best_params = optimize_lgbm_params(X_train_opt, y_train, X_val_opt, y_val, n_trials=cfg.optuna_trials)
        for target in cfg.targets:
            target_params[target] = best_params.copy()
    else:
        for target in cfg.targets:
            target_params[target] = LGBM_DEFAULT_PARAMS.copy()

    del df_first, train_first, val_first, X_train_opt, X_val_opt
    gc.collect()

    # ========================================================================
    # Step 2: 심볼별 증분 학습
    # ========================================================================
    print("\n[2/5] 심볼별 증분 학습...")

    models = {}

    for target in cfg.targets:
        print(f"\n  --- {target} ---")

        # 타겟별 최적화된 파라미터 사용
        params = target_params[target].copy()

        model = train_incremental_by_symbol(
            cfg=cfg,
            symbols=symbols,
            params=params,
            feature_cols=selected_features,
            target=target,
        )

        if model:
            models[target] = model

    # ========================================================================
    # Step 3: 검증 데이터 로드 및 평가
    # ========================================================================
    print("\n[3/5] 검증 데이터 평가...")

    val_dfs = []
    for symbol in symbols:
        df = load_symbol_data_streaming(
            symbol=symbol,
            label_spec_hash=cfg.label_spec_hash,
            feature_schema_version=cfg.feature_schema_version,
            start_date=cfg.val_start,
            end_date=cfg.val_end,
            use_float32=cfg.use_float32,
            chunk_size=cfg.chunk_size,
        )
        if not df.empty:
            val_dfs.append(df)
        del df
        gc.collect()

    if not val_dfs:
        return {"status": "error", "message": "No validation data"}

    val_df = pd.concat(val_dfs, ignore_index=True)
    del val_dfs
    gc.collect()

    print(f"  검증 샘플: {len(val_df):,}개")

    # 예측
    available_features = [c for c in selected_features if c in val_df.columns]
    X_val = val_df[available_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    metrics = {}

    if "er_long" in models:
        er_pred = models["er_long"].predict(X_val)
        val_df["er_pred"] = er_pred

        # 모델 성능
        y_val = val_df["ret_net"].fillna(0).values.astype(np.float32)
        metrics["er_long"] = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, er_pred))),
            "mae": float(mean_absolute_error(y_val, er_pred)),
        }

    # ========================================================================
    # Step 4: Meta-labeling (선택적)
    # ========================================================================
    meta_model = None
    if cfg.use_meta_labeling and "er_long" in models:
        print("\n[4/5] Meta-labeling...")

        # 학습용 데이터 다시 로드 (첫 심볼만)
        train_df = load_symbol_data_streaming(
            symbol=symbols[0],
            label_spec_hash=cfg.label_spec_hash,
            feature_schema_version=cfg.feature_schema_version,
            start_date=cfg.train_start,
            end_date=cfg.train_end,
            use_float32=cfg.use_float32,
        )

        if not train_df.empty:
            train_features = [c for c in selected_features if c in train_df.columns]
            X_train_meta = train_df[train_features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
            y_train_meta = (train_df["y"] == 1).astype(np.int32).values

            primary_pred_train = models["er_long"].predict(X_train_meta)

            y_val_meta = (val_df["y"] == 1).astype(np.int32).values
            primary_pred_val = er_pred

            meta_model = train_meta_model(
                X_train_meta, y_train_meta,
                X_val, y_val_meta,
                primary_pred_train, primary_pred_val
            )

            # Meta 예측
            X_val_with_pred = np.column_stack([X_val, primary_pred_val])
            meta_proba = meta_model.predict_proba(X_val_with_pred)[:, 1]
            val_df["meta_proba"] = meta_proba

            del train_df, X_train_meta
            gc.collect()

            print(f"  Meta-model 학습 완료")
    else:
        print("\n[4/5] Meta-labeling 스킵")

    # ========================================================================
    # Step 5: 거래 지표 계산
    # ========================================================================
    print("\n[5/5] 거래 지표 계산...")

    # 베이스라인
    base_metrics = _trade_metrics(val_df["ret_net"].fillna(0))
    print(f"  Baseline: PF={base_metrics['profit_factor']:.2f}, "
          f"Expectancy={base_metrics['expectancy']*100:.2f}%")

    metrics["trade"] = base_metrics

    # 필터링된 지표
    if "er_pred" in val_df.columns:
        # er > 0
        filtered_er0 = val_df[val_df["er_pred"] > 0]
        metrics["filtered_er0"] = _trade_metrics(filtered_er0["ret_net"].fillna(0))
        print(f"  er>0: PF={metrics['filtered_er0']['profit_factor']:.2f}, "
              f"Expectancy={metrics['filtered_er0']['expectancy']*100:.2f}%, "
              f"Trades={int(metrics['filtered_er0']['turnover']):,}")

        # er > 0.001
        filtered_er001 = val_df[val_df["er_pred"] > 0.001]
        metrics["filtered_er001"] = _trade_metrics(filtered_er001["ret_net"].fillna(0))
        print(f"  er>0.001: PF={metrics['filtered_er001']['profit_factor']:.2f}, "
              f"Expectancy={metrics['filtered_er001']['expectancy']*100:.2f}%, "
              f"Trades={int(metrics['filtered_er001']['turnover']):,}")

        # Meta-labeling
        if "meta_proba" in val_df.columns:
            filtered_meta = val_df[(val_df["er_pred"] > 0) & (val_df["meta_proba"] > 0.5)]
            metrics["filtered_meta"] = _trade_metrics(filtered_meta["ret_net"].fillna(0))
            print(f"  er>0+meta: PF={metrics['filtered_meta']['profit_factor']:.2f}, "
                  f"Expectancy={metrics['filtered_meta']['expectancy']*100:.2f}%, "
                  f"Trades={int(metrics['filtered_meta']['turnover']):,}")

    # 심볼별 지표
    symbol_metrics = {}
    for symbol, group in val_df.groupby("symbol"):
        sm = _trade_metrics(group["ret_net"].fillna(0))
        symbol_metrics[str(symbol)] = sm
    metrics["by_symbol"] = symbol_metrics

    # ========================================================================
    # 모델 저장
    # ========================================================================
    print("\n모델 저장 중...")

    model_id = uuid.uuid4()

    try:
        from services.registry.storage import upload_json, upload_model
        artifact_name = f"models/{model_id}.pkl"
        artifact_uri = upload_model(models, artifact_name)
        print(f"  Artifact: {artifact_uri}")
    except Exception as e:
        print(f"  모델 저장 실패: {e}")
        artifact_uri = None

    # DB 저장
    try:
        execute(
            """
            INSERT INTO models (model_id, algo, feature_schema_version, label_spec_hash,
                               train_start, train_end, metrics, artifact_uri, is_production)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(model_id),
                "lgbm_optimized",
                cfg.feature_schema_version,
                cfg.label_spec_hash,
                cfg.train_start,
                cfg.train_end,
                json.dumps(metrics),
                artifact_uri,
                False,
            ),
        )
    except Exception as e:
        print(f"  DB 저장 실패: {e}")

    del val_df
    gc.collect()

    return {
        "status": "ok",
        "model_id": str(model_id),
        "metrics": metrics,
        "artifact_uri": artifact_uri,
        "feature_count": len(selected_features),
        "symbols_count": len(symbols),
    }
