from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from packages.common.db import execute, fetch_all
from services.registry.storage import upload_json, upload_model
from services.training.splits import SplitConfig, walk_forward_splits

Timeframe = Literal["1m", "15m", "1h"]


@dataclass
class TrainConfig:
    label_spec_hash: str
    feature_schema_version: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    targets: Tuple[str, ...]
    algo: str = "lgbm"
    purge_bars: int = 0
    embargo_pct: float = 0.0
    label_spec: Dict[str, Any] | None = None
    timeframe: Timeframe = "1m"
    use_multi_tf: bool = False  # 멀티 타임프레임 피처 사용 여부


def _trade_metrics(returns: pd.Series) -> Dict[str, float]:
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
    }


def _pinball_loss(y_true: pd.Series, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true.to_numpy() - y_pred
    loss = np.maximum(alpha * diff, (alpha - 1) * diff)
    return float(loss.mean())


def _summarize_window(idx: pd.DatetimeIndex) -> Dict[str, Any]:
    if idx.empty:
        return {"count": 0}
    return {
        "count": len(idx),
        "start": idx.min().isoformat(),
        "end": idx.max().isoformat(),
    }


def _normalize_features(df: pd.DataFrame, prefix: str = "f") -> Tuple[pd.DataFrame, list[str]]:
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


def _get_feature_table(timeframe: Timeframe) -> str:
    return f"features_{timeframe}"


def _get_label_table(timeframe: Timeframe) -> str:
    return f"labels_long_{timeframe}"


def _load_dataset(
    label_spec_hash: str,
    feature_schema_version: int,
    start_date: str | None = None,
    end_date: str | None = None,
    timeframe: Timeframe = "1m",
    symbols: List[str] | None = None,
) -> pd.DataFrame:
    """Load dataset symbol-by-symbol to reduce peak memory usage."""
    import gc

    feature_table = _get_feature_table(timeframe)
    label_table = _get_label_table(timeframe)

    # Get symbols if not specified
    if symbols:
        symbols_list = symbols
    else:
        symbols_rows = fetch_all(
            "SELECT DISTINCT symbol FROM instruments WHERE status = 'active'"
        )
        symbols_list = [r[0] for r in symbols_rows]

    all_dfs = []
    feature_cols = None

    for symbol in symbols_list:
        query = f"""
            SELECT f.symbol, f.ts, f.features, f.schema_version, f.atr, f.funding_z, f.btc_regime,
                   l.ret_net, l.mae, l.time_to_event_min
            FROM {feature_table} f
            JOIN {label_table} l
              ON f.symbol = l.symbol AND f.ts = l.ts
            WHERE f.schema_version = %s AND l.spec_hash = %s AND f.symbol = %s
        """
        params: list = [feature_schema_version, label_spec_hash, symbol]
        if start_date:
            query += " AND f.ts >= %s"
            params.append(start_date)
        if end_date:
            query += " AND f.ts <= %s"
            params.append(end_date)
        query += " ORDER BY f.ts"

        rows = fetch_all(query, tuple(params))
        if not rows:
            continue

        df = pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "ts",
                "features",
                "schema_version",
                "atr",
                "funding_z",
                "btc_regime",
                "ret_net",
                "mae",
                "time_to_event_min",
            ],
        )

        df, cols = _normalize_features(df)
        if feature_cols is None:
            feature_cols = cols

        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        all_dfs.append(df)

        del rows
        gc.collect()

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    result = result.sort_values("ts")
    result["feature_cols"] = [feature_cols] * len(result)
    return result


def _load_multi_tf_dataset(
    label_spec_hash: str,
    feature_schema_version: int,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: List[str] | None = None,
) -> pd.DataFrame:
    """Load dataset with multi-timeframe features (1m + 15m + 1h)."""
    import gc

    # Get symbols if not specified
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
        # Load 1m features with labels
        query_1m = """
            SELECT f.symbol, f.ts, f.features, f.schema_version, f.atr, f.funding_z, f.btc_regime,
                   l.ret_net, l.mae, l.time_to_event_min
            FROM features_1m f
            JOIN labels_long_1m l
              ON f.symbol = l.symbol AND f.ts = l.ts
            WHERE f.schema_version = %s AND l.spec_hash = %s AND f.symbol = %s
        """
        params: list = [feature_schema_version, label_spec_hash, symbol]
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
                "ret_net", "mae", "time_to_event_min",
            ],
        )
        df_1m["ts"] = pd.to_datetime(df_1m["ts"], utc=True)

        # Normalize 1m features
        df_1m, cols_1m = _normalize_features(df_1m, prefix="f_1m")

        # Load 15m features
        query_15m = """
            SELECT ts, features FROM features_15m WHERE symbol = %s
        """
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

            # Join 15m features to 1m using forward-fill
            df_1m = df_1m.set_index("ts")
            for col in cols_15m:
                df_1m[col] = df_15m[col].reindex(df_1m.index, method="ffill").values
            df_1m = df_1m.reset_index()
        else:
            cols_15m = []

        # Load 1h features
        query_1h = """
            SELECT ts, features FROM features_1h WHERE symbol = %s
        """
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

            # Join 1h features to 1m using forward-fill
            if "ts" in df_1m.columns:
                df_1m = df_1m.set_index("ts")
            for col in cols_1h:
                df_1m[col] = df_1h[col].reindex(df_1m.index, method="ffill").values
            df_1m = df_1m.reset_index()
        else:
            cols_1h = []

        # Combine all feature columns
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


def _train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    objective: str,
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    params: Dict[str, Any] = {
        "objective": objective,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae" if objective == "regression" else "quantile",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def run_training_job(cfg: TrainConfig, symbols: List[str] | None = None) -> Dict[str, Any]:
    if cfg.use_multi_tf:
        df = _load_multi_tf_dataset(
            cfg.label_spec_hash,
            cfg.feature_schema_version,
            start_date=cfg.train_start,
            end_date=cfg.val_end,
            symbols=symbols,
        )
    else:
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

    split_cfg = SplitConfig(purge_bars=cfg.purge_bars, embargo_pct=cfg.embargo_pct)
    splits = list(walk_forward_splits(df["ts"], split_cfg))
    if not splits:
        return {"status": "empty", "message": "Insufficient split data"}
    train_idx, val_idx, test_idx = splits[-1]
    train_df = df[df["ts"].isin(train_idx)]
    val_df = df[df["ts"].isin(val_idx)]
    reg_df = val_df

    if train_df.empty or val_df.empty:
        return {"status": "empty", "message": "Insufficient split data"}

    feature_cols = train_df.iloc[0]["feature_cols"]

    X_train = train_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    X_val = val_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    if X_train.isnull().any().any() or np.isinf(X_train.values).any():
        raise ValueError("X_train contains NaN or inf values after cleanup")
    if X_val.isnull().any().any() or np.isinf(X_val.values).any():
        raise ValueError("X_val contains NaN or inf values after cleanup")

    metrics: Dict[str, Any] = {}
    target_map = {
        "er_long": "ret_net",
        "q05_long": "ret_net",
        "e_mae_long": "mae",
        "e_hold_long": "time_to_event_min",
    }
    models: Dict[str, Any] = {}

    for target in cfg.targets:
        raw_target = target_map.get(target)
        if raw_target is None:
            continue
        objective = "quantile" if target.startswith("q05") else "regression"
        alpha = 0.05 if objective == "quantile" else None
        y_train = train_df[raw_target].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        y_val = val_df[raw_target].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        model = _train_regressor(
            X_train, y_train,
            X_val, y_val,
            objective, alpha
        )
        preds = model.predict(X_val)
        y_val_clean = y_val.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        mse = mean_squared_error(y_val_clean, preds)
        metrics[target] = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_val_clean, preds)),
            "best_iteration": model.best_iteration_ if hasattr(model, 'best_iteration_') else None,
        }
        models[target] = model
        if alpha is not None:
            metrics[target]["pinball_loss"] = _pinball_loss(y_val_clean, preds, alpha)

    metrics["trade"] = _trade_metrics(val_df["ret_net"].fillna(0.0))

    # Symbol-wise metrics
    symbol_metrics: Dict[str, Any] = {}
    for symbol, group in val_df.groupby("symbol"):
        if group.empty:
            continue
        symbol_metrics[str(symbol)] = _trade_metrics(group["ret_net"].fillna(0.0))
    metrics["by_symbol"] = symbol_metrics

    regime_metrics: Dict[str, Any] = {}
    valid_regime_df = reg_df[reg_df["btc_regime"].notna()]
    for regime, group in valid_regime_df.groupby("btc_regime"):
        if group.empty:
            continue
        regime_metrics[str(int(regime))] = _trade_metrics(group["ret_net"].fillna(0.0))

    spec_meta = cfg.label_spec or {}
    cost_breakdown = {
        "fee_per_trade": spec_meta.get("fee_rate", 0.0) * 2,
        "slippage_per_trade": spec_meta.get("slippage_k", 0.0) * 0.0001 * 2,
        "funding_per_trade": 0.0,
    }
    metrics["cost_breakdown"] = cost_breakdown
    metrics["regime"] = regime_metrics

    model_id = uuid.uuid4()
    artifact_name = f"models/{model_id}.pkl"
    artifact_uri = upload_model(models, artifact_name)

    split_summary = {
        "train": _summarize_window(train_idx),
        "val": _summarize_window(val_idx),
        "test": _summarize_window(test_idx),
    }

    report = {
        "model_id": str(model_id),
        "targets": cfg.targets,
        "feature_schema_version": cfg.feature_schema_version,
        "label_spec_hash": cfg.label_spec_hash,
        "label_spec": spec_meta,
        "timeframe": cfg.timeframe,
        "use_multi_tf": cfg.use_multi_tf,
        "split_config": {"purge_bars": cfg.purge_bars, "embargo_pct": cfg.embargo_pct},
        "split_summary": split_summary,
        "metrics": metrics,
        "feature_count": len(feature_cols),
    }

    report_uri = upload_json(report, f"reports/{model_id}.json")

    execute(
        """
        INSERT INTO models (model_id, algo, feature_schema_version, label_spec_hash, train_start, train_end, metrics, artifact_uri, is_production)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            str(model_id),
            cfg.algo,
            cfg.feature_schema_version,
            cfg.label_spec_hash,
            cfg.train_start,
            cfg.train_end,
            json.dumps(metrics),
            artifact_uri,
            False,
        ),
    )

    return {
        "status": "ok",
        "model_id": str(model_id),
        "metrics": metrics,
        "artifact_uri": artifact_uri,
        "report_uri": report_uri,
        "report": report,
        "train_start": cfg.train_start,
        "train_end": cfg.train_end,
        "label_spec_hash": cfg.label_spec_hash,
    }
