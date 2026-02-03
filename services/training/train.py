from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from packages.common.db import execute, fetch_all
from services.registry.storage import upload_json, upload_model
from services.training.splits import SplitConfig, walk_forward_splits


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


def _normalize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    def _parse_features(val: Any) -> Any:
        if isinstance(val, (bytes, bytearray)):
            val = val.decode()
        if isinstance(val, str):
            return json.loads(val)
        return val

    df["features"] = df["features"].apply(_parse_features)
    feats = pd.json_normalize(df["features"])
    feats.columns = [f"f_{c}" for c in feats.columns]

    # Convert all feature columns to float and fill NaN with 0
    for col in feats.columns:
        feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0)

    return pd.concat([df.drop(columns=["features"]), feats], axis=1), feats.columns.tolist()


def _load_dataset(
    label_spec_hash: str,
    feature_schema_version: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load dataset symbol-by-symbol to reduce peak memory usage."""
    import gc

    # First, get list of symbols
    symbols_rows = fetch_all(
        "SELECT DISTINCT symbol FROM instruments WHERE status = 'active'"
    )
    symbols = [r[0] for r in symbols_rows]

    all_dfs = []
    feature_cols = None

    for symbol in symbols:
        # Build query for single symbol
        query = """
            SELECT f.symbol, f.ts, f.features, f.schema_version, f.atr, f.funding_z, f.btc_regime,
                   l.ret_net, l.mae, l.time_to_event_min
            FROM features_1m f
            JOIN labels_long_1m l
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

        # Normalize features for this symbol
        df, cols = _normalize_features(df)
        if feature_cols is None:
            feature_cols = cols

        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        all_dfs.append(df)

        # Force garbage collection after each symbol
        del rows
        gc.collect()

    if not all_dfs:
        return pd.DataFrame()

    # Concatenate all symbol DataFrames
    result = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    result = result.sort_values("ts")
    result["feature_cols"] = [feature_cols] * len(result)
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
        "n_estimators": 500,  # 증가 (Early Stopping으로 조기 종료)
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,  # 로그 숨김
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
            lgb.log_evaluation(period=0),  # 로그 숨김
        ],
    )
    return model


def run_training_job(cfg: TrainConfig) -> Dict[str, Any]:
    df = _load_dataset(
        cfg.label_spec_hash,
        cfg.feature_schema_version,
        start_date=cfg.train_start,
        end_date=cfg.val_end,  # Load only train+val period
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
    reg_df = val_df  # use val for regime slices

    if train_df.empty or val_df.empty:
        return {"status": "empty", "message": "Insufficient split data"}

    feature_cols = train_df.iloc[0]["feature_cols"]

    X_train = train_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    X_val = val_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Validate no NaN/inf in training data
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
        metrics[target] = {
            "rmse": float(mean_squared_error(y_val_clean, preds, squared=False)),
            "mae": float(mean_absolute_error(y_val_clean, preds)),
            "best_iteration": model.best_iteration_ if hasattr(model, 'best_iteration_') else None,
        }
        models[target] = model
        if alpha is not None:
            metrics[target]["pinball_loss"] = _pinball_loss(y_val_clean, preds, alpha)

    metrics["trade"] = _trade_metrics(val_df["ret_net"].fillna(0.0))
    regime_metrics: Dict[str, Any] = {}
    # Filter out NaN btc_regime values before groupby
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
        "split_config": {"purge_bars": cfg.purge_bars, "embargo_pct": cfg.embargo_pct},
        "split_summary": split_summary,
        "metrics": metrics,
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
