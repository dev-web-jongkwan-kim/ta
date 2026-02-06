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


def _get_label_table(timeframe: Timeframe, direction: str = "long") -> str:
    return f"labels_{direction}_{timeframe}"


def _load_dataset(
    label_spec_hash: str,
    feature_schema_version: int,
    start_date: str | None = None,
    end_date: str | None = None,
    timeframe: Timeframe = "1m",
    symbols: List[str] | None = None,
    include_short: bool = True,
) -> pd.DataFrame:
    """Load dataset symbol-by-symbol to reduce peak memory usage.

    Optimized: loads tables separately and merges in pandas (faster than SQL JOIN).

    Args:
        include_short: If True, also load short labels and include them as separate columns.
    """
    import gc
    import logging

    logger = logging.getLogger(__name__)

    feature_table = _get_feature_table(timeframe)
    label_table_long = _get_label_table(timeframe, "long")
    label_table_short = _get_label_table(timeframe, "short")

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

    # Build date filter
    date_filter = ""
    date_params: list = []
    if start_date:
        date_filter += " AND ts >= %s"
        date_params.append(start_date)
    if end_date:
        date_filter += " AND ts <= %s"
        date_params.append(end_date)

    total_symbols = len(symbols_list)
    for idx, symbol in enumerate(symbols_list, 1):
        logger.info(f"[{idx}/{total_symbols}] {symbol} 데이터 로딩...")

        # 1. Load features (no JOIN)
        query_features = f"""
            SELECT symbol, ts, features, schema_version, atr, funding_z, btc_regime
            FROM {feature_table}
            WHERE symbol = %s AND schema_version = %s {date_filter}
            ORDER BY ts
        """
        params_features = [symbol, feature_schema_version] + date_params
        rows_features = fetch_all(query_features, tuple(params_features))
        if not rows_features:
            continue

        df_features = pd.DataFrame(
            rows_features,
            columns=["symbol", "ts", "features", "schema_version", "atr", "funding_z", "btc_regime"],
        )
        df_features["ts"] = pd.to_datetime(df_features["ts"], utc=True)

        # 2. Load long labels (no JOIN)
        query_long = f"""
            SELECT ts, ret_net, mae, time_to_event_min
            FROM {label_table_long}
            WHERE symbol = %s AND spec_hash = %s {date_filter}
            ORDER BY ts
        """
        params_long = [symbol, label_spec_hash] + date_params
        rows_long = fetch_all(query_long, tuple(params_long))
        if not rows_long:
            continue

        df_long = pd.DataFrame(
            rows_long,
            columns=["ts", "ret_net_long", "mae_long", "time_to_event_min_long"],
        )
        df_long["ts"] = pd.to_datetime(df_long["ts"], utc=True)

        # 3. Load short labels if needed (no JOIN)
        df_short = None
        if include_short:
            query_short = f"""
                SELECT ts, ret_net, mae, time_to_event_min
                FROM {label_table_short}
                WHERE symbol = %s AND spec_hash = %s {date_filter}
                ORDER BY ts
            """
            params_short = [symbol, label_spec_hash] + date_params
            rows_short = fetch_all(query_short, tuple(params_short))
            if rows_short:
                df_short = pd.DataFrame(
                    rows_short,
                    columns=["ts", "ret_net_short", "mae_short", "time_to_event_min_short"],
                )
                df_short["ts"] = pd.to_datetime(df_short["ts"], utc=True)

        # 4. Merge in pandas (much faster than SQL JOIN)
        df = df_features.merge(df_long, on="ts", how="inner")
        if df_short is not None:
            df = df.merge(df_short, on="ts", how="inner")
        else:
            # For backward compatibility when not including short
            df["ret_net"] = df["ret_net_long"]
            df["mae"] = df["mae_long"]
            df["time_to_event_min"] = df["time_to_event_min_long"]

        del rows_features, rows_long, df_features, df_long
        if df_short is not None:
            del rows_short, df_short
        gc.collect()

        if df.empty:
            continue

        df, cols = _normalize_features(df)
        if feature_cols is None:
            feature_cols = cols

        all_dfs.append(df)
        logger.info(f"  → {symbol}: {len(df):,}행")

        gc.collect()

    if not all_dfs:
        return pd.DataFrame()

    logger.info(f"전체 {len(all_dfs)}개 심볼 데이터 병합 중...")
    result = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    result = result.sort_values("ts")
    result["feature_cols"] = [feature_cols] * len(result)
    logger.info(f"데이터 로딩 완료: 총 {len(result):,}행")
    return result


def _load_multi_tf_dataset(
    label_spec_hash: str,
    feature_schema_version: int,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: List[str] | None = None,
    base_timeframe: Timeframe = "1m",
    include_short: bool = True,
) -> pd.DataFrame:
    """Load dataset with multi-timeframe features (1m + 15m + 1h).

    The base timeframe controls which labels/features define the rows.
    Other timeframes are forward-filled onto the base timestamps.

    Optimized: loads tables separately and merges in pandas (faster than SQL JOIN).

    Args:
        include_short: If True, also load short labels and include them as separate columns.
    """
    import gc
    import logging

    logger = logging.getLogger(__name__)

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

    total_symbols = len(symbols_list)
    for idx, symbol in enumerate(symbols_list, 1):
        logger.info(f"[{idx}/{total_symbols}] {symbol} 데이터 로딩 시작...")

        base_feature_table = _get_feature_table(base_timeframe)
        base_label_table_long = _get_label_table(base_timeframe, "long")
        base_label_table_short = _get_label_table(base_timeframe, "short")
        prefix_base = f"f_{base_timeframe}"

        # Build date filter
        date_filter = ""
        date_params: list = []
        if start_date:
            date_filter += " AND ts >= %s"
            date_params.append(start_date)
        if end_date:
            date_filter += " AND ts <= %s"
            date_params.append(end_date)

        # 1. Load features (no JOIN)
        query_features = f"""
            SELECT symbol, ts, features, schema_version, atr, funding_z, btc_regime
            FROM {base_feature_table}
            WHERE symbol = %s AND schema_version = %s {date_filter}
            ORDER BY ts
        """
        params_features = [symbol, feature_schema_version] + date_params
        rows_features = fetch_all(query_features, tuple(params_features))
        if not rows_features:
            logger.info(f"  → {symbol}: features 없음, 건너뜀")
            continue

        df_features = pd.DataFrame(
            rows_features,
            columns=["symbol", "ts", "features", "schema_version", "atr", "funding_z", "btc_regime"],
        )
        df_features["ts"] = pd.to_datetime(df_features["ts"], utc=True)
        logger.info(f"  → features: {len(df_features):,}행")

        # 2. Load long labels (no JOIN)
        query_long = f"""
            SELECT ts, ret_net, mae, time_to_event_min
            FROM {base_label_table_long}
            WHERE symbol = %s AND spec_hash = %s {date_filter}
            ORDER BY ts
        """
        params_long = [symbol, label_spec_hash] + date_params
        rows_long = fetch_all(query_long, tuple(params_long))
        if not rows_long:
            logger.info(f"  → {symbol}: long labels 없음, 건너뜀")
            continue

        df_long = pd.DataFrame(
            rows_long,
            columns=["ts", "ret_net_long", "mae_long", "time_to_event_min_long"],
        )
        df_long["ts"] = pd.to_datetime(df_long["ts"], utc=True)
        logger.info(f"  → long labels: {len(df_long):,}행")

        # 3. Load short labels if needed (no JOIN)
        df_short = None
        if include_short:
            query_short = f"""
                SELECT ts, ret_net, mae, time_to_event_min
                FROM {base_label_table_short}
                WHERE symbol = %s AND spec_hash = %s {date_filter}
                ORDER BY ts
            """
            params_short = [symbol, label_spec_hash] + date_params
            rows_short = fetch_all(query_short, tuple(params_short))
            if rows_short:
                df_short = pd.DataFrame(
                    rows_short,
                    columns=["ts", "ret_net_short", "mae_short", "time_to_event_min_short"],
                )
                df_short["ts"] = pd.to_datetime(df_short["ts"], utc=True)
                logger.info(f"  → short labels: {len(df_short):,}행")

        # 4. Merge in pandas (much faster than SQL JOIN)
        df_base = df_features.merge(df_long, on="ts", how="inner")
        if df_short is not None:
            df_base = df_base.merge(df_short, on="ts", how="inner")
        logger.info(f"  → merge 후: {len(df_base):,}행")

        del rows_features, rows_long, df_features, df_long
        if df_short is not None:
            del rows_short, df_short
        gc.collect()

        if df_base.empty:
            continue

        # Normalize base features
        df_base, cols_base = _normalize_features(df_base, prefix=prefix_base)

        # Join other timeframes' features onto base timestamps
        cols_other: list[str] = []
        for tf in ["1m", "15m", "1h"]:
            if tf == base_timeframe:
                continue

            query_tf = """
                SELECT ts, features FROM features_{tf} WHERE symbol = %s
            """.format(tf=tf)
            params_tf = [symbol]
            if start_date:
                query_tf += " AND ts >= %s"
                params_tf.append(start_date)
            if end_date:
                query_tf += " AND ts <= %s"
                params_tf.append(end_date)
            query_tf += " ORDER BY ts"

            rows_tf = fetch_all(query_tf, tuple(params_tf))
            if not rows_tf:
                continue

            df_tf = pd.DataFrame(rows_tf, columns=["ts", "features"])
            df_tf["ts"] = pd.to_datetime(df_tf["ts"], utc=True)
            df_tf, cols_tf = _normalize_features(df_tf, prefix=f"f_{tf}")
            df_tf = df_tf.set_index("ts")

            df_base = df_base.set_index("ts")
            for col in cols_tf:
                df_base[col] = df_tf[col].reindex(df_base.index, method="ffill").values
            df_base = df_base.reset_index()

            cols_other += cols_tf
            logger.info(f"  → {tf} features 병합 완료")

        # Combine all feature columns
        all_cols = cols_base + cols_other
        if all_feature_cols is None:
            all_feature_cols = all_cols

        df_base["feature_cols"] = [all_cols] * len(df_base)
        all_dfs.append(df_base)
        logger.info(f"  → {symbol} 완료: {len(df_base):,}행, {len(all_cols)}개 피처")

        gc.collect()

    if not all_dfs:
        return pd.DataFrame()

    logger.info(f"전체 {len(all_dfs)}개 심볼 데이터 병합 중...")
    result = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    result = result.sort_values("ts")
    logger.info(f"데이터 로딩 완료: 총 {len(result):,}행")
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
    # Check if any short targets are requested
    include_short = any(t.endswith("_short") for t in cfg.targets)

    if cfg.use_multi_tf:
        df = _load_multi_tf_dataset(
            cfg.label_spec_hash,
            cfg.feature_schema_version,
            start_date=cfg.train_start,
            end_date=cfg.val_end,
            symbols=symbols,
            base_timeframe=cfg.timeframe,
            include_short=include_short,
        )
    else:
        df = _load_dataset(
            cfg.label_spec_hash,
            cfg.feature_schema_version,
            start_date=cfg.train_start,
            end_date=cfg.val_end,
            timeframe=cfg.timeframe,
            symbols=symbols,
            include_short=include_short,
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
    # Check if we have short labels
    has_short = "ret_net_short" in train_df.columns

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
    # Backward compatibility: map old column names if they exist
    if "ret_net" in train_df.columns and "ret_net_long" not in train_df.columns:
        target_map["er_long"] = "ret_net"
        target_map["q05_long"] = "ret_net"
        target_map["e_mae_long"] = "mae"
        target_map["e_hold_long"] = "time_to_event_min"

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

    # Trade metrics for long (use ret_net_long if available, otherwise ret_net for backward compat)
    ret_col_long = "ret_net_long" if "ret_net_long" in val_df.columns else "ret_net"
    metrics["trade_long"] = _trade_metrics(val_df[ret_col_long].fillna(0.0))
    metrics["trade"] = metrics["trade_long"]  # Backward compatibility

    # Trade metrics for short if available
    if "ret_net_short" in val_df.columns:
        metrics["trade_short"] = _trade_metrics(val_df["ret_net_short"].fillna(0.0))

    # Filtered metrics using model predictions - LONG
    if "er_long" in models and "er_long" in cfg.targets:
        er_preds_long = models["er_long"].predict(X_val)
        val_df_copy = val_df.copy()
        val_df_copy["pred_er_long"] = er_preds_long

        filtered_metrics_long = {}
        for threshold in [0, 0.0005, 0.001, 0.0015, 0.002]:
            mask = val_df_copy["pred_er_long"] > threshold
            filtered_rets = val_df_copy.loc[mask, ret_col_long].fillna(0.0)
            if len(filtered_rets) > 0:
                m = _trade_metrics(filtered_rets)
                filtered_metrics_long[f"er>{threshold}"] = m
        metrics["filtered_long"] = filtered_metrics_long
        metrics["filtered"] = filtered_metrics_long  # Backward compatibility

    # Filtered metrics using model predictions - SHORT
    if "er_short" in models and "er_short" in cfg.targets and "ret_net_short" in val_df.columns:
        er_preds_short = models["er_short"].predict(X_val)
        val_df_copy = val_df.copy()
        val_df_copy["pred_er_short"] = er_preds_short

        filtered_metrics_short = {}
        for threshold in [0, 0.0005, 0.001, 0.0015, 0.002]:
            mask = val_df_copy["pred_er_short"] > threshold
            filtered_rets = val_df_copy.loc[mask, "ret_net_short"].fillna(0.0)
            if len(filtered_rets) > 0:
                m = _trade_metrics(filtered_rets)
                filtered_metrics_short[f"er>{threshold}"] = m
        metrics["filtered_short"] = filtered_metrics_short

    # Symbol-wise metrics
    symbol_metrics: Dict[str, Any] = {}
    for symbol, group in val_df.groupby("symbol"):
        if group.empty:
            continue
        symbol_metrics[str(symbol)] = _trade_metrics(group[ret_col_long].fillna(0.0))
    metrics["by_symbol"] = symbol_metrics

    regime_metrics: Dict[str, Any] = {}
    valid_regime_df = reg_df[reg_df["btc_regime"].notna()]
    for regime, group in valid_regime_df.groupby("btc_regime"):
        if group.empty:
            continue
        regime_metrics[str(int(regime))] = _trade_metrics(group[ret_col_long].fillna(0.0))

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
