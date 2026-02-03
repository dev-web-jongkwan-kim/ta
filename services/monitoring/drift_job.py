from __future__ import annotations

import json
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional

import pandas as pd

from packages.common.config import get_settings
from packages.common.db import execute, fetch_all
from packages.common.runtime import set_drift_metrics, set_drift_status
from services.monitoring.drift import compute_psi, compute_missing_rate, compute_latency_ms


DRIFT_FEATURES = ["atr", "funding_z", "ret_1"]


def _fetch_features(symbol: str, schema_version: int, start: datetime, end: datetime) -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT ts, features
        FROM features_1m
        WHERE symbol=%s AND schema_version=%s AND ts BETWEEN %s AND %s
        ORDER BY ts
        """,
        (symbol, schema_version, start, end),
    )
    df = pd.DataFrame(rows, columns=["ts", "features"])
    if df.empty:
        return df
    def normalize(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}

    df["features"] = df["features"].apply(lambda s: normalize(s))
    return df


def _series(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    values = [row.get(key) if isinstance(row, dict) else None for row in df["features"]]
    series = pd.Series(values).dropna()
    if series.empty:
        return None
    return series.astype(float)


def compute_symbol_drift(symbol: str, schema_version: int) -> Optional[Dict[str, float]]:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    current_start = now - pd.Timedelta(hours=settings.drift_current_hours)
    reference_start = now - pd.Timedelta(hours=settings.drift_reference_hours)

    current = _fetch_features(symbol, schema_version, current_start, now)
    reference = _fetch_features(symbol, schema_version, reference_start, current_start)
    if current.empty or reference.empty:
        return None

    psi_values = []
    for key in DRIFT_FEATURES:
        cur_series = _series(current, key)
        ref_series = _series(reference, key)
        if cur_series is None or ref_series is None:
            continue
        psi_values.append(compute_psi(cur_series, ref_series))
    psi = mean(psi_values) if psi_values else 0.0

    missing = compute_missing_rate(pd.DataFrame(list(current["features"]))) if not current.empty else 0.0
    latency = compute_latency_ms(pd.Series(reference["ts"]), pd.Series(current["ts"]))

    outlier_count = _compute_outlier_count(current, reference)
    execute(
        """
        INSERT INTO drift_metrics (ts, symbol, schema_version, psi, missing_rate, latency_ms, outlier_count)
        VALUES (now(), %s, %s, %s, %s, %s, %s)
        """,
        (symbol, schema_version, psi, missing, latency, outlier_count),
    )

    metrics = {
        "symbol": symbol,
        "psi": psi,
        "missing_rate": missing,
        "latency_ms": latency,
        "outlier_count": outlier_count,
    }
    set_drift_metrics(metrics)

    status = _determine_drift_status(settings, psi, missing, latency)
    set_drift_status(status)

    _maybe_log_drift_events(symbol, metrics, status, settings)

    return {**metrics, "status": status}


def _compute_outlier_count(current: pd.DataFrame, reference: pd.DataFrame) -> int:
    def _series_for_key(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
        values = [item.get(key) for item in df["features"] if isinstance(item, dict)]
        series = pd.Series(values).dropna()
        if series.empty:
            return None
        return series.astype(float)

    count = 0
    for key in DRIFT_FEATURES:
        cur_series = _series_for_key(current, key)
        ref_series = _series_for_key(reference, key)
        if cur_series is None or ref_series is None:
            continue
        std = float(ref_series.std())
        if not std or pd.isna(std):
            continue
        z = (cur_series - ref_series.mean()) / std
        count += int((z.abs() > 3).sum())
    return count


def _determine_drift_status(settings: Any, psi: float, missing: float, latency_ms: float) -> str:
    if (
        psi >= settings.drift_block_psi
        or missing >= settings.drift_missing_block_rate
        or latency_ms > settings.drift_latency_threshold_ms
    ):
        return "block"
    if psi >= settings.drift_alert_psi:
        return "alert"
    return "ok"


def _maybe_log_drift_events(symbol: str, metrics: Dict[str, Any], status: str, settings: Any) -> None:
    details = metrics.copy()
    if status == "block":
        execute(
            """
            INSERT INTO risk_events (ts, event_type, symbol, severity, message, details)
            VALUES (now(), 'DRIFT_BLOCK', %s, 3, %s, %s)
            """,
            (symbol, f"DRIFT block (psi={metrics['psi']:.3f})", details),
        )
    elif status == "alert":
        execute(
            """
            INSERT INTO risk_events (ts, event_type, symbol, severity, message, details)
            VALUES (now(), 'DRIFT_ALERT', %s, 2, %s, %s)
            """,
            (symbol, f"DRIFT alert (psi={metrics['psi']:.3f})", details),
        )
    if metrics["missing_rate"] >= settings.drift_missing_block_rate:
        execute(
            """
            INSERT INTO risk_events (ts, event_type, symbol, severity, message, details)
            VALUES (now(), 'MISSING_BLOCK', %s, 3, %s, %s)
            """,
            (symbol, f"Missing rate {metrics['missing_rate']:.3f} >= {settings.drift_missing_block_rate}", details),
        )
    elif metrics["missing_rate"] >= settings.drift_missing_alert_rate:
        execute(
            """
            INSERT INTO risk_events (ts, event_type, symbol, severity, message, details)
            VALUES (now(), 'MISSING_DATA', %s, 2, %s, %s)
            """,
            (symbol, f"Missing rate {metrics['missing_rate']:.3f} >= {settings.drift_missing_alert_rate}", details),
        )
    if metrics["latency_ms"] > settings.drift_latency_threshold_ms:
        execute(
            """
            INSERT INTO risk_events (ts, event_type, symbol, severity, message, details)
            VALUES (now(), 'LATENCY_ALERT', %s, 2, %s, %s)
            """,
            (symbol, f"Latency {metrics['latency_ms']:.0f}ms > {settings.drift_latency_threshold_ms}ms", details),
        )

