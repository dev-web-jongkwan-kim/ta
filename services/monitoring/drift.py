from __future__ import annotations

import numpy as np
import pandas as pd


def compute_psi(current: pd.Series, reference: pd.Series, buckets: int = 10) -> float:
    current = current.dropna()
    reference = reference.dropna()
    if current.empty or reference.empty:
        return 0.0
    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(reference.quantile(quantiles).values)
    if len(cuts) < 2:
        return 0.0
    current_bins = pd.cut(current, bins=cuts, include_lowest=True)
    ref_bins = pd.cut(reference, bins=cuts, include_lowest=True)
    current_dist = current_bins.value_counts(normalize=True)
    ref_dist = ref_bins.value_counts(normalize=True)
    psi = ((current_dist - ref_dist) * np.log((current_dist + 1e-9) / (ref_dist + 1e-9))).sum()
    return float(psi)


def compute_missing_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df.isna().mean().mean())


def compute_latency_ms(expected_ts: pd.Series, observed_ts: pd.Series) -> float:
    if expected_ts.empty or observed_ts.empty:
        return 0.0
    lag = (observed_ts - expected_ts).dt.total_seconds() * 1000
    return float(lag.mean())
