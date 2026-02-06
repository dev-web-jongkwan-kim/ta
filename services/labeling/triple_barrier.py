from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    k_tp: float
    k_sl: float
    h_bars: int
    risk_mae_atr: float
    fee_rate: float
    slippage_k: float
    atr_timeframe: str = ""  # ATR 계산용 타임프레임 (예: "15m"). 빈 문자열이면 라벨 타임프레임과 동일

    def hash(self) -> str:
        payload = f"{self.k_tp}-{self.k_sl}-{self.h_bars}-{self.risk_mae_atr}-{self.fee_rate}-{self.slippage_k}-{self.atr_timeframe}"
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _first_hit(mask: np.ndarray) -> np.ndarray:
    """Return index of first True along axis=1; if none, -1."""
    any_hit = mask.any(axis=1)
    idx = mask.argmax(axis=1)
    idx[~any_hit] = -1
    return idx


def label_direction_vectorized(
    symbol: str,
    ts_index: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    mark: np.ndarray,
    atr: np.ndarray,
    funding_rate: np.ndarray,
    next_funding_ts: np.ndarray,
    spec: LabelSpec,
    direction: int,
) -> pd.DataFrame:
    h = spec.h_bars
    n = len(ts_index)
    if n <= h:
        return pd.DataFrame(columns=["symbol", "ts", "spec_hash", "y", "ret_net", "mae", "mfe", "time_to_event_min"])

    entry = mark
    tp = entry + direction * spec.k_tp * atr
    sl = entry - direction * spec.k_sl * atr
    risk_sl = entry - direction * spec.risk_mae_atr * atr

    highs = np.lib.stride_tricks.sliding_window_view(high, h + 1)[:, 1:]
    lows = np.lib.stride_tricks.sliding_window_view(low, h + 1)[:, 1:]
    closes = np.lib.stride_tricks.sliding_window_view(close, h + 1)[:, 1:]

    if direction == 1:
        tp_hit = highs >= tp[: n - h][:, None]
        sl_hit = lows <= sl[: n - h][:, None]
        risk_hit = lows <= risk_sl[: n - h][:, None]
    else:
        tp_hit = lows <= tp[: n - h][:, None]
        sl_hit = highs >= sl[: n - h][:, None]
        risk_hit = highs >= risk_sl[: n - h][:, None]

    tp_idx = _first_hit(tp_hit)
    sl_idx = _first_hit(sl_hit)
    risk_idx = _first_hit(risk_hit)

    time_idx = np.full(n - h, h - 1)

    event_idx = np.where(tp_idx == -1, time_idx, tp_idx)
    event_type = np.where(tp_idx == -1, 0, 1)

    sl_effective = np.where(sl_idx == -1, 1_000_000, sl_idx)
    event_idx = np.minimum(event_idx, sl_effective)
    event_type = np.where(event_idx == sl_effective, -1, event_type)

    risk_effective = np.where(risk_idx == -1, 1_000_000, risk_idx)
    risk_triggered = risk_effective < event_idx
    event_idx = np.where(risk_triggered, risk_effective, event_idx)
    event_type = np.where(risk_triggered, -1, event_type)

    exit_price = closes[np.arange(n - h), event_idx]
    raw_ret = (exit_price - entry[: n - h]) / entry[: n - h]
    raw_ret = raw_ret * direction

    mae = np.where(direction == 1, (entry[: n - h] - lows.min(axis=1)) / entry[: n - h], (highs.max(axis=1) - entry[: n - h]) / entry[: n - h])
    mfe = np.where(direction == 1, (highs.max(axis=1) - entry[: n - h]) / entry[: n - h], (entry[: n - h] - lows.min(axis=1)) / entry[: n - h])

    fee_cost = spec.fee_rate * 2
    slippage_cost = spec.slippage_k * 0.0001 * 2

    hold_min = event_idx + 1
    funding_cost = np.where(hold_min > 0, funding_rate[: n - h], 0) * 0.0

    ret_net = raw_ret - fee_cost - slippage_cost - funding_cost

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "ts": ts_index[: n - h],
            "spec_hash": spec.hash(),
            "y": event_type,
            "ret_net": ret_net,
            "mae": mae,
            "mfe": mfe,
            "time_to_event_min": hold_min,
        }
    )
    return out
