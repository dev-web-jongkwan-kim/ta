from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class FillModel:
    fee_rate: float
    slippage_k: float

    def slippage(self, notional: float, volume: float) -> float:
        if volume <= 0:
            return self.slippage_k * 0.0001
        return min(0.005, self.slippage_k * (notional / max(volume, 1.0)) * 0.0001)


def simulate_trade_path(
    candles: pd.DataFrame,
    premium: pd.DataFrame,
    entry_ts: str,
    side: int,
    notional: float,
    sl_price: float,
    tp_price: float,
    fill: FillModel,
) -> Dict[str, Any]:
    df = candles.sort_values("ts").reset_index(drop=True)
    entry_idx = df.index[df["ts"] == entry_ts]
    if len(entry_idx) == 0 or entry_idx[0] + 1 >= len(df):
        return {"realized_pnl": 0.0, "fees": 0.0, "slippage": 0.0, "funding": 0.0, "exit_ts": entry_ts, "exit_reason": "NO_ENTRY"}

    idx = int(entry_idx[0] + 1)
    entry_price = float(df.loc[idx, "open"])
    volume = float(df.loc[idx, "volume"])
    slip = fill.slippage(notional, volume)
    entry_price = entry_price * (1 + slip * side)

    mark = premium.sort_values("ts").set_index("ts").reindex(df["ts"], method="ffill")
    for j in range(idx, len(df)):
        mark_price = float(mark.iloc[j]["mark_price"]) if "mark_price" in mark.columns else float(df.loc[j, "close"])
        if side == 1:
            if mark_price <= sl_price:
                exit_price = sl_price * (1 - slip)
                exit_reason = "SL"
                break
            if mark_price >= tp_price:
                exit_price = tp_price * (1 - slip)
                exit_reason = "TP"
                break
        else:
            if mark_price >= sl_price:
                exit_price = sl_price * (1 + slip)
                exit_reason = "SL"
                break
            if mark_price <= tp_price:
                exit_price = tp_price * (1 + slip)
                exit_reason = "TP"
                break
    else:
        exit_price = float(df.loc[len(df) - 1, "close"])
        exit_reason = "TIME"
        j = len(df) - 1

    pnl = (exit_price - entry_price) * side
    fees = notional * fill.fee_rate * 2
    return {
        "realized_pnl": pnl,
        "fees": fees,
        "slippage": notional * slip * 2,
        "funding": 0.0,
        "exit_ts": str(df.loc[j, "ts"]),
        "exit_reason": exit_reason,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_ts": str(df.loc[idx, "ts"]),
        "side": side,
        "notional": notional,
    }
