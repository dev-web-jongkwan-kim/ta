from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, fetch_all
from services.labeling.triple_barrier import LabelSpec, label_direction_vectorized


@dataclass(frozen=True)
class LabelingConfig:
    k_tp: float = 1.5
    k_sl: float = 1.0
    h_bars: int = 360
    risk_mae_atr: float = 3.0

    def spec(self) -> LabelSpec:
        settings = get_settings()
        return LabelSpec(
            k_tp=self.k_tp,
            k_sl=self.k_sl,
            h_bars=self.h_bars,
            risk_mae_atr=self.risk_mae_atr,
            fee_rate=settings.taker_fee_rate,
            slippage_k=settings.slippage_k,
        )


def _load_candles(symbol: str) -> pd.DataFrame:
    rows = fetch_all(
        "SELECT ts, high, low, close FROM candles_1m WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    return pd.DataFrame(rows, columns=["ts", "high", "low", "close"])


def _load_premium(symbol: str) -> pd.DataFrame:
    rows = fetch_all(
        "SELECT ts, mark_price, last_funding_rate, next_funding_time FROM premium_index WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    return pd.DataFrame(rows, columns=["ts", "mark_price", "funding_rate", "next_funding_time"])


def _persist_label_rows(table: str, rows: List[Tuple]) -> None:
    bulk_upsert(
        table,
        ["symbol", "ts", "spec_hash", "y", "ret_net", "mae", "mfe", "time_to_event_min"],
        rows,
        conflict_cols=["symbol", "ts", "spec_hash"],
        update_cols=["y", "ret_net", "mae", "mfe", "time_to_event_min"],
    )


def label_symbol(symbol: str, spec: LabelSpec) -> None:
    candles = _load_candles(symbol)
    premium = _load_premium(symbol)
    if candles.empty or premium.empty:
        return
    premium = premium.set_index("ts").reindex(candles["ts"], method="ffill").reset_index()
    atr = candles["high"].rolling(window=14).max() - candles["low"].rolling(window=14).min()

    data = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=premium["mark_price"].values,
        atr=atr.fillna(method="bfill").values,
        funding_rate=premium["funding_rate"].fillna(0.0).values,
        next_funding_ts=premium["next_funding_time"].fillna(method="ffill").values,
        spec=spec,
        direction=1,
    )
    rows = [
        (
            r.symbol,
            r.ts,
            r.spec_hash,
            int(r.y),
            float(r.ret_net),
            float(r.mae),
            float(r.mfe),
            int(r.time_to_event_min),
        )
        for r in data.itertuples(index=False)
    ]
    _persist_label_rows("labels_long_1m", rows)

    data_short = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=premium["mark_price"].values,
        atr=atr.fillna(method="bfill").values,
        funding_rate=premium["funding_rate"].fillna(0.0).values,
        next_funding_ts=premium["next_funding_time"].fillna(method="ffill").values,
        spec=spec,
        direction=-1,
    )
    rows_short = [
        (
            r.symbol,
            r.ts,
            r.spec_hash,
            int(r.y),
            float(r.ret_net),
            float(r.mae),
            float(r.mfe),
            int(r.time_to_event_min),
        )
        for r in data_short.itertuples(index=False)
    ]
    _persist_label_rows("labels_short_1m", rows_short)


def run_labeling(config: Optional[LabelingConfig] = None, symbols: Optional[Iterable[str]] = None) -> str:
    settings = get_settings()
    cfg = config or LabelingConfig()
    spec = cfg.spec()
    universe = list(symbols or settings.universe_list())
    for symbol in universe:
        label_symbol(symbol, spec)
    return spec.hash()
