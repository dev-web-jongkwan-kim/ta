from __future__ import annotations

import time
from typing import List, Tuple

from packages.common.config import get_settings
from packages.common.db import bulk_upsert


class IngestBuffer:
    def __init__(self) -> None:
        settings = get_settings()
        self.flush_interval_sec = settings.flush_interval_sec
        self.flush_batch_size = settings.flush_batch_size
        self._last_flush = time.time()
        self.candles: List[Tuple] = []
        self.candles_15m: List[Tuple] = []
        self.candles_1h: List[Tuple] = []
        self.premium: List[Tuple] = []
        self.open_interest: List[Tuple] = []
        self.long_short_ratio: List[Tuple] = []

    def add_candle(self, row: Tuple) -> None:
        self.candles.append(row)

    def add_candle_15m(self, row: Tuple) -> None:
        self.candles_15m.append(row)

    def add_candle_1h(self, row: Tuple) -> None:
        self.candles_1h.append(row)

    def add_premium(self, row: Tuple) -> None:
        self.premium.append(row)

    def add_open_interest(self, row: Tuple) -> None:
        self.open_interest.append(row)

    def add_long_short_ratio(self, row: Tuple) -> None:
        self.long_short_ratio.append(row)

    def should_flush(self) -> bool:
        total = (
            len(self.candles)
            + len(self.candles_15m)
            + len(self.candles_1h)
            + len(self.premium)
            + len(self.open_interest)
            + len(self.long_short_ratio)
        )
        if total >= self.flush_batch_size:
            return True
        return (time.time() - self._last_flush) >= self.flush_interval_sec

    def flush(self) -> None:
        if self.candles:
            bulk_upsert(
                "candles_1m",
                ["symbol", "ts", "open", "high", "low", "close", "volume"],
                self.candles,
                conflict_cols=["symbol", "ts"],
                update_cols=["open", "high", "low", "close", "volume"],
            )
            self.candles.clear()
        if self.candles_15m:
            bulk_upsert(
                "candles_15m",
                ["symbol", "ts", "open", "high", "low", "close", "volume"],
                self.candles_15m,
                conflict_cols=["symbol", "ts"],
                update_cols=["open", "high", "low", "close", "volume"],
            )
            self.candles_15m.clear()
        if self.candles_1h:
            bulk_upsert(
                "candles_1h",
                ["symbol", "ts", "open", "high", "low", "close", "volume"],
                self.candles_1h,
                conflict_cols=["symbol", "ts"],
                update_cols=["open", "high", "low", "close", "volume"],
            )
            self.candles_1h.clear()
        if self.premium:
            bulk_upsert(
                "premium_index",
                [
                    "symbol",
                    "ts",
                    "mark_price",
                    "index_price",
                    "last_price",
                    "last_funding_rate",
                    "next_funding_time",
                ],
                self.premium,
                conflict_cols=["symbol", "ts"],
                update_cols=[
                    "mark_price",
                    "index_price",
                    "last_price",
                    "last_funding_rate",
                    "next_funding_time",
                ],
            )
            self.premium.clear()
        if self.open_interest:
            bulk_upsert(
                "open_interest",
                ["symbol", "ts", "open_interest", "open_interest_value"],
                self.open_interest,
                conflict_cols=["symbol", "ts"],
                update_cols=["open_interest", "open_interest_value"],
            )
            self.open_interest.clear()
        if self.long_short_ratio:
            bulk_upsert(
                "long_short_ratio",
                [
                    "symbol",
                    "ts",
                    "long_short_ratio",
                    "long_account",
                    "short_account",
                    "top_long_short_ratio",
                    "top_long_account",
                    "top_short_account",
                    "taker_buy_sell_ratio",
                    "taker_buy_vol",
                    "taker_sell_vol",
                ],
                self.long_short_ratio,
                conflict_cols=["symbol", "ts"],
                update_cols=[
                    "long_short_ratio",
                    "long_account",
                    "short_account",
                    "top_long_short_ratio",
                    "top_long_account",
                    "top_short_account",
                    "taker_buy_sell_ratio",
                    "taker_buy_vol",
                    "taker_sell_vol",
                ],
            )
            self.long_short_ratio.clear()
        self._last_flush = time.time()
