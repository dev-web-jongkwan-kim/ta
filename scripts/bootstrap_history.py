from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import httpx

from packages.common.config import get_settings
from packages.common.db import bulk_upsert
from packages.common.symbol_map import to_internal
from packages.common.time import ms_to_dt

BINANCE_BASE = "https://fapi.binance.com"


def fetch_klines(symbol: str, interval: str, limit: int) -> List[List]:
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def main() -> None:
    settings = get_settings()
    symbols = settings.universe_list()

    for symbol in symbols:
        klines = fetch_klines(symbol, "1m", 1000)
        rows = []
        for k in klines:
            ts = ms_to_dt(k[0])
            rows.append(
                (
                    to_internal(symbol, "rest"),
                    ts,
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                )
            )
        bulk_upsert(
            "candles_1m",
            ["symbol", "ts", "open", "high", "low", "close", "volume"],
            rows,
            conflict_cols=["symbol", "ts"],
            update_cols=["open", "high", "low", "close", "volume"],
        )
        print(f"Bootstrapped {len(rows)} candles for {symbol}")


if __name__ == "__main__":
    main()
