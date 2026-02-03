from __future__ import annotations

from typing import Dict, List, Tuple

import httpx

from packages.common.db import bulk_upsert, execute, fetch_all
from packages.common.symbol_map import to_internal

BINANCE_BASE = "https://fapi.binance.com"


def _fetch_exchange_info() -> List[Dict]:
    url = f"{BINANCE_BASE}/fapi/v1/exchangeInfo"
    with httpx.Client(timeout=20) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json().get("symbols", [])


def _fetch_ticker_24h() -> Dict[str, float]:
    url = f"{BINANCE_BASE}/fapi/v1/ticker/24hr"
    with httpx.Client(timeout=20) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    volumes: Dict[str, float] = {}
    for item in data:
        symbol = to_internal(item.get("symbol", ""), "rest")
        try:
            volumes[symbol] = float(item.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            volumes[symbol] = 0.0
    return volumes


def build_universe(top_n: int = 50) -> List[str]:
    info = _fetch_exchange_info()
    volumes = _fetch_ticker_24h()

    eligible: List[Tuple[str, float]] = []
    for item in info:
        if item.get("contractType") != "PERPETUAL":
            continue
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        symbol = to_internal(item.get("symbol", ""), "rest")
        eligible.append((symbol, volumes.get(symbol, 0.0)))

    eligible.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in eligible[:top_n]]

    rows = [(s, "active", "A") for s in top_symbols]
    bulk_upsert(
        "instruments",
        ["symbol", "status", "liquidity_tier"],
        rows,
        conflict_cols=["symbol"],
        update_cols=["status", "liquidity_tier"],
    )

    if top_symbols:
        execute(
            "UPDATE instruments SET status='inactive' WHERE symbol NOT IN %s",
            (tuple(top_symbols),),
        )

    return top_symbols


if __name__ == "__main__":
    symbols = build_universe(top_n=50)
    print(f"Universe built: {len(symbols)} symbols")
