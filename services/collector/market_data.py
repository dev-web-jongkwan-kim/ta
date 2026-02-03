from __future__ import annotations

from typing import Dict, List, Optional, Any

import httpx

from packages.common.symbol_map import to_internal, to_rest

BINANCE_BASE = "https://fapi.binance.com"


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 2) -> List[List]:
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    params = {"symbol": to_rest(symbol), "interval": interval, "limit": limit}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_premium_index() -> List[Dict]:
    url = f"{BINANCE_BASE}/fapi/v1/premiumIndex"
    with httpx.Client(timeout=10) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return [data]
        return data


def fetch_funding_rates(symbol: str, limit: int = 100) -> List[Dict]:
    url = f"{BINANCE_BASE}/fapi/v1/fundingRate"
    params = {"symbol": to_rest(symbol), "limit": limit}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_open_interest(symbol: str) -> Optional[Dict]:
    """Fetch current open interest for a symbol."""
    url = f"{BINANCE_BASE}/fapi/v1/openInterest"
    params = {"symbol": to_rest(symbol)}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_open_interest_hist(symbol: str, period: str = "5m", limit: int = 100) -> List[Dict]:
    """Fetch historical open interest data."""
    url = f"{BINANCE_BASE}/futures/data/openInterestHist"
    params = {"symbol": to_rest(symbol), "period": period, "limit": limit}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_long_short_ratio(symbol: str, period: str = "5m", limit: int = 100) -> List[Dict]:
    """Fetch global long/short account ratio."""
    url = f"{BINANCE_BASE}/futures/data/globalLongShortAccountRatio"
    params = {"symbol": to_rest(symbol), "period": period, "limit": limit}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_top_long_short_ratio(symbol: str, period: str = "5m", limit: int = 100) -> List[Dict]:
    """Fetch top trader long/short position ratio."""
    url = f"{BINANCE_BASE}/futures/data/topLongShortPositionRatio"
    params = {"symbol": to_rest(symbol), "period": period, "limit": limit}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_taker_buy_sell_ratio(symbol: str, period: str = "5m", limit: int = 100) -> List[Dict]:
    """Fetch taker buy/sell volume ratio."""
    url = f"{BINANCE_BASE}/futures/data/takerlongshortRatio"
    params = {"symbol": to_rest(symbol), "period": period, "limit": limit}
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
