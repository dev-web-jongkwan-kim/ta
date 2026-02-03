"""
OI/Long-Short Ratio 과거 데이터 백필
바이낸스 API에서 최대 30일치 가져오기
"""
import os
import sys
import time
from datetime import datetime, timedelta

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.common.db import get_conn

BASE_URL = "https://fapi.binance.com"

# 30개 고정 유니버스
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT",
    "SUIUSDT", "AAVEUSDT", "1000PEPEUSDT", "FILUSDT", "AXSUSDT", "ENAUSDT",
    "ZKUSDT", "XAUUSDT", "XAGUSDT", "HYPEUSDT", "ZECUSDT", "BNXUSDT",
    "ALPHAUSDT", "ZORAUSDT", "TRUMPUSDT", "PUMPUSDT", "NEARUSDT", "PAXGUSDT"
]


def fetch_open_interest_hist(symbol: str, period: str = "5m", limit: int = 500) -> list:
    """Open Interest 히스토리 가져오기"""
    url = f"{BASE_URL}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}

    try:
        resp = httpx.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Error fetching OI for {symbol}: {e}")
    return []


def fetch_long_short_ratio(symbol: str, period: str = "5m", limit: int = 500) -> list:
    """Global Long/Short Ratio 히스토리"""
    url = f"{BASE_URL}/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol, "period": period, "limit": limit}

    try:
        resp = httpx.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Error fetching LS ratio for {symbol}: {e}")
    return []


def fetch_top_long_short_ratio(symbol: str, period: str = "5m", limit: int = 500) -> list:
    """Top Trader Long/Short Ratio 히스토리"""
    url = f"{BASE_URL}/futures/data/topLongShortAccountRatio"
    params = {"symbol": symbol, "period": period, "limit": limit}

    try:
        resp = httpx.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Error fetching top LS ratio for {symbol}: {e}")
    return []


def fetch_taker_buy_sell_ratio(symbol: str, period: str = "5m", limit: int = 500) -> list:
    """Taker Buy/Sell Volume Ratio 히스토리"""
    url = f"{BASE_URL}/futures/data/takerlongshortRatio"
    params = {"symbol": symbol, "period": period, "limit": limit}

    try:
        resp = httpx.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Error fetching taker ratio for {symbol}: {e}")
    return []


def save_open_interest(conn, symbol: str, data: list):
    """OI 데이터 저장"""
    if not data:
        return 0

    rows = []
    for item in data:
        rows.append({
            "symbol": symbol,
            "ts": datetime.fromtimestamp(item["timestamp"] / 1000),
            "open_interest": float(item["sumOpenInterest"]),
            "open_interest_value": float(item["sumOpenInterestValue"]),
        })

    df = pd.DataFrame(rows)

    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO open_interest (symbol, ts, open_interest, open_interest_value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol, ts) DO NOTHING
            """, (row["symbol"], row["ts"], row["open_interest"], row["open_interest_value"]))
    conn.commit()
    return len(rows)


def save_long_short_ratio(conn, symbol: str, ls_data: list, top_ls_data: list, taker_data: list):
    """Long/Short Ratio 데이터 저장"""
    # 타임스탬프 기준으로 데이터 병합
    merged = {}

    for item in ls_data:
        ts = item["timestamp"]
        if ts not in merged:
            merged[ts] = {"symbol": symbol, "ts": datetime.fromtimestamp(ts / 1000)}
        merged[ts]["long_short_ratio"] = float(item["longShortRatio"])
        merged[ts]["long_account"] = float(item["longAccount"])
        merged[ts]["short_account"] = float(item["shortAccount"])

    for item in top_ls_data:
        ts = item["timestamp"]
        if ts not in merged:
            merged[ts] = {"symbol": symbol, "ts": datetime.fromtimestamp(ts / 1000)}
        merged[ts]["top_long_short_ratio"] = float(item["longShortRatio"])
        merged[ts]["top_long_account"] = float(item["longAccount"])
        merged[ts]["top_short_account"] = float(item["shortAccount"])

    for item in taker_data:
        ts = item["timestamp"]
        if ts not in merged:
            merged[ts] = {"symbol": symbol, "ts": datetime.fromtimestamp(ts / 1000)}
        merged[ts]["taker_buy_sell_ratio"] = float(item["buySellRatio"])
        merged[ts]["taker_buy_vol"] = float(item["buyVol"])
        merged[ts]["taker_sell_vol"] = float(item["sellVol"])

    if not merged:
        return 0

    with conn.cursor() as cur:
        for ts, row in merged.items():
            cur.execute("""
                INSERT INTO long_short_ratio (
                    symbol, ts, long_short_ratio, long_account, short_account,
                    top_long_short_ratio, top_long_account, top_short_account,
                    taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    long_short_ratio = COALESCE(EXCLUDED.long_short_ratio, long_short_ratio.long_short_ratio),
                    top_long_short_ratio = COALESCE(EXCLUDED.top_long_short_ratio, long_short_ratio.top_long_short_ratio),
                    taker_buy_sell_ratio = COALESCE(EXCLUDED.taker_buy_sell_ratio, long_short_ratio.taker_buy_sell_ratio)
            """, (
                row.get("symbol"),
                row.get("ts"),
                row.get("long_short_ratio"),
                row.get("long_account"),
                row.get("short_account"),
                row.get("top_long_short_ratio"),
                row.get("top_long_account"),
                row.get("top_short_account"),
                row.get("taker_buy_sell_ratio"),
                row.get("taker_buy_vol"),
                row.get("taker_sell_vol"),
            ))
    conn.commit()
    return len(merged)


def main():
    print("=" * 60)
    print("OI/Long-Short Ratio 과거 데이터 백필")
    print("=" * 60)

    total_oi = 0
    total_ls = 0

    with get_conn() as conn:
        for i, symbol in enumerate(SYMBOLS):
            print(f"\n[{i+1}/{len(SYMBOLS)}] {symbol}")

            # OI 히스토리
            print("  Fetching Open Interest...")
            oi_data = fetch_open_interest_hist(symbol, period="5m", limit=500)
            oi_count = save_open_interest(conn, symbol, oi_data)
            total_oi += oi_count
            print(f"    → {oi_count} rows")

            time.sleep(0.2)  # Rate limit

            # Long/Short Ratio
            print("  Fetching Long/Short Ratio...")
            ls_data = fetch_long_short_ratio(symbol, period="5m", limit=500)
            time.sleep(0.2)

            top_ls_data = fetch_top_long_short_ratio(symbol, period="5m", limit=500)
            time.sleep(0.2)

            taker_data = fetch_taker_buy_sell_ratio(symbol, period="5m", limit=500)
            time.sleep(0.2)

            ls_count = save_long_short_ratio(conn, symbol, ls_data, top_ls_data, taker_data)
            total_ls += ls_count
            print(f"    → {ls_count} rows")

    print("\n" + "=" * 60)
    print(f"완료!")
    print(f"  Open Interest: {total_oi:,} rows")
    print(f"  Long/Short Ratio: {total_ls:,} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
