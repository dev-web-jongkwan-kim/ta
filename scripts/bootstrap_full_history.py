"""
대량 히스토리 데이터 수집 스크립트
7-14일치 캔들 + 프리미엄 인덱스 데이터를 Binance에서 가져옴
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import httpx

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, get_conn
from packages.common.symbol_map import to_internal
from packages.common.time import ms_to_dt

BINANCE_BASE = "https://fapi.binance.com"
LIMIT_PER_REQUEST = 1500  # Binance 최대
DAYS_TO_FETCH = 365  # 1년치 데이터


def fetch_klines_paginated(
    symbol: str, interval: str, start_time: int, end_time: int
) -> List[List]:
    """페이지네이션으로 대량 캔들 데이터 수집 (재시도 로직 포함)"""
    all_klines = []
    current_start = start_time
    max_retries = 5

    with httpx.Client(timeout=30) as client:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": LIMIT_PER_REQUEST,
            }

            for retry in range(max_retries):
                try:
                    resp = client.get(f"{BINANCE_BASE}/fapi/v1/klines", params=params)
                    if resp.status_code == 429:
                        wait_time = 60 * (retry + 1)  # 점점 더 오래 대기
                        print(f"  Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        raise
                    print(f"  Retry {retry + 1}/{max_retries}: {e}")
                    time.sleep(10 * (retry + 1))

            if not data:
                break

            all_klines.extend(data)
            # 다음 시작점: 마지막 캔들 시간 + 1분
            current_start = data[-1][0] + 60000
            print(f"  {symbol}: {len(all_klines)} candles fetched...")
            time.sleep(0.5)  # rate limit 준수 (더 긴 딜레이)

    return all_klines


def fetch_funding_history(symbol: str, start_time: int, end_time: int) -> List[dict]:
    """펀딩 레이트 히스토리 수집"""
    all_funding = []
    current_start = start_time

    with httpx.Client(timeout=30) as client:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000,
            }
            resp = client.get(f"{BINANCE_BASE}/fapi/v1/fundingRate", params=params)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_funding.extend(data)
            current_start = data[-1]["fundingTime"] + 1
            time.sleep(0.1)

    return all_funding


def fetch_premium_index_current(symbol: str) -> dict:
    """현재 프리미엄 인덱스 (히스토리 없음, 현재값만)"""
    with httpx.Client(timeout=10) as client:
        params = {"symbol": symbol}
        resp = client.get(f"{BINANCE_BASE}/fapi/v1/premiumIndex", params=params)
        resp.raise_for_status()
        return resp.json()


def main() -> None:
    settings = get_settings()
    symbols = settings.universe_list()

    # 시간 범위 계산
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_TO_FETCH)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    print(f"=== 히스토리 데이터 수집 시작 ===")
    print(f"기간: {start_time.date()} ~ {end_time.date()} ({DAYS_TO_FETCH}일)")
    print(f"종목: {len(symbols)}개")
    print()

    total_candles = 0
    total_funding = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol} 처리 중...")

        # 1. 캔들 데이터 수집
        try:
            klines = fetch_klines_paginated(symbol, "1m", start_ms, end_ms)
            if klines:
                rows = []
                for k in klines:
                    ts = ms_to_dt(k[0])
                    rows.append(
                        (
                            to_internal(symbol, "rest"),
                            ts,
                            float(k[1]),  # open
                            float(k[2]),  # high
                            float(k[3]),  # low
                            float(k[4]),  # close
                            float(k[5]),  # volume
                        )
                    )
                bulk_upsert(
                    "candles_1m",
                    ["symbol", "ts", "open", "high", "low", "close", "volume"],
                    rows,
                    conflict_cols=["symbol", "ts"],
                    update_cols=["open", "high", "low", "close", "volume"],
                )
                total_candles += len(rows)
                print(f"  → 캔들 {len(rows)}개 저장 완료")
        except Exception as e:
            print(f"  ✗ 캔들 수집 실패: {e}")

        # 2. 펀딩 레이트 히스토리 수집
        try:
            funding = fetch_funding_history(symbol, start_ms, end_ms)
            if funding:
                rows = []
                for f in funding:
                    ts = ms_to_dt(f["fundingTime"])
                    rows.append(
                        (
                            to_internal(symbol, "rest"),
                            ts,
                            float(f["fundingRate"]),
                        )
                    )
                bulk_upsert(
                    "funding_rates",
                    ["symbol", "ts", "funding_rate"],
                    rows,
                    conflict_cols=["symbol", "ts"],
                    update_cols=["funding_rate"],
                )
                total_funding += len(rows)
                print(f"  → 펀딩 {len(rows)}개 저장 완료")
        except Exception as e:
            print(f"  ✗ 펀딩 수집 실패: {e}")

        # 3. 현재 프리미엄 인덱스 저장 (mark price 등)
        try:
            pi = fetch_premium_index_current(symbol)
            ts = ms_to_dt(pi["time"])
            row = (
                to_internal(symbol, "rest"),
                ts,
                float(pi["markPrice"]),
                float(pi["indexPrice"]),
                float(pi["lastFundingRate"]),
                ms_to_dt(pi["nextFundingTime"]) if pi.get("nextFundingTime") else None,
            )
            bulk_upsert(
                "premium_index",
                ["symbol", "ts", "mark_price", "index_price", "funding_rate", "next_funding_time"],
                [row],
                conflict_cols=["symbol", "ts"],
                update_cols=["mark_price", "index_price", "funding_rate", "next_funding_time"],
            )
        except Exception as e:
            print(f"  ✗ 프리미엄 인덱스 저장 실패: {e}")

        print()

    print("=== 수집 완료 ===")
    print(f"총 캔들: {total_candles:,}개")
    print(f"총 펀딩: {total_funding:,}개")

    # 최종 확인
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(ts), MAX(ts) FROM candles_1m")
            row = cur.fetchone()
            if row[0]:
                print(f"캔들 기간: {row[0]} ~ {row[1]}")


if __name__ == "__main__":
    main()
