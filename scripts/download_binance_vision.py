"""
Binance Vision에서 히스토리 데이터 대량 다운로드
Rate limit 없이 1년치 데이터 수집 가능
"""
from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timedelta, timezone
from typing import List

import httpx
import pandas as pd

from packages.common.db import bulk_upsert, get_conn

# 주요 메인코인 20개
TOP_COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT",
    "XLMUSDT", "FILUSDT", "APTUSDT", "NEARUSDT", "ARBUSDT",
]

BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/daily/klines"
DAYS_TO_FETCH = 365


def download_and_parse_zip(url: str) -> pd.DataFrame | None:
    """ZIP 파일 다운로드 후 CSV 파싱"""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Binance Vision CSV는 헤더가 있음
                    df = pd.read_csv(f)
                    # 컬럼명 통일
                    df = df.rename(columns={
                        "count": "trades",
                        "taker_buy_volume": "taker_buy_base",
                        "taker_buy_quote_volume": "taker_buy_quote"
                    })
                    return df
    except Exception as e:
        print(f"    Error downloading {url}: {e}")
        return None


def process_symbol(symbol: str, start_date: datetime, end_date: datetime) -> int:
    """심볼의 날짜 범위 데이터 수집"""
    total_rows = 0
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{BINANCE_VISION_BASE}/{symbol}/1m/{symbol}-1m-{date_str}.zip"

        df = download_and_parse_zip(url)
        if df is not None and not df.empty:
            rows = []
            for _, row in df.iterrows():
                ts = pd.to_datetime(row["open_time"], unit="ms", utc=True)
                rows.append((
                    symbol,
                    ts,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                ))

            if rows:
                bulk_upsert(
                    "candles_1m",
                    ["symbol", "ts", "open", "high", "low", "close", "volume"],
                    rows,
                    conflict_cols=["symbol", "ts"],
                    update_cols=["open", "high", "low", "close", "volume"],
                )
                total_rows += len(rows)

        current += timedelta(days=1)

    return total_rows


def ensure_instruments(symbols: List[str]) -> None:
    """instruments 테이블에 심볼 등록"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for symbol in symbols:
                cur.execute("""
                    INSERT INTO instruments (symbol, status, liquidity_tier)
                    VALUES (%s, 'active', 1)
                    ON CONFLICT (symbol) DO UPDATE SET status = 'active'
                """, (symbol,))
        conn.commit()


def main() -> None:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)

    print(f"=== Binance Vision 데이터 다운로드 ===")
    print(f"기간: {start_date} ~ {end_date} ({DAYS_TO_FETCH}일)")
    print(f"종목: {len(TOP_COINS)}개")
    print(f"종목 목록: {', '.join(TOP_COINS)}")
    print()

    # 종목 등록
    ensure_instruments(TOP_COINS)
    print("종목 등록 완료")
    print()

    total_candles = 0

    for i, symbol in enumerate(TOP_COINS, 1):
        print(f"[{i}/{len(TOP_COINS)}] {symbol} 처리 중...")
        start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)

        count = process_symbol(symbol, start, end)
        total_candles += count
        print(f"  → {count:,}개 캔들 저장 완료")

    print()
    print(f"=== 다운로드 완료 ===")
    print(f"총 캔들: {total_candles:,}개")

    # 최종 확인
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM candles_1m")
            row = cur.fetchone()
            if row[0]:
                print(f"캔들 기간: {row[0]} ~ {row[1]}")
                print(f"총 레코드: {row[2]:,}개")


if __name__ == "__main__":
    main()
