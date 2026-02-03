"""
Binance Vision에서 히스토리 데이터 대량 다운로드
Rate limit 없이 1년치 데이터 수집 가능

Usage:
    python scripts/download_binance_vision.py
    python scripts/download_binance_vision.py --days 30 --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx
import pandas as pd

from packages.common.db import bulk_upsert, get_conn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# 주요 메인코인 20개
DEFAULT_COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT",
    "XLMUSDT", "FILUSDT", "APTUSDT", "NEARUSDT", "ARBUSDT",
]

BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/daily/klines"


def download_and_parse_zip(url: str) -> Optional[pd.DataFrame]:
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
                    df = pd.read_csv(f)
                    return df
    except Exception as e:
        logger.warning(f"Error downloading {url}: {e}")
        return None


def process_symbol_candles(symbol: str, start_date: datetime, end_date: datetime) -> int:
    """심볼의 날짜 범위 캔들 데이터 수집"""
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


def generate_premium_index_from_candles(symbols: List[str]) -> int:
    """캔들 데이터에서 premium_index 생성 (close를 mark_price로 사용)

    과거 데이터의 경우 close ≈ mark_price이므로 이를 활용하여
    premium_index 테이블을 생성합니다.
    """
    total_rows = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for symbol in symbols:
                # candles_1m에서 premium_index로 데이터 복사
                # close를 mark_price로, index_price도 close로 설정
                # funding_rate는 0으로 설정 (히스토리 데이터 없음)
                cur.execute("""
                    INSERT INTO premium_index (symbol, ts, mark_price, index_price, last_funding_rate, next_funding_time)
                    SELECT
                        symbol,
                        ts,
                        close as mark_price,
                        close as index_price,
                        0.0 as last_funding_rate,
                        NULL as next_funding_time
                    FROM candles_1m
                    WHERE symbol = %s
                    ON CONFLICT (symbol, ts) DO UPDATE SET
                        mark_price = EXCLUDED.mark_price,
                        index_price = EXCLUDED.index_price
                """, (symbol,))
                rows_affected = cur.rowcount
                total_rows += rows_affected
                logger.info(f"  {symbol}: premium_index {rows_affected:,}개 생성")
            conn.commit()

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


def get_data_summary() -> dict:
    """현재 DB의 데이터 요약"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # candles
            cur.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM candles_1m")
            candles_row = cur.fetchone()

            # premium_index
            cur.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM premium_index")
            premium_row = cur.fetchone()

            # symbols
            cur.execute("SELECT COUNT(DISTINCT symbol) FROM candles_1m")
            symbol_count = cur.fetchone()[0]

    return {
        "candles": {
            "min_ts": candles_row[0],
            "max_ts": candles_row[1],
            "count": candles_row[2],
        },
        "premium_index": {
            "min_ts": premium_row[0],
            "max_ts": premium_row[1],
            "count": premium_row[2],
        },
        "symbol_count": symbol_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance Vision 데이터 다운로드")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="다운로드할 일수 (기본값: 365)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="다운로드할 심볼 목록 (기본값: 20개 메인코인)",
    )
    parser.add_argument(
        "--skip-premium",
        action="store_true",
        help="premium_index 생성 스킵",
    )
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_COINS
    days_to_fetch = args.days

    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)  # 어제까지
    start_date = end_date - timedelta(days=days_to_fetch)

    logger.info("=" * 60)
    logger.info("Binance Vision 데이터 다운로드")
    logger.info("=" * 60)
    logger.info(f"기간: {start_date} ~ {end_date} ({days_to_fetch}일)")
    logger.info(f"종목: {len(symbols)}개")
    logger.info(f"종목 목록: {', '.join(symbols)}")
    logger.info("")

    # 종목 등록
    ensure_instruments(symbols)
    logger.info("종목 등록 완료")
    logger.info("")

    # 1. 캔들 데이터 다운로드
    logger.info("=== 캔들 데이터 다운로드 ===")
    total_candles = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol} 처리 중...")
        start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)

        count = process_symbol_candles(symbol, start, end)
        total_candles += count
        logger.info(f"  → {count:,}개 캔들 저장 완료")

    logger.info("")
    logger.info(f"캔들 다운로드 완료: 총 {total_candles:,}개")
    logger.info("")

    # 2. premium_index 생성
    if not args.skip_premium:
        logger.info("=== Premium Index 생성 ===")
        logger.info("(close 가격을 mark_price로 사용)")
        total_premium = generate_premium_index_from_candles(symbols)
        logger.info(f"Premium Index 생성 완료: 총 {total_premium:,}개")
        logger.info("")

    # 3. 최종 요약
    logger.info("=" * 60)
    logger.info("다운로드 완료")
    logger.info("=" * 60)

    summary = get_data_summary()
    logger.info(f"심볼 수: {summary['symbol_count']}개")

    if summary["candles"]["min_ts"]:
        logger.info(f"캔들 기간: {summary['candles']['min_ts']} ~ {summary['candles']['max_ts']}")
        logger.info(f"캔들 레코드: {summary['candles']['count']:,}개")

    if summary["premium_index"]["min_ts"]:
        logger.info(f"Premium 기간: {summary['premium_index']['min_ts']} ~ {summary['premium_index']['max_ts']}")
        logger.info(f"Premium 레코드: {summary['premium_index']['count']:,}개")

    # 라벨링/학습 가이드
    logger.info("")
    logger.info("=" * 60)
    logger.info("다음 단계")
    logger.info("=" * 60)
    logger.info("1. 라벨링 실행:")
    logger.info("   python -m services.labeling.run_labeling")
    logger.info("")
    logger.info("2. 학습 실행:")
    logger.info("   python scripts/train_model.py")


if __name__ == "__main__":
    main()
