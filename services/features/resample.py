"""캔들 리샘플링: 1분 → 15분/1시간 변환"""
from __future__ import annotations

import logging
from typing import List, Literal

import pandas as pd

from packages.common.db import bulk_upsert, fetch_all, get_conn

logger = logging.getLogger(__name__)

Timeframe = Literal["15m", "1h"]

TIMEFRAME_MINUTES = {
    "15m": 15,
    "1h": 60,
}

TIMEFRAME_RULE = {
    "15m": "15min",
    "1h": "1h",
}


def _load_candles_1m(symbol: str) -> pd.DataFrame:
    """1분 캔들 로드"""
    rows = fetch_all(
        "SELECT ts, open, high, low, close, volume FROM candles_1m WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def resample_candles(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """1분 캔들을 상위 타임프레임으로 리샘플링

    Args:
        df: 1분 캔들 DataFrame (ts, open, high, low, close, volume)
        timeframe: 목표 타임프레임 ("15m" 또는 "1h")

    Returns:
        리샘플링된 DataFrame
    """
    if df.empty:
        return df

    df = df.set_index("ts").sort_index()
    rule = TIMEFRAME_RULE[timeframe]

    resampled = df.resample(rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled.reset_index()


def resample_and_store_symbol(symbol: str, timeframe: Timeframe) -> int:
    """심볼의 1분 캔들을 리샘플링하여 DB에 저장

    Args:
        symbol: 심볼명
        timeframe: 목표 타임프레임

    Returns:
        저장된 행 수
    """
    df_1m = _load_candles_1m(symbol)
    if df_1m.empty:
        logger.warning(f"{symbol}: no 1m candles to resample")
        return 0

    resampled = resample_candles(df_1m, timeframe)
    if resampled.empty:
        return 0

    table = f"candles_{timeframe}"
    rows = [
        (symbol, row.ts, row.open, row.high, row.low, row.close, row.volume)
        for row in resampled.itertuples(index=False)
    ]

    bulk_upsert(
        table,
        ["symbol", "ts", "open", "high", "low", "close", "volume"],
        rows,
        conflict_cols=["symbol", "ts"],
        update_cols=["open", "high", "low", "close", "volume"],
    )

    logger.info(f"{symbol}: {len(rows):,} {timeframe} candles stored")
    return len(rows)


def resample_all_symbols(symbols: List[str], timeframe: Timeframe) -> int:
    """여러 심볼의 캔들을 리샘플링

    Args:
        symbols: 심볼 목록
        timeframe: 목표 타임프레임

    Returns:
        총 저장된 행 수
    """
    total = 0
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Resampling {symbol} to {timeframe}...")
        count = resample_and_store_symbol(symbol, timeframe)
        total += count

    logger.info(f"Total: {total:,} {timeframe} candles stored for {len(symbols)} symbols")
    return total


def get_resampled_candles(symbol: str, timeframe: Timeframe) -> pd.DataFrame:
    """DB에서 리샘플링된 캔들 조회

    Args:
        symbol: 심볼명
        timeframe: 타임프레임

    Returns:
        캔들 DataFrame
    """
    table = f"candles_{timeframe}"
    rows = fetch_all(
        f"SELECT ts, open, high, low, close, volume FROM {table} WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df
