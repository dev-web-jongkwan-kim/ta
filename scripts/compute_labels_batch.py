"""
메모리 효율적인 라벨 생성 (종목별 순차 처리)
"""
from __future__ import annotations

import gc
from typing import List, Tuple

import pandas as pd

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, fetch_all
from services.labeling.triple_barrier import LabelSpec, label_direction_vectorized


def persist_label_rows(table: str, rows: List[Tuple]) -> None:
    if not rows:
        return
    # 배치로 분할하여 저장
    batch_size = 50000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        bulk_upsert(
            table,
            ["symbol", "ts", "spec_hash", "y", "ret_net", "mae", "mfe", "time_to_event_min"],
            batch,
            conflict_cols=["symbol", "ts", "spec_hash"],
            update_cols=["y", "ret_net", "mae", "mfe", "time_to_event_min"],
        )


def label_single_symbol(symbol: str, spec: LabelSpec) -> int:
    """단일 종목 라벨링 (메모리 효율)"""
    # 캔들 로드
    rows = fetch_all(
        "SELECT ts, high, low, close FROM candles_1m WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    if not rows:
        return 0

    candles = pd.DataFrame(rows, columns=["ts", "high", "low", "close"])
    if len(candles) < 360:  # h_bars보다 작으면 스킵
        return 0

    # ATR 계산
    atr = candles["high"].rolling(window=14).max() - candles["low"].rolling(window=14).min()
    atr = atr.bfill().values

    # close를 mark로 사용
    mark = candles["close"].values
    funding_rate = [0.0] * len(candles)
    next_funding_ts = candles["ts"].values

    # Long 라벨
    data_long = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=mark,
        atr=atr,
        funding_rate=funding_rate,
        next_funding_ts=next_funding_ts,
        spec=spec,
        direction=1,
    )

    rows_long = [
        (r.symbol, r.ts, r.spec_hash, int(r.y), float(r.ret_net), float(r.mae), float(r.mfe), int(r.time_to_event_min))
        for r in data_long.itertuples(index=False)
    ]
    long_count = len(rows_long)
    persist_label_rows("labels_long_1m", rows_long)
    del data_long, rows_long
    gc.collect()

    # Short 라벨
    data_short = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=mark,
        atr=atr,
        funding_rate=funding_rate,
        next_funding_ts=next_funding_ts,
        spec=spec,
        direction=-1,
    )

    rows_short = [
        (r.symbol, r.ts, r.spec_hash, int(r.y), float(r.ret_net), float(r.mae), float(r.mfe), int(r.time_to_event_min))
        for r in data_short.itertuples(index=False)
    ]
    short_count = len(rows_short)
    persist_label_rows("labels_short_1m", rows_short)

    # 메모리 정리
    del candles, data_short, rows_short
    gc.collect()

    return long_count + short_count


def main() -> None:
    settings = get_settings()
    symbols = settings.universe_list()

    spec = LabelSpec(
        k_tp=1.5,
        k_sl=1.0,
        h_bars=360,
        risk_mae_atr=3.0,
        fee_rate=settings.taker_fee_rate,
        slippage_k=settings.slippage_k,
    )

    print("=== 라벨 생성 (메모리 효율) ===")
    print(f"종목: {len(symbols)}개")
    print(f"스펙 해시: {spec.hash()}")

    total = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)
        count = label_single_symbol(symbol, spec)
        total += count
        print(f"{count:,}개")

    print(f"\n=== 완료: 총 {total:,}개 라벨 ===")


if __name__ == "__main__":
    main()
