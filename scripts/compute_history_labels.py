"""
히스토리 캔들에 대한 라벨 생성 스크립트
premium_index가 없는 경우 close price를 mark price로 사용
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, fetch_all
from services.labeling.triple_barrier import LabelSpec, label_direction_vectorized


def load_candles(symbol: str) -> pd.DataFrame:
    """캔들 데이터 로드"""
    rows = fetch_all(
        "SELECT ts, high, low, close FROM candles_1m WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    return pd.DataFrame(rows, columns=["ts", "high", "low", "close"])


def load_funding_rates(symbol: str) -> pd.DataFrame:
    """펀딩레이트 데이터 로드"""
    rows = fetch_all(
        "SELECT ts, funding_rate FROM funding_rates WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    if not rows:
        return pd.DataFrame(columns=["ts", "funding_rate"])
    return pd.DataFrame(rows, columns=["ts", "funding_rate"])


def persist_label_rows(table: str, rows: List[Tuple]) -> None:
    bulk_upsert(
        table,
        ["symbol", "ts", "spec_hash", "y", "ret_net", "mae", "mfe", "time_to_event_min"],
        rows,
        conflict_cols=["symbol", "ts", "spec_hash"],
        update_cols=["y", "ret_net", "mae", "mfe", "time_to_event_min"],
    )


def label_symbol_history(symbol: str, spec: LabelSpec, candles: pd.DataFrame) -> int:
    """히스토리 캔들에 대해 라벨 생성"""
    if candles.empty:
        return 0

    # funding rate 로드 및 캔들에 매핑
    funding_df = load_funding_rates(symbol)
    if funding_df.empty:
        # 펀딩레이트 없으면 0으로 채움
        funding_rate = pd.Series([0.0] * len(candles), index=candles.index)
        next_funding_ts = pd.Series([pd.NaT] * len(candles), index=candles.index)
    else:
        funding_df = funding_df.set_index("ts")
        funding_rate = funding_df["funding_rate"].reindex(candles["ts"], method="ffill").fillna(0.0).values
        # 다음 펀딩 시간 계산 (8시간마다)
        next_funding_ts = candles["ts"].values  # 단순화

    # ATR 계산
    atr = candles["high"].rolling(window=14).max() - candles["low"].rolling(window=14).min()

    # close를 mark price로 사용
    mark = candles["close"].values

    # Long 라벨 생성
    data_long = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=mark,
        atr=atr.bfill().values,
        funding_rate=funding_rate if isinstance(funding_rate, pd.Series) else pd.Series(funding_rate).values,
        next_funding_ts=next_funding_ts if isinstance(next_funding_ts, pd.Series) else pd.Series(next_funding_ts).values,
        spec=spec,
        direction=1,
    )

    rows_long = [
        (
            r.symbol,
            r.ts,
            r.spec_hash,
            int(r.y),
            float(r.ret_net),
            float(r.mae),
            float(r.mfe),
            int(r.time_to_event_min),
        )
        for r in data_long.itertuples(index=False)
    ]
    persist_label_rows("labels_long_1m", rows_long)

    # Short 라벨 생성
    data_short = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=mark,
        atr=atr.bfill().values,
        funding_rate=funding_rate if isinstance(funding_rate, pd.Series) else pd.Series(funding_rate).values,
        next_funding_ts=next_funding_ts if isinstance(next_funding_ts, pd.Series) else pd.Series(next_funding_ts).values,
        spec=spec,
        direction=-1,
    )

    rows_short = [
        (
            r.symbol,
            r.ts,
            r.spec_hash,
            int(r.y),
            float(r.ret_net),
            float(r.mae),
            float(r.mfe),
            int(r.time_to_event_min),
        )
        for r in data_short.itertuples(index=False)
    ]
    persist_label_rows("labels_short_1m", rows_short)

    return len(rows_long) + len(rows_short)


def main() -> None:
    settings = get_settings()
    symbols = settings.universe_list()

    # 라벨링 스펙
    spec = LabelSpec(
        k_tp=1.5,
        k_sl=1.0,
        h_bars=360,
        risk_mae_atr=3.0,
        fee_rate=settings.taker_fee_rate,
        slippage_k=settings.slippage_k,
    )

    print("=== 히스토리 라벨 생성 시작 ===")
    print(f"종목: {symbols}")
    print(f"스펙: k_tp={spec.k_tp}, k_sl={spec.k_sl}, h_bars={spec.h_bars}")
    print(f"스펙 해시: {spec.hash()}")

    total_labels = 0

    for symbol in symbols:
        print(f"\n[{symbol}] 처리 중...")

        candles = load_candles(symbol)
        if candles.empty:
            print(f"  ✗ 캔들 없음")
            continue

        print(f"  캔들: {len(candles)}개")

        # 라벨 생성
        count = label_symbol_history(symbol, spec, candles)
        total_labels += count
        print(f"  → 라벨 {count:,}개 저장 완료")

    print(f"\n=== 완료 ===")
    print(f"총 라벨: {total_labels:,}개")

    # 조인 가능 데이터 확인
    result = fetch_all(
        """
        SELECT COUNT(*) as cnt
        FROM features_1m f
        JOIN labels_long_1m l ON f.symbol = l.symbol AND f.ts = l.ts
        WHERE f.schema_version = 1
        """
    )
    print(f"피처-라벨 조인 가능: {result[0][0]:,}개")


if __name__ == "__main__":
    main()
