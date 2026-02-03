"""
히스토리 캔들에 대한 피처 생성 스크립트
"""
from __future__ import annotations

import json
from typing import List

import pandas as pd

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, fetch_all
from services.features.compute import compute_features_for_symbol


def load_candles(symbol: str) -> pd.DataFrame:
    """캔들 데이터 로드"""
    rows = fetch_all(
        """
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles_1m
        WHERE symbol = %s
        ORDER BY ts
        """,
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows,
        columns=["symbol", "ts", "open", "high", "low", "close", "volume"],
    )


def load_premium(symbol: str) -> pd.DataFrame:
    """프리미엄 인덱스 로드 (펀딩 등)"""
    rows = fetch_all(
        """
        SELECT symbol, ts, mark_price, index_price, last_funding_rate, next_funding_time
        FROM premium_index
        WHERE symbol = %s
        ORDER BY ts
        """,
        (symbol,),
    )
    if not rows:
        # 펀딩 레이트 테이블에서 대체 로드
        funding_rows = fetch_all(
            """
            SELECT symbol, ts, funding_rate as last_funding_rate
            FROM funding_rates
            WHERE symbol = %s
            ORDER BY ts
            """,
            (symbol,),
        )
        if funding_rows:
            df = pd.DataFrame(funding_rows, columns=["symbol", "ts", "last_funding_rate"])
            df["mark_price"] = None
            df["index_price"] = None
            df["next_funding_time"] = None
            return df
        return pd.DataFrame()
    return pd.DataFrame(
        rows,
        columns=["symbol", "ts", "mark_price", "index_price", "last_funding_rate", "next_funding_time"],
    )


def main() -> None:
    settings = get_settings()
    symbols = settings.universe_list()

    print("=== 히스토리 피처 생성 시작 ===")
    print(f"종목: {symbols}")

    # BTC 캔들 로드 (btc_regime 계산용)
    btc_candles = load_candles("BTCUSDT")
    print(f"BTC 캔들: {len(btc_candles)}개")

    total_features = 0

    for symbol in symbols:
        print(f"\n[{symbol}] 처리 중...")

        candles = load_candles(symbol)
        if candles.empty:
            print(f"  ✗ 캔들 없음")
            continue

        premium = load_premium(symbol)
        print(f"  캔들: {len(candles)}개, 프리미엄: {len(premium)}개")

        # 피처 계산
        features_df = compute_features_for_symbol(
            symbol=symbol,
            candles=candles,
            premium=premium,
            btc_candles=btc_candles if symbol != "BTCUSDT" else candles,
        )

        # NaN 제거 (초기 윈도우로 인한 결측)
        features_df = features_df.dropna(subset=["atr"])

        if features_df.empty:
            print(f"  ✗ 피처 생성 실패 (데이터 부족)")
            continue

        # DB 저장 (NaN, Infinity를 None으로 변환)
        def clean_features(feat_dict):
            """NaN, Infinity 값을 None으로 변환"""
            import math
            def clean_value(v):
                if v is None:
                    return None
                if isinstance(v, float):
                    if pd.isna(v) or math.isinf(v):
                        return None
                return v
            return {k: clean_value(v) for k, v in feat_dict.items()}

        rows = []
        for _, row in features_df.iterrows():
            cleaned_features = clean_features(row["features"])
            rows.append(
                (
                    row["symbol"],
                    row["ts"],
                    row["schema_version"],
                    json.dumps(cleaned_features),
                    float(row["atr"]) if pd.notna(row["atr"]) else None,
                    float(row["funding_z"]) if pd.notna(row["funding_z"]) else None,
                    int(row["btc_regime"]) if pd.notna(row["btc_regime"]) else None,
                )
            )

        bulk_upsert(
            "features_1m",
            ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
            rows,
            conflict_cols=["symbol", "ts"],
            update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
        )

        total_features += len(rows)
        print(f"  → 피처 {len(rows)}개 저장 완료")

    print(f"\n=== 완료 ===")
    print(f"총 피처: {total_features:,}개")

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
