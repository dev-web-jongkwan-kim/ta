#!/usr/bin/env python3
"""
15분봉 전용 모델 학습 파이프라인

전체 10개 심볼에 대해 15분봉 기준으로 학습
- 1분봉 → 15분봉 리샘플링
- 15분봉 피처 계산
- 15분봉 레이블 생성
- 15분봉 모델 학습

Usage:
    python scripts/train_15m_model.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd

from packages.common.db import bulk_upsert, fetch_all, get_conn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_all_symbols() -> List[str]:
    """DB에서 전체 심볼 목록 조회"""
    rows = fetch_all("SELECT symbol FROM instruments ORDER BY symbol")
    return [r[0] for r in rows]


def step1_resample_candles(symbols: List[str]) -> None:
    """Step 1: 1분 캔들을 15분으로 리샘플링"""
    from services.features.resample import resample_all_symbols

    print("\n" + "=" * 60)
    print("Step 1: 1분봉 → 15분봉 리샘플링")
    print("=" * 60)

    count = resample_all_symbols(symbols, "15m")
    print(f"  → 총 {count:,}개 15분봉 캔들 생성")


def step2_compute_features(symbols: List[str]) -> None:
    """Step 2: 15분봉 피처 계산"""
    import json
    import math
    from services.features.compute import compute_features_for_symbol

    print("\n" + "=" * 60)
    print("Step 2: 15분봉 피처 계산")
    print("=" * 60)

    def clean_value(v):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def clean_features(features_dict):
        if not isinstance(features_dict, dict):
            return features_dict
        return {k: clean_value(v) for k, v in features_dict.items()}

    # BTC 캔들 로드 (레짐 피처용)
    btc_candles = _load_candles("BTCUSDT", "15m")

    total = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol} 처리 중...")

        candles = _load_candles(symbol, "15m")
        premium = _load_premium(symbol)
        oi = _load_open_interest(symbol)
        ls_ratio = _load_long_short_ratio(symbol)

        if candles.empty:
            print(f"    → 캔들 없음, 스킵")
            continue

        features_df = compute_features_for_symbol(
            symbol=symbol,
            candles=candles,
            premium=premium,
            btc_candles=btc_candles,
            open_interest=oi,
            long_short_ratio=ls_ratio,
            timeframe="15m",
        )

        if features_df.empty:
            print(f"    → 피처 계산 실패")
            continue

        # DB 저장
        rows = []
        for _, row in features_df.iterrows():
            cleaned_features = clean_features(row["features"])
            features_json = json.dumps(cleaned_features)
            rows.append((
                row["symbol"],
                row["ts"],
                int(row["schema_version"]),
                features_json,
                float(row["atr"]) if pd.notna(row["atr"]) else None,
                float(row["funding_z"]) if pd.notna(row["funding_z"]) else None,
                int(row["btc_regime"]) if pd.notna(row.get("btc_regime")) else None,
            ))

        bulk_upsert(
            "features_15m",
            ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
            rows,
            conflict_cols=["symbol", "ts"],
            update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
        )

        print(f"    → {len(rows):,}개 피처 저장")
        total += len(rows)

    print(f"\n  총 {total:,}개 15분봉 피처 저장 완료")


def step3_labeling(symbols: List[str]) -> None:
    """Step 3: 15분봉 레이블 생성"""
    from services.labeling.pipeline import LabelingConfig, run_labeling

    print("\n" + "=" * 60)
    print("Step 3: 15분봉 레이블 생성")
    print("=" * 60)

    config = LabelingConfig(timeframe="15m")
    stats = run_labeling(config=config, symbols=symbols, force_full=True, timeframe="15m")
    print(f"  → {stats.total_new_labels:,}개 레이블 생성")


def step4_train_model(symbols: List[str]) -> dict:
    """Step 4: 15분봉 모델 학습"""
    from services.labeling.pipeline import LabelingConfig
    from services.training.train import TrainConfig, run_training_job

    print("\n" + "=" * 60)
    print("Step 4: 15분봉 모델 학습")
    print("=" * 60)

    # 데이터 범위 확인
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(ts), MAX(ts) FROM candles_15m WHERE symbol = 'BTCUSDT'")
            row = cur.fetchone()
            min_ts, max_ts = row[0], row[1]

    if not min_ts or not max_ts:
        print("ERROR: 15분봉 데이터가 없습니다")
        return {"status": "error", "message": "No data"}

    # Train: 처음 ~ 마지막-30일, Val: 마지막 30일
    val_end = max_ts
    val_start = val_end - timedelta(days=30)
    train_end = val_start - timedelta(days=1)
    train_start = min_ts

    print(f"  Train: {train_start.date()} ~ {train_end.date()}")
    print(f"  Val: {val_start.date()} ~ {val_end.date()}")

    # Label spec
    label_config = LabelingConfig(timeframe="15m")
    spec = label_config.spec()
    spec_hash = spec.hash()

    print(f"  Label spec hash: {spec_hash}")
    print(f"  Symbols: {', '.join(symbols)}")

    # 15분봉 전용 학습 (멀티 타임프레임 비활성화)
    train_cfg = TrainConfig(
        label_spec_hash=spec_hash,
        feature_schema_version=4,
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        val_start=val_start.isoformat(),
        val_end=val_end.isoformat(),
        targets=("er_long", "q05_long", "e_mae_long", "e_hold_long"),
        use_multi_tf=False,  # 15분봉 단일 타임프레임
        timeframe="15m",     # 15분봉 사용
    )

    result = run_training_job(train_cfg, symbols=symbols)
    return result


def _load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    """캔들 데이터 로드"""
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


def _load_premium(symbol: str) -> pd.DataFrame:
    """프리미엄 인덱스 로드"""
    rows = fetch_all(
        "SELECT ts, mark_price, index_price, last_price, last_funding_rate, next_funding_time FROM premium_index WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "mark_price", "index_price", "last_price", "last_funding_rate", "next_funding_time"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _load_open_interest(symbol: str) -> pd.DataFrame:
    """OI 데이터 로드"""
    rows = fetch_all(
        "SELECT ts, open_interest, open_interest_value FROM open_interest WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open_interest", "open_interest_value"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _load_long_short_ratio(symbol: str) -> pd.DataFrame:
    """롱숏 비율 데이터 로드"""
    rows = fetch_all(
        """SELECT ts, long_short_ratio, long_account, short_account,
                  top_long_short_ratio, top_long_account, top_short_account,
                  taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
           FROM long_short_ratio WHERE symbol=%s ORDER BY ts""",
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "ts", "long_short_ratio", "long_account", "short_account",
        "top_long_short_ratio", "top_long_account", "top_short_account",
        "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"
    ])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def print_results(result: dict) -> None:
    """학습 결과 출력"""
    if result.get("status") == "error":
        print(f"\n❌ 학습 실패: {result.get('message')}")
        return

    metrics = result.get("metrics", {})
    trade = metrics.get("trade", {})

    print("\n" + "=" * 60)
    print("📊 15분봉 모델 학습 결과")
    print("=" * 60)
    print(f"  Model ID: {result.get('model_id', 'N/A')}")
    print(f"  승률: {trade.get('win_rate', 0):.1f}%")
    print(f"  기대값: {trade.get('expectancy', 0)*100:.3f}%")
    print(f"  Profit Factor: {trade.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: ${trade.get('max_drawdown', 0):.2f}")

    # 필터링된 결과
    filtered = metrics.get("filtered_er0_long", {})
    if filtered:
        print(f"\n  [er>0 필터 적용 시]")
        print(f"    승률: {filtered.get('win_rate', 0):.1f}%")
        print(f"    기대값: {filtered.get('expectancy', 0)*100:.3f}%")
        print(f"    Profit Factor: {filtered.get('profit_factor', 0):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="15분봉 모델 학습")
    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="리샘플링 단계 스킵",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="피처 계산 단계 스킵",
    )
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="라벨링 단계 스킵",
    )
    args = parser.parse_args()

    # 전체 심볼 목록 조회
    symbols = get_all_symbols()

    print("=" * 60)
    print("🚀 15분봉 전용 모델 학습 파이프라인")
    print("=" * 60)
    print(f"대상 심볼 ({len(symbols)}개): {', '.join(symbols)}")
    print(f"타임프레임: 15m")

    # Step 1: 리샘플링
    if not args.skip_resample:
        step1_resample_candles(symbols)
    else:
        print("\n[스킵] Step 1: 리샘플링")

    # Step 2: 피처 계산
    if not args.skip_features:
        step2_compute_features(symbols)
    else:
        print("\n[스킵] Step 2: 피처 계산")

    # Step 3: 라벨링
    if not args.skip_labeling:
        step3_labeling(symbols)
    else:
        print("\n[스킵] Step 3: 라벨링")

    # Step 4: 학습
    result = step4_train_model(symbols)

    # 결과 출력
    print_results(result)

    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
