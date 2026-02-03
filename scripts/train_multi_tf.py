#!/usr/bin/env python3
"""
멀티 타임프레임 학습 파이프라인

3개 심볼 (BTCUSDT, ETHUSDT, DOTUSDT) + 3개 타임프레임 (1m, 15m, 1h)
새로운 피처: 심볼 상관관계, 변동성 레짐, 시간대, 모멘텀 강도

Usage:
    python scripts/train_multi_tf.py
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

# 대상 심볼
TARGET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOTUSDT"]

# 타임프레임
TIMEFRAMES = ["1m", "15m", "1h"]


def step1_resample_candles(symbols: List[str]) -> None:
    """Step 1: 1분 캔들을 15분/1시간으로 리샘플링"""
    from services.features.resample import resample_all_symbols

    print("\n" + "=" * 50)
    print("Step 1: 캔들 리샘플링")
    print("=" * 50)

    for tf in ["15m", "1h"]:
        print(f"  {tf} 리샘플링 중...")
        count = resample_all_symbols(symbols, tf)  # type: ignore
        print(f"  → {tf}: {count:,}개 캔들 저장")


def step2_compute_features(symbols: List[str]) -> None:
    """Step 2: 각 타임프레임별 피처 계산"""
    from services.features.compute import compute_features_for_symbol

    print("\n" + "=" * 50)
    print("Step 2: 피처 계산")
    print("=" * 50)

    for tf in TIMEFRAMES:
        print(f"\n  === {tf} 피처 계산 ===")
        total = 0

        # BTC, ETH 캔들 로드 (상관관계 피처용)
        btc_candles = _load_candles("BTCUSDT", tf)
        eth_candles = _load_candles("ETHUSDT", tf)

        for symbol in symbols:
            candles = _load_candles(symbol, tf)
            premium = _load_premium(symbol)

            if candles.empty:
                print(f"    {symbol}: 캔들 없음")
                continue

            features_df = compute_features_for_symbol(
                symbol=symbol,
                candles=candles,
                premium=premium,
                btc_candles=btc_candles,
                eth_candles=eth_candles,
                timeframe=tf,  # type: ignore
            )

            if not features_df.empty:
                _save_features(features_df, tf)
                total += len(features_df)
                print(f"    {symbol}: {len(features_df):,}개 피처 저장")

        print(f"  → {tf} 총 {total:,}개 피처")


def step3_labeling(symbols: List[str]) -> None:
    """Step 3: 각 타임프레임별 라벨링"""
    from services.labeling.pipeline import LabelingConfig, run_labeling

    print("\n" + "=" * 50)
    print("Step 3: 라벨링")
    print("=" * 50)

    for tf in TIMEFRAMES:
        print(f"\n  === {tf} 라벨링 ===")
        config = LabelingConfig(timeframe=tf)  # type: ignore
        stats = run_labeling(config=config, symbols=symbols, force_full=True, timeframe=tf)  # type: ignore
        print(f"  → {tf}: {stats.total_new_labels:,}개 라벨 생성")


def step4_train_models(symbols: List[str]) -> dict:
    """Step 4: 모델 학습 (멀티 타임프레임)"""
    from services.labeling.pipeline import LabelingConfig
    from services.training.train import TrainConfig, run_training_job

    print("\n" + "=" * 50)
    print("Step 4: 모델 학습")
    print("=" * 50)

    # 데이터 범위 확인
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(ts), MAX(ts) FROM candles_1m WHERE symbol = 'BTCUSDT'")
            row = cur.fetchone()
            min_ts, max_ts = row[0], row[1]

    if not min_ts or not max_ts:
        print("ERROR: 데이터가 없습니다")
        return {"status": "error", "message": "No data"}

    # Train: 처음 ~ 마지막-1달, Val: 마지막 1달
    val_end = max_ts
    val_start = val_end - timedelta(days=30)
    train_end = val_start - timedelta(days=1)
    train_start = min_ts

    print(f"\n  Train: {train_start.date()} ~ {train_end.date()}")
    print(f"  Val: {val_start.date()} ~ {val_end.date()}")

    # Label spec hash 계산
    label_config = LabelingConfig()
    spec = label_config.spec()
    spec_hash = spec.hash()

    print(f"  Label spec hash: {spec_hash}")

    # 멀티 타임프레임 학습
    print("\n  멀티 타임프레임 학습 중...")

    train_cfg = TrainConfig(
        label_spec_hash=spec_hash,
        feature_schema_version=3,  # 새 피처 스키마
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        val_start=val_start.isoformat(),
        val_end=val_end.isoformat(),
        targets=("er_long", "q05_long", "e_mae_long", "e_hold_long"),
        use_multi_tf=True,
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
        "SELECT ts, mark_price, last_funding_rate FROM premium_index WHERE symbol=%s ORDER BY ts",
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "mark_price", "last_funding_rate"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _save_features(df: pd.DataFrame, timeframe: str) -> None:
    """피처를 DB에 저장"""
    import json
    import math

    def clean_value(v):
        """NaN, Inf 값을 None으로 변환"""
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def clean_features(features_dict):
        """피처 딕셔너리의 NaN/Inf 값을 None으로 변환"""
        if not isinstance(features_dict, dict):
            return features_dict
        return {k: clean_value(v) for k, v in features_dict.items()}

    table = f"features_{timeframe}"
    rows = []
    for _, row in df.iterrows():
        cleaned_features = clean_features(row["features"])
        features_json = json.dumps(cleaned_features) if isinstance(cleaned_features, dict) else cleaned_features
        rows.append((
            row["symbol"],
            row["ts"],
            int(row["schema_version"]),
            features_json,
            float(row["atr"]) if pd.notna(row["atr"]) else None,
            float(row["funding_z"]) if pd.notna(row["funding_z"]) else None,
            int(row["btc_regime"]) if pd.notna(row["btc_regime"]) else None,
        ))

    bulk_upsert(
        table,
        ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
        rows,
        conflict_cols=["symbol", "ts"],
        update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
    )


def print_results(result: dict) -> None:
    """결과 출력"""
    print("\n" + "=" * 60)
    print("학습 완료")
    print("=" * 60)

    if result.get("status") != "ok":
        print(f"Error: {result.get('message', 'Unknown error')}")
        return

    metrics = result.get("metrics", {})
    trade = metrics.get("trade", {})

    # 모델 성능
    print("\n모델 성능")
    print("-" * 40)
    print(f"{'Target':<15} {'RMSE':<10} {'Best Iter':<10}")
    print("-" * 40)

    for target in ["er_long", "q05_long", "e_mae_long", "e_hold_long"]:
        if target in metrics:
            m = metrics[target]
            print(f"{target:<15} {m.get('rmse', 0):.4f}     {m.get('best_iteration', '-')}")

    # 거래 성과
    print("\n검증 결과")
    print("-" * 40)
    print(f"Profit Factor: {trade.get('profit_factor', 0):.2f}")
    print(f"Max Drawdown: {trade.get('max_drawdown', 0):.2f}")
    print(f"Expectancy: {trade.get('expectancy', 0):.4f}")
    print(f"Turnover: {trade.get('turnover', 0):,.0f}")

    # 심볼별 성과
    by_symbol = metrics.get("by_symbol", {})
    if by_symbol:
        print("\n심볼별 성능")
        print("-" * 40)
        print(f"{'Symbol':<12} {'PF':<10} {'Expectancy':<12}")
        print("-" * 40)

        for symbol, sm in sorted(by_symbol.items(), key=lambda x: x[1].get("profit_factor", 0), reverse=True):
            pf = sm.get("profit_factor", 0)
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"{symbol:<12} {pf_str:<10} {sm.get('expectancy', 0):.4f}")

    # 결론
    print("\n" + "=" * 60)
    print("결론")
    print("=" * 60)
    print(f"  - Model ID: {result.get('model_id', 'N/A')}")
    print(f"  - Feature Count: {result.get('report', {}).get('feature_count', 'N/A')}")

    expectancy = trade.get("expectancy", 0)
    if expectancy > 0:
        print(f"  - 양수 기대값({expectancy*100:.2f}%)으로 수익 가능성 있음")
    else:
        print(f"  - 음수 기대값({expectancy*100:.2f}%)으로 추가 개선 필요")


def main() -> None:
    parser = argparse.ArgumentParser(description="멀티 타임프레임 학습")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=TARGET_SYMBOLS,
        help=f"학습할 심볼 (기본값: {TARGET_SYMBOLS})",
    )
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

    symbols = args.symbols

    print("=" * 60)
    print("멀티 타임프레임 학습 파이프라인")
    print("=" * 60)
    print(f"심볼: {', '.join(symbols)}")
    print(f"타임프레임: {', '.join(TIMEFRAMES)}")

    # Step 1: 리샘플링
    if not args.skip_resample:
        step1_resample_candles(symbols)

    # Step 2: 피처 계산
    if not args.skip_features:
        step2_compute_features(symbols)

    # Step 3: 라벨링
    if not args.skip_labeling:
        step3_labeling(symbols)

    # Step 4: 학습
    result = step4_train_models(symbols)

    # 결과 출력
    print_results(result)


if __name__ == "__main__":
    main()
