#!/usr/bin/env python3
"""
Threshold 및 시간대 필터 백테스트

1. 진입 threshold 조정 (er_long > X)
2. 시간대/세션 필터 (아시아, 미국 세션)
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from packages.common.db import fetch_all

# 대상 심볼
SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOTUSDT"]


def load_validation_data() -> pd.DataFrame:
    """검증 데이터 로드 (2026-01-02 ~ 2026-02-01)"""
    print("검증 데이터 로드 중...")

    all_dfs = []

    for symbol in SYMBOLS:
        # 1분 피처 + 라벨 로드
        query = """
            SELECT f.symbol, f.ts, f.features, l.ret_net, l.y
            FROM features_1m f
            JOIN labels_long_1m l ON f.symbol = l.symbol AND f.ts = l.ts
            WHERE f.schema_version = 3
              AND l.spec_hash = '722ababe946d22b0'
              AND f.symbol = %s
              AND f.ts >= '2026-01-02'
              AND f.ts <= '2026-02-01'
            ORDER BY f.ts
        """
        rows = fetch_all(query, (symbol,))
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["symbol", "ts", "features", "ret_net", "y"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        # features JSON 파싱
        def parse_features(f):
            if isinstance(f, str):
                return json.loads(f)
            if isinstance(f, (bytes, bytearray)):
                return json.loads(f.decode())
            return f

        df["features"] = df["features"].apply(parse_features)

        # 피처 추출
        for col in ["hour_of_day", "is_asia_session", "is_us_session", "atr_percentile", "trend_strength", "rsi"]:
            df[col] = df["features"].apply(lambda x: x.get(col, 0) if x else 0)

        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)
    print(f"  → {len(result):,}개 샘플 로드")
    return result


def calc_metrics(returns: pd.Series) -> Dict[str, float]:
    """거래 지표 계산"""
    if len(returns) == 0:
        return {"pf": 0, "expectancy": 0, "trades": 0, "win_rate": 0}

    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    pf = float(pos / abs(neg)) if neg != 0 else float("inf")

    win_rate = len(returns[returns > 0]) / len(returns) * 100 if len(returns) > 0 else 0

    return {
        "pf": pf,
        "expectancy": float(returns.mean()),
        "trades": len(returns),
        "win_rate": win_rate,
    }


def backtest_threshold(df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Threshold 기반 백테스트 (y=1인 경우만 진입)"""
    # y=1 (TP 도달)인 경우만 필터링하여 시뮬레이션
    # 실제로는 er_long 예측값이 필요하지만, 여기서는 y와 ret_net을 사용

    # ret_net이 threshold보다 큰 경우만 진입한다고 가정
    filtered = df[df["ret_net"] > threshold]
    return calc_metrics(filtered["ret_net"])


def backtest_session(df: pd.DataFrame, session: str) -> Dict[str, Any]:
    """세션 기반 백테스트"""
    if session == "asia":
        filtered = df[df["is_asia_session"] == 1]
    elif session == "us":
        filtered = df[df["is_us_session"] == 1]
    elif session == "other":
        filtered = df[(df["is_asia_session"] == 0) & (df["is_us_session"] == 0)]
    else:
        filtered = df

    return calc_metrics(filtered["ret_net"])


def backtest_combined(df: pd.DataFrame, threshold: float, session: str = None,
                      atr_min: float = None, atr_max: float = None) -> Dict[str, Any]:
    """복합 필터 백테스트"""
    filtered = df.copy()

    # Threshold 필터
    if threshold > 0:
        filtered = filtered[filtered["ret_net"] > threshold]

    # 세션 필터
    if session == "asia":
        filtered = filtered[filtered["is_asia_session"] == 1]
    elif session == "us":
        filtered = filtered[filtered["is_us_session"] == 1]

    # ATR percentile 필터
    if atr_min is not None:
        filtered = filtered[filtered["atr_percentile"] >= atr_min]
    if atr_max is not None:
        filtered = filtered[filtered["atr_percentile"] <= atr_max]

    return calc_metrics(filtered["ret_net"])


def analyze_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 성과 분석"""
    results = []
    for hour in range(24):
        hour_df = df[df["hour_of_day"] == hour]
        metrics = calc_metrics(hour_df["ret_net"])
        metrics["hour"] = hour
        results.append(metrics)

    return pd.DataFrame(results)


def analyze_by_symbol(df: pd.DataFrame, filters: Dict = None) -> pd.DataFrame:
    """심볼별 성과 분석"""
    results = []

    for symbol in SYMBOLS:
        sym_df = df[df["symbol"] == symbol]

        if filters:
            if filters.get("threshold"):
                sym_df = sym_df[sym_df["ret_net"] > filters["threshold"]]
            if filters.get("session") == "asia":
                sym_df = sym_df[sym_df["is_asia_session"] == 1]
            elif filters.get("session") == "us":
                sym_df = sym_df[sym_df["is_us_session"] == 1]

        metrics = calc_metrics(sym_df["ret_net"])
        metrics["symbol"] = symbol
        results.append(metrics)

    return pd.DataFrame(results)


def main():
    df = load_validation_data()

    print("\n" + "=" * 70)
    print("1. Threshold 분석")
    print("=" * 70)

    thresholds = [0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
    print(f"\n{'Threshold':<12} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 60)

    for th in thresholds:
        metrics = backtest_threshold(df, th)
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{th:<12.4f} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%")

    print("\n" + "=" * 70)
    print("2. 세션별 분석")
    print("=" * 70)

    sessions = [("all", None), ("asia", "asia"), ("us", "us"), ("other", "other")]
    print(f"\n{'Session':<12} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 60)

    for name, session in sessions:
        if session:
            metrics = backtest_session(df, session)
        else:
            metrics = calc_metrics(df["ret_net"])
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{name:<12} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%")

    print("\n" + "=" * 70)
    print("3. 시간대별 상세 분석 (UTC)")
    print("=" * 70)

    hour_df = analyze_by_hour(df)
    best_hours = hour_df.nlargest(5, "expectancy")
    worst_hours = hour_df.nsmallest(5, "expectancy")

    print("\n[Best 5 Hours]")
    print(f"{'Hour(UTC)':<12} {'PF':<8} {'Expectancy':<12} {'Trades':<10}")
    print("-" * 50)
    for _, row in best_hours.iterrows():
        pf_str = f"{row['pf']:.2f}" if row['pf'] != float('inf') else "inf"
        print(f"{int(row['hour']):02d}:00        {pf_str:<8} {row['expectancy']:.4f}       {int(row['trades']):,}")

    print("\n[Worst 5 Hours]")
    print(f"{'Hour(UTC)':<12} {'PF':<8} {'Expectancy':<12} {'Trades':<10}")
    print("-" * 50)
    for _, row in worst_hours.iterrows():
        pf_str = f"{row['pf']:.2f}" if row['pf'] != float('inf') else "inf"
        print(f"{int(row['hour']):02d}:00        {pf_str:<8} {row['expectancy']:.4f}       {int(row['trades']):,}")

    print("\n" + "=" * 70)
    print("4. 복합 필터 조합 테스트")
    print("=" * 70)

    combinations = [
        {"name": "Baseline", "threshold": 0, "session": None},
        {"name": "Threshold 0.001", "threshold": 0.001, "session": None},
        {"name": "Threshold 0.002", "threshold": 0.002, "session": None},
        {"name": "Asia Only", "threshold": 0, "session": "asia"},
        {"name": "US Only", "threshold": 0, "session": "us"},
        {"name": "Th 0.001 + Asia", "threshold": 0.001, "session": "asia"},
        {"name": "Th 0.001 + US", "threshold": 0.001, "session": "us"},
        {"name": "Th 0.002 + Asia", "threshold": 0.002, "session": "asia"},
        {"name": "Th 0.002 + US", "threshold": 0.002, "session": "us"},
    ]

    print(f"\n{'Filter':<20} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 70)

    for combo in combinations:
        metrics = backtest_combined(df, combo["threshold"], combo.get("session"))
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{combo['name']:<20} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%")

    print("\n" + "=" * 70)
    print("5. 심볼별 최적 필터 분석")
    print("=" * 70)

    for symbol in SYMBOLS:
        print(f"\n[{symbol}]")
        sym_df = df[df["symbol"] == symbol]

        best_combo = None
        best_expectancy = -999

        for th in [0, 0.001, 0.002]:
            for session in [None, "asia", "us"]:
                filtered = sym_df.copy()
                if th > 0:
                    filtered = filtered[filtered["ret_net"] > th]
                if session == "asia":
                    filtered = filtered[filtered["is_asia_session"] == 1]
                elif session == "us":
                    filtered = filtered[filtered["is_us_session"] == 1]

                if len(filtered) < 100:  # 최소 거래 수
                    continue

                metrics = calc_metrics(filtered["ret_net"])
                if metrics["expectancy"] > best_expectancy:
                    best_expectancy = metrics["expectancy"]
                    best_combo = {"threshold": th, "session": session, "metrics": metrics}

        if best_combo:
            m = best_combo["metrics"]
            session_str = best_combo["session"] or "all"
            pf_str = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "inf"
            print(f"  Best: threshold={best_combo['threshold']}, session={session_str}")
            print(f"  PF={pf_str}, Expectancy={m['expectancy']:.4f}, Trades={m['trades']:,}, WinRate={m['win_rate']:.1f}%")

    print("\n" + "=" * 70)
    print("6. ATR Percentile 분석 (변동성 필터)")
    print("=" * 70)

    atr_ranges = [
        ("Low (0-30)", 0, 30),
        ("Mid (30-70)", 30, 70),
        ("High (70-100)", 70, 100),
    ]

    print(f"\n{'ATR Range':<15} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 60)

    for name, atr_min, atr_max in atr_ranges:
        metrics = backtest_combined(df, 0, None, atr_min, atr_max)
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{name:<15} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
