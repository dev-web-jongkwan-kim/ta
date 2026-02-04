#!/usr/bin/env python3
"""
30종목 학습 파이프라인 (배치 학습)

최적화 설정:
- 배치 학습 (전체 데이터 한번에 학습) - 최고 성능
- Optuna 타겟별 개별 튜닝 (50 trials × 4 targets)
- 피처 선택 (누적 중요도 99.9%)
- Meta-labeling
- CatBoost 앙상블
- 멀티 타임프레임 (1m, 15m, 1h)

최고 결과: PF=2.64, Expectancy=+0.30% (er>0.001 필터)

Usage:
    python scripts/train_30_symbols.py
    python scripts/train_30_symbols.py --symbols BTCUSDT,ETHUSDT,DOTUSDT
    python scripts/train_30_symbols.py --optuna-trials 30
    python scripts/train_30_symbols.py --skip-optuna  # 기본 파라미터 사용
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import timedelta
from typing import Dict, List, Any

from packages.common.config import get_settings
from packages.common.db import get_conn
from services.labeling.pipeline import LabelingConfig
from services.training.train_improved import (
    ImprovedTrainConfig,
    run_improved_training,
)


def get_universe() -> List[str]:
    """활성 심볼 유니버스 조회"""
    settings = get_settings()
    return settings.universe_list()


def get_data_range() -> tuple:
    """데이터 범위 확인"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(ts), MAX(ts) FROM candles_1m WHERE symbol = 'BTCUSDT'")
            row = cur.fetchone()
            return row[0], row[1]


def print_results(result: Dict[str, Any]) -> None:
    """결과 출력"""
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    if result.get("status") != "ok":
        print(f"Error: {result.get('message', 'Unknown error')}")
        return

    metrics = result.get("metrics", {})

    # 거래 지표
    print("\n[거래 지표]")
    print("-" * 50)

    trade = metrics.get("trade", {})
    pf = trade.get("profit_factor", 0)
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  Baseline:   PF={pf_str}, Expectancy={trade.get('expectancy', 0)*100:.2f}%, "
          f"Trades={int(trade.get('turnover', 0)):,}")

    # 필터링된 지표 (롱)
    for name, key in [("Long er>0", "filtered_er0_long"), ("Long er>0.001", "filtered_er001_long"), ("Long+meta", "filtered_meta")]:
        m = metrics.get(key, {})
        if m:
            pf = m.get("profit_factor", 0)
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"  {name:<14}: PF={pf_str}, Expectancy={m.get('expectancy', 0)*100:.2f}%, "
                  f"Trades={int(m.get('turnover', 0)):,}")

    # 필터링된 지표 (숏)
    for name, key in [("Short er>0", "filtered_er0_short"), ("Short er>0.001", "filtered_er001_short")]:
        m = metrics.get(key, {})
        if m:
            pf = m.get("profit_factor", 0)
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"  {name:<14}: PF={pf_str}, Expectancy={m.get('expectancy', 0)*100:.2f}%, "
                  f"Trades={int(m.get('turnover', 0)):,}")

    # 결합 지표
    combined = metrics.get("filtered_combined", {})
    if combined:
        pf = combined.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  {'Combined':<14}: PF={pf_str}, Expectancy={combined.get('expectancy', 0)*100:.2f}%, "
              f"Trades={int(combined.get('turnover', 0)):,}")

    # 심볼별 지표
    by_symbol = metrics.get("by_symbol", {})
    if by_symbol:
        print("\n[심볼별 성과]")
        print("-" * 50)

        # 기대값 순 정렬
        sorted_symbols = sorted(by_symbol.items(), key=lambda x: x[1].get("expectancy", 0), reverse=True)

        positive_count = 0
        for symbol, sm in sorted_symbols[:10]:  # Top 10만 출력
            pf = sm.get("profit_factor", 0)
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            exp = sm.get("expectancy", 0) * 100
            exp_sign = "+" if exp > 0 else ""
            print(f"  {symbol:<12}: PF={pf_str:<6}, Expectancy={exp_sign}{exp:.2f}%")
            if exp > 0:
                positive_count += 1

        if len(sorted_symbols) > 10:
            print(f"  ... ({len(sorted_symbols) - 10}개 더)")

        print(f"\n  양수 기대값 심볼: {positive_count}/{len(by_symbol)}개")

    # Kelly 계산
    print("\n[포지션 사이징 (Half Kelly)]")
    print("-" * 50)

    if trade.get("win_rate") and trade.get("expectancy"):
        win_rate = trade.get("win_rate", 50) / 100
        avg_win = trade.get("expectancy", 0.001) * 2 if trade.get("expectancy", 0) > 0 else 0.001
        avg_loss = abs(trade.get("expectancy", 0.001))

        from packages.common.risk import calculate_kelly_fraction
        kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        print(f"  Full Kelly: {kelly.kelly_fraction*100:.1f}%")
        print(f"  Half Kelly: {kelly.half_kelly*100:.1f}%")
        print(f"  권장 포지션: {kelly.position_pct*100:.1f}%")

    # 모델 정보
    print("\n[모델 정보]")
    print("-" * 50)
    print(f"  Model ID: {result.get('model_id', 'N/A')}")
    print(f"  Feature Count: {result.get('feature_count', 'N/A')}")
    print(f"  Symbols: {result.get('symbols_count', 'N/A')}개")

    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    best_exp = 0
    best_filter = "baseline"
    for name, key in [("baseline", "trade"), ("er>0", "filtered_er0"),
                      ("er>0.001", "filtered_er001"), ("meta", "filtered_meta")]:
        m = metrics.get(key, {})
        if m and m.get("expectancy", 0) > best_exp:
            best_exp = m.get("expectancy", 0)
            best_filter = name

    if best_exp > 0:
        print(f"  ✓ 양수 기대값 달성! ({best_filter}: +{best_exp*100:.2f}%)")
        print(f"  ✓ 실거래 테스트 권장")
    else:
        print(f"  △ 최고 기대값: {best_exp*100:.2f}% ({best_filter})")
        print(f"  → 추가 개선 필요")


def main():
    parser = argparse.ArgumentParser(description="30종목 배치 학습")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="학습할 심볼 (쉼표 구분, 기본값: 전체 유니버스)",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Optuna 시행 횟수 (기본값: 50, 타겟별 개별 튜닝)",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Optuna 튜닝 스킵 (기본 파라미터 사용)",
    )
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="Meta-labeling 스킵",
    )
    parser.add_argument(
        "--skip-catboost",
        action="store_true",
        help="CatBoost 앙상블 스킵",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="병렬 학습 타겟 수 (기본값: 2)",
    )
    args = parser.parse_args()

    # 심볼 목록
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_universe()

    print("=" * 70)
    print("30종목 배치 학습 파이프라인")
    print("=" * 70)
    print(f"심볼: {len(symbols)}개")
    if len(symbols) <= 10:
        print(f"  {', '.join(symbols)}")
    else:
        print(f"  {', '.join(symbols[:5])} ... {', '.join(symbols[-3:])}")

    print(f"\n학습 설정:")
    print(f"  - 학습 방식: 배치 (전체 데이터 한번에)")
    print(f"  - 멀티 타임프레임: Yes (1m, 15m, 1h)")
    print(f"  - Optuna: {'No' if args.skip_optuna else f'Yes ({args.optuna_trials} trials × 8 targets)'}")
    print(f"  - Meta-labeling: {'No' if args.skip_meta else 'Yes'}")
    print(f"  - CatBoost 앙상블: {'No' if args.skip_catboost else 'Yes'}")
    print(f"  - Feature Selection: 누적 중요도 99.9%")
    print(f"  - 병렬 학습: {args.parallel}개 동시")

    # 데이터 범위 확인
    min_ts, max_ts = get_data_range()
    if not min_ts or not max_ts:
        print("\nERROR: 데이터가 없습니다")
        sys.exit(1)

    # Train/Val 분할 (마지막 1달 = Val)
    val_end = max_ts
    val_start = val_end - timedelta(days=30)
    train_end = val_start - timedelta(days=1)
    train_start = min_ts

    print(f"\n데이터 범위:")
    print(f"  Train: {train_start.date()} ~ {train_end.date()}")
    print(f"  Val: {val_start.date()} ~ {val_end.date()}")

    # Label spec hash
    label_config = LabelingConfig()
    spec = label_config.spec()
    spec_hash = spec.hash()
    print(f"  Label spec: {spec_hash}")

    # 학습 설정
    cfg = ImprovedTrainConfig(
        label_spec_hash=spec_hash,
        feature_schema_version=4,  # v4로 업데이트
        train_start=train_start.strftime("%Y-%m-%d"),
        train_end=train_end.strftime("%Y-%m-%d"),
        val_start=val_start.strftime("%Y-%m-%d"),
        val_end=val_end.strftime("%Y-%m-%d"),
        # 롱 + 숏 모두 학습
        targets=("er_long", "q05_long", "e_mae_long", "e_hold_long",
                 "er_short", "q05_short", "e_mae_short", "e_hold_short"),
        use_multi_tf=True,  # 멀티 타임프레임 활성화
        use_optuna=not args.skip_optuna,
        optuna_trials=args.optuna_trials,
        use_feature_selection=True,
        feature_importance_threshold=0.001,  # 누적 중요도 99.9% 기준
        use_meta_labeling=not args.skip_meta,
        use_catboost_ensemble=not args.skip_catboost,
        parallel_workers=args.parallel,  # 병렬 학습
    )

    # 학습 실행
    print("\n" + "-" * 70)
    result = run_improved_training(cfg, symbols=symbols)

    # 결과 출력
    print_results(result)

    # 메모리 정리
    gc.collect()


if __name__ == "__main__":
    main()
