#!/usr/bin/env python3
"""
Phase 1 + Phase 2 개선된 학습 파이프라인

Phase 1:
- LightGBM 하이퍼파라미터 튜닝 (Optuna)
- 피처 선택
- 클래스 불균형 처리

Phase 2:
- Meta-labeling
- Triple Barrier 최적화
- CatBoost 앙상블

Usage:
    python scripts/train_phase1_phase2.py
    python scripts/train_phase1_phase2.py --skip-tb-opt  # Triple Barrier 최적화 스킵
    python scripts/train_phase1_phase2.py --optuna-trials 100  # Optuna 시행 횟수 증가
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from typing import Dict, List, Any

import pandas as pd

from packages.common.db import fetch_all, get_conn
from services.labeling.pipeline import LabelingConfig, run_labeling
from services.training.train_improved import ImprovedTrainConfig, run_improved_training

# 대상 심볼
TARGET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOTUSDT"]


def get_data_range() -> tuple:
    """데이터 범위 확인"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(ts), MAX(ts) FROM candles_1m WHERE symbol = 'BTCUSDT'")
            row = cur.fetchone()
            return row[0], row[1]


def optimize_triple_barrier(symbols: List[str]) -> Dict[str, Any]:
    """
    Triple Barrier 파라미터 최적화

    테스트할 파라미터:
    - k_tp: [1.0, 1.5, 2.0, 2.5]  (TP 배수)
    - k_sl: [0.5, 0.75, 1.0, 1.5] (SL 배수)
    - h_bars: [180, 360, 720]     (홀딩 기간: 3h, 6h, 12h)
    """
    print("\n" + "=" * 60)
    print("Triple Barrier 파라미터 최적화")
    print("=" * 60)

    # 테스트할 파라미터 조합
    k_tp_values = [1.0, 1.5, 2.0, 2.5]
    k_sl_values = [0.5, 0.75, 1.0, 1.5]
    h_bars_values = [180, 360, 720]

    # 모든 조합 테스트는 시간이 오래 걸리므로 핵심 조합만 테스트
    test_configs = [
        # 기본 설정
        {"k_tp": 1.5, "k_sl": 1.0, "h_bars": 360, "name": "baseline"},
        # TP 조정
        {"k_tp": 2.0, "k_sl": 1.0, "h_bars": 360, "name": "tp_2.0"},
        {"k_tp": 1.0, "k_sl": 1.0, "h_bars": 360, "name": "tp_1.0"},
        # SL 조정
        {"k_tp": 1.5, "k_sl": 0.75, "h_bars": 360, "name": "sl_0.75"},
        {"k_tp": 1.5, "k_sl": 1.5, "h_bars": 360, "name": "sl_1.5"},
        # 홀딩 기간 조정
        {"k_tp": 1.5, "k_sl": 1.0, "h_bars": 180, "name": "h_180"},
        {"k_tp": 1.5, "k_sl": 1.0, "h_bars": 720, "name": "h_720"},
        # 최적 후보 조합
        {"k_tp": 2.0, "k_sl": 0.75, "h_bars": 360, "name": "tp2_sl075"},
        {"k_tp": 1.5, "k_sl": 0.75, "h_bars": 480, "name": "tp15_sl075_h480"},
    ]

    results = []

    for i, config in enumerate(test_configs):
        print(f"\n[{i+1}/{len(test_configs)}] Testing: {config['name']}")
        print(f"  k_tp={config['k_tp']}, k_sl={config['k_sl']}, h_bars={config['h_bars']}")

        # 라벨링 설정
        label_config = LabelingConfig(
            k_tp=config["k_tp"],
            k_sl=config["k_sl"],
            h_bars=config["h_bars"],
        )
        spec = label_config.spec()
        spec_hash = spec.hash()

        # 라벨링 실행
        stats = run_labeling(config=label_config, symbols=symbols, force_full=True)
        print(f"  → Labels: {stats.total_new_labels:,}")

        # 라벨 통계 조회
        label_stats = _get_label_stats(spec_hash, symbols)
        print(f"  → TP hit rate: {label_stats['tp_rate']*100:.1f}%, "
              f"SL hit rate: {label_stats['sl_rate']*100:.1f}%, "
              f"Timeout rate: {label_stats['timeout_rate']*100:.1f}%")
        print(f"  → Avg return: {label_stats['avg_ret']*100:.3f}%, "
              f"Win rate: {label_stats['win_rate']*100:.1f}%")

        results.append({
            "config": config,
            "spec_hash": spec_hash,
            "stats": label_stats,
        })

    # 결과 요약
    print("\n" + "=" * 60)
    print("Triple Barrier 최적화 결과")
    print("=" * 60)

    print(f"\n{'Config':<20} {'TP Rate':<10} {'SL Rate':<10} {'Avg Ret':<12} {'Win Rate':<10}")
    print("-" * 70)

    best_result = None
    best_score = -999

    for r in results:
        cfg = r["config"]
        s = r["stats"]
        print(f"{cfg['name']:<20} {s['tp_rate']*100:>6.1f}%   {s['sl_rate']*100:>6.1f}%   "
              f"{s['avg_ret']*100:>8.3f}%   {s['win_rate']*100:>6.1f}%")

        # 점수 계산: avg_ret * win_rate - 패널티 (너무 낮은 거래 수)
        score = s["avg_ret"] * s["win_rate"]
        if score > best_score and s["total"] > 1000:
            best_score = score
            best_result = r

    if best_result:
        print(f"\n  Best config: {best_result['config']['name']}")
        print(f"  spec_hash: {best_result['spec_hash']}")

    return best_result


def _get_label_stats(spec_hash: str, symbols: List[str]) -> Dict[str, float]:
    """라벨 통계 조회"""
    placeholders = ",".join(["%s"] * len(symbols))
    query = f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END) as tp_count,
            SUM(CASE WHEN y = -1 THEN 1 ELSE 0 END) as sl_count,
            SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END) as timeout_count,
            AVG(ret_net) as avg_ret,
            SUM(CASE WHEN ret_net > 0 THEN 1 ELSE 0 END) as win_count
        FROM labels_long_1m
        WHERE spec_hash = %s AND symbol IN ({placeholders})
    """
    params = [spec_hash] + symbols
    rows = fetch_all(query, tuple(params))

    if not rows or rows[0][0] == 0:
        return {"total": 0, "tp_rate": 0, "sl_rate": 0, "timeout_rate": 0, "avg_ret": 0, "win_rate": 0}

    total, tp, sl, timeout, avg_ret, wins = rows[0]
    total = total or 1

    return {
        "total": total,
        "tp_rate": (tp or 0) / total,
        "sl_rate": (sl or 0) / total,
        "timeout_rate": (timeout or 0) / total,
        "avg_ret": avg_ret or 0,
        "win_rate": (wins or 0) / total,
    }


def run_improved_training_pipeline(
    symbols: List[str],
    spec_hash: str,
    optuna_trials: int = 50,
    skip_optuna: bool = False,
    skip_meta: bool = False,
    skip_catboost: bool = False,
) -> Dict[str, Any]:
    """개선된 학습 파이프라인 실행"""
    print("\n" + "=" * 60)
    print("개선된 학습 파이프라인 (Phase 1 + Phase 2)")
    print("=" * 60)

    # 데이터 범위 확인
    min_ts, max_ts = get_data_range()
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
    print(f"  Label spec hash: {spec_hash}")

    print(f"\n  설정:")
    print(f"    - Optuna 튜닝: {'Yes' if not skip_optuna else 'No'} ({optuna_trials} trials)")
    print(f"    - 피처 선택: Yes")
    print(f"    - Meta-labeling: {'Yes' if not skip_meta else 'No'}")
    print(f"    - CatBoost 앙상블: {'Yes' if not skip_catboost else 'No'}")

    # 학습 설정
    cfg = ImprovedTrainConfig(
        label_spec_hash=spec_hash,
        feature_schema_version=3,
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        val_start=val_start.isoformat(),
        val_end=val_end.isoformat(),
        targets=("er_long", "q05_long", "e_mae_long", "e_hold_long"),
        use_multi_tf=True,
        use_optuna=not skip_optuna,
        optuna_trials=optuna_trials,
        use_feature_selection=True,
        feature_importance_threshold=0.001,
        handle_imbalance=True,
        use_meta_labeling=not skip_meta,
        meta_threshold=0.5,
        use_catboost_ensemble=not skip_catboost,
        ensemble_weights=(0.6, 0.4),
    )

    # 학습 실행
    result = run_improved_training(cfg, symbols=symbols)

    return result


def print_final_results(result: Dict[str, Any], tb_result: Dict[str, Any] = None) -> None:
    """최종 결과 출력"""
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)

    if result.get("status") != "ok":
        print(f"Error: {result.get('message', 'Unknown error')}")
        return

    metrics = result.get("metrics", {})
    report = result.get("report", {})

    # Triple Barrier 최적화 결과
    if tb_result:
        print("\n[Triple Barrier 최적화]")
        cfg = tb_result["config"]
        print(f"  Best config: k_tp={cfg['k_tp']}, k_sl={cfg['k_sl']}, h_bars={cfg['h_bars']}")
        print(f"  TP hit rate: {tb_result['stats']['tp_rate']*100:.1f}%")
        print(f"  Win rate: {tb_result['stats']['win_rate']*100:.1f}%")

    # 모델 성능
    print("\n[모델 성능 (RMSE)]")
    print("-" * 40)
    for target in ["er_long", "q05_long", "e_mae_long", "e_hold_long"]:
        if target in metrics:
            m = metrics[target]
            print(f"  {target:<15}: {m.get('rmse', 0):.4f}")

    # 필터링된 거래 지표
    print("\n[거래 지표]")
    print("-" * 40)

    # 베이스라인
    trade = metrics.get("trade", {})
    pf = trade.get("profit_factor", 0)
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  Baseline:     PF={pf_str}, Expectancy={trade.get('expectancy', 0)*100:.2f}%, "
          f"Trades={int(trade.get('turnover', 0)):,}")

    # er>0 필터
    er0 = metrics.get("filtered_er0", {})
    if er0:
        pf = er0.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  er>0:         PF={pf_str}, Expectancy={er0.get('expectancy', 0)*100:.2f}%, "
              f"Trades={int(er0.get('turnover', 0)):,}")

    # er>0.001 필터
    er001 = metrics.get("filtered_er001", {})
    if er001:
        pf = er001.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  er>0.001:     PF={pf_str}, Expectancy={er001.get('expectancy', 0)*100:.2f}%, "
              f"Trades={int(er001.get('turnover', 0)):,}")

    # Meta-labeling 필터
    meta = metrics.get("filtered_meta", {})
    if meta:
        pf = meta.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  er>0 + meta:  PF={pf_str}, Expectancy={meta.get('expectancy', 0)*100:.2f}%, "
              f"Trades={int(meta.get('turnover', 0)):,}")

    # 심볼별 성능
    by_symbol = metrics.get("by_symbol", {})
    if by_symbol:
        print("\n[심볼별 성능]")
        print("-" * 40)
        for symbol, sm in sorted(by_symbol.items(), key=lambda x: x[1].get("expectancy", 0), reverse=True):
            pf = sm.get("profit_factor", 0)
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            exp_pct = sm.get("expectancy", 0) * 100
            exp_sign = "+" if exp_pct > 0 else ""
            print(f"  {symbol:<12}: PF={pf_str}, Expectancy={exp_sign}{exp_pct:.2f}%")

    # 모델 정보
    print("\n[모델 정보]")
    print("-" * 40)
    print(f"  Model ID: {result.get('model_id', 'N/A')}")
    print(f"  Feature Count: {report.get('feature_count', 'N/A')}")
    print(f"  Improvements: {report.get('improvements', {})}")

    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    best_exp = 0
    best_filter = "baseline"
    for name, m in [("baseline", trade), ("er>0", er0), ("er>0.001", er001), ("meta", meta)]:
        if m and m.get("expectancy", 0) > best_exp:
            best_exp = m.get("expectancy", 0)
            best_filter = name

    if best_exp > 0:
        print(f"  ✓ 양수 기대값 달성! ({best_filter}: +{best_exp*100:.2f}%)")
        print(f"  ✓ 실거래 테스트 권장")
    else:
        print(f"  △ 최고 기대값: {best_exp*100:.2f}% ({best_filter})")
        print(f"  → 추가 개선 필요")

    # 양수 기대값 심볼
    positive_symbols = [s for s, m in by_symbol.items() if m.get("expectancy", 0) > 0]
    if positive_symbols:
        print(f"\n  양수 기대값 심볼: {', '.join(positive_symbols)}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 + Phase 2 학습 파이프라인")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=TARGET_SYMBOLS,
        help=f"학습할 심볼 (기본값: {TARGET_SYMBOLS})",
    )
    parser.add_argument(
        "--skip-tb-opt",
        action="store_true",
        help="Triple Barrier 최적화 스킵",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Optuna 하이퍼파라미터 튜닝 스킵",
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
        "--optuna-trials",
        type=int,
        default=50,
        help="Optuna 시행 횟수 (기본값: 50)",
    )
    parser.add_argument(
        "--spec-hash",
        type=str,
        default=None,
        help="특정 label spec hash 사용 (Triple Barrier 최적화 스킵 시 필요)",
    )
    args = parser.parse_args()

    symbols = args.symbols

    print("=" * 70)
    print("Phase 1 + Phase 2 학습 파이프라인")
    print("=" * 70)
    print(f"심볼: {', '.join(symbols)}")
    print(f"Optuna trials: {args.optuna_trials}")

    # Triple Barrier 최적화
    tb_result = None
    if not args.skip_tb_opt:
        tb_result = optimize_triple_barrier(symbols)
        if tb_result:
            spec_hash = tb_result["spec_hash"]
        else:
            # 기본 설정 사용
            label_config = LabelingConfig()
            spec_hash = label_config.spec().hash()
    else:
        if args.spec_hash:
            spec_hash = args.spec_hash
        else:
            # 기본 설정의 spec_hash 사용
            label_config = LabelingConfig()
            spec_hash = label_config.spec().hash()
            print(f"\nUsing default label spec: {spec_hash}")

    # 개선된 학습 실행
    result = run_improved_training_pipeline(
        symbols=symbols,
        spec_hash=spec_hash,
        optuna_trials=args.optuna_trials,
        skip_optuna=args.skip_optuna,
        skip_meta=args.skip_meta,
        skip_catboost=args.skip_catboost,
    )

    # 결과 출력
    print_final_results(result, tb_result)


if __name__ == "__main__":
    main()
