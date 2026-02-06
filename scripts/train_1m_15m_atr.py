"""
1분봉 라벨 + 15m ATR + 멀티타임프레임 피처 모델 훈련
- 라벨: 1m (15m ATR 기반 TP/SL)
- 피처: 1m + 15m + 1h
- 방향: 롱 + 숏 양방향
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from packages.common.db import fetch_all
from services.training.train import TrainConfig, run_training_job


def get_available_data_range():
    """1분봉 학습 가능한 데이터 범위 확인"""
    rows = fetch_all("""
        SELECT
            MIN(f.ts) as min_ts,
            MAX(f.ts) as max_ts,
            COUNT(*) as total_rows
        FROM features_1m f
        JOIN labels_long_1m ll ON f.symbol = ll.symbol AND f.ts = ll.ts
        JOIN labels_short_1m ls ON f.symbol = ls.symbol AND f.ts = ls.ts
        WHERE f.schema_version = 4
        AND ll.spec_hash = '8c8d570e2343c185'
        AND ls.spec_hash = '8c8d570e2343c185'
    """)
    return rows[0] if rows else None


def main() -> None:
    print("=" * 60)
    print("1분봉 + 15m ATR + 멀티타임프레임 피처 모델 훈련")
    print("=" * 60)

    # 데이터 범위 확인
    data_range = get_available_data_range()
    if data_range and data_range[0]:
        print(f"사용 가능 데이터: {data_range[0]} ~ {data_range[1]} ({data_range[2]:,}개)")
    else:
        print("경고: 데이터가 없습니다!")
        sys.exit(1)

    # 동적 날짜 설정
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    # train: 데이터 시작 ~ val 시작 전 7일
    # val: 최근 7일간
    train_start = datetime(2025, 2, 2)
    val_start = datetime.combine(yesterday - timedelta(days=7), datetime.min.time())
    train_end = val_start
    val_end = datetime.combine(yesterday, datetime.max.time())

    print(f"Train: {train_start.date()} ~ {train_end.date()}")
    print(f"Val: {val_start.date()} ~ {val_end.date()}")

    cfg = TrainConfig(
        label_spec_hash='8c8d570e2343c185',  # 1m 라벨 (15m ATR 기반)
        feature_schema_version=4,
        train_start=train_start.strftime('%Y-%m-%d'),
        train_end=train_end.strftime('%Y-%m-%d'),
        val_start=val_start.strftime('%Y-%m-%d'),
        val_end=val_end.strftime('%Y-%m-%d'),
        targets=(
            'er_long', 'q05_long', 'e_mae_long', 'e_hold_long',
            'er_short', 'q05_short', 'e_mae_short', 'e_hold_short',
        ),
        algo='lgbm',
        timeframe='1m',  # 1분봉 기준
        use_multi_tf=True,  # 15분봉 + 1시간봉 피처 추가
        label_spec={
            'k_tp': 1.5,
            'k_sl': 1.0,
            'h_bars': 360,  # 6시간
            'fee_rate': 0.0004,
            'slippage_k': 0.15,
            'atr_timeframe': '15m',  # 15m ATR 사용
        }
    )

    print()
    print("Config:")
    print(f"  label_spec_hash: {cfg.label_spec_hash}")
    print(f"  timeframe: {cfg.timeframe}")
    print(f"  use_multi_tf: {cfg.use_multi_tf}")
    print(f"  targets: {cfg.targets}")
    print()
    print("훈련 중...")

    result = run_training_job(cfg)

    if result.get('status') == 'empty':
        print(f"실패: {result.get('message')}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("훈련 완료")
    print("=" * 60)
    print(f"Model ID: {result.get('model_id')}")
    print()

    metrics = result.get('metrics', {})

    # Baseline metrics - Long
    trade_long = metrics.get('trade_long', metrics.get('trade', {}))
    print("=== Baseline Long (전체 롱 거래) ===")
    print(f"Profit Factor: {trade_long.get('profit_factor', 0):.2f}")
    print(f"Expectancy: {trade_long.get('expectancy', 0)*100:.4f}%")
    print(f"Turnover: {int(trade_long.get('turnover', 0)):,}")
    print()

    # Baseline metrics - Short
    trade_short = metrics.get('trade_short', {})
    if trade_short:
        print("=== Baseline Short (전체 숏 거래) ===")
        print(f"Profit Factor: {trade_short.get('profit_factor', 0):.2f}")
        print(f"Expectancy: {trade_short.get('expectancy', 0)*100:.4f}%")
        print(f"Turnover: {int(trade_short.get('turnover', 0)):,}")
        print()

    # Filtered metrics - Long
    filtered_long = metrics.get('filtered_long', metrics.get('filtered', {}))
    if filtered_long:
        print("=== 필터별 성능 (Long) ===")
        print(f"{'Filter':<12} {'PF':>8} {'Expectancy':>12} {'Trades':>10}")
        print("-" * 45)
        for key in sorted(filtered_long.keys()):
            m = filtered_long[key]
            pf = m.get('profit_factor', 0)
            pf_str = f"{pf:.2f}" if pf < 100 else "inf"
            exp = m.get('expectancy', 0) * 100
            trades = int(m.get('turnover', 0))
            print(f"{key:<12} {pf_str:>8} {exp:>+11.4f}% {trades:>10,}")
        print()

    # Filtered metrics - Short
    filtered_short = metrics.get('filtered_short', {})
    if filtered_short:
        print("=== 필터별 성능 (Short) ===")
        print(f"{'Filter':<12} {'PF':>8} {'Expectancy':>12} {'Trades':>10}")
        print("-" * 45)
        for key in sorted(filtered_short.keys()):
            m = filtered_short[key]
            pf = m.get('profit_factor', 0)
            pf_str = f"{pf:.2f}" if pf < 100 else "inf"
            exp = m.get('expectancy', 0) * 100
            trades = int(m.get('turnover', 0))
            print(f"{key:<12} {pf_str:>8} {exp:>+11.4f}% {trades:>10,}")


if __name__ == "__main__":
    main()
