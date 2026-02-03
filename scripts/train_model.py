"""
1년 데이터로 모델 훈련
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

from packages.common.db import fetch_all
from services.training.train import TrainConfig, run_training_job


def get_available_data_stats():
    """훈련 가능한 데이터 통계 확인"""
    rows = fetch_all("""
        SELECT
            COUNT(*) as total_rows,
            MIN(f.ts) as min_ts,
            MAX(f.ts) as max_ts
        FROM features_1m f
        JOIN labels_long_1m l ON f.symbol = l.symbol AND f.ts = l.ts
        WHERE f.schema_version = 1
        AND f.symbol IN ('BTCUSDT', 'ETHUSDT')
    """)
    return rows[0] if rows else None


def main() -> None:
    print("=== 모델 훈련 시작 ===")

    # 훈련 기간 설정
    # train: 300일, val: 30일
    end_date = datetime(2026, 1, 31)
    val_end = end_date
    val_start = end_date - timedelta(days=30)
    train_end = val_start
    train_start = train_end - timedelta(days=300)

    print(f"Train: {train_start.date()} ~ {train_end.date()}")
    print(f"Val: {val_start.date()} ~ {val_end.date()}")

    cfg = TrainConfig(
        label_spec_hash='722ababe946d22b0',
        feature_schema_version=1,
        train_start=train_start.strftime('%Y-%m-%d'),
        train_end=train_end.strftime('%Y-%m-%d'),
        val_start=val_start.strftime('%Y-%m-%d'),
        val_end=val_end.strftime('%Y-%m-%d'),
        targets=('er_long', 'q05_long', 'e_mae_long', 'e_hold_long'),
        algo='lgbm',
        label_spec={
            'k_tp': 1.5,
            'k_sl': 1.0,
            'h_bars': 360,
            'fee_rate': 0.0004,
            'slippage_k': 0.15
        }
    )

    print("훈련 중...")
    result = run_training_job(cfg)

    if result.get('status') == 'empty':
        print(f"실패: {result.get('message')}")
        sys.exit(1)

    print("=== 훈련 완료 ===")
    print(f"Model ID: {result.get('model_id')}")

    metrics = result.get('metrics', {})
    trade = metrics.get('trade', {})
    print(f"Profit Factor: {trade.get('profit_factor', 'N/A'):.2f}")
    print(f"Max Drawdown: {trade.get('max_drawdown', 'N/A'):.4f}")
    print(f"Expectancy: {trade.get('expectancy', 'N/A'):.6f}")
    print(f"Turnover: {trade.get('turnover', 'N/A')}")


if __name__ == "__main__":
    main()
