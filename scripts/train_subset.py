"""
주요 종목 서브셋으로 모델 훈련
메모리 효율적인 방식
"""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from packages.common.db import execute, fetch_all
from services.registry.storage import upload_json, upload_model


# 훈련할 종목 (주요 6개)
TRAIN_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']


def load_dataset_subset(
    label_spec_hash: str,
    feature_schema_version: int,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """제한된 심볼과 날짜 범위로 데이터 로드"""
    placeholders = ','.join(['%s'] * len(symbols))
    rows = fetch_all(
        f"""
        SELECT f.symbol, f.ts, f.features, f.schema_version, f.atr, f.funding_z, f.btc_regime,
               l.ret_net, l.mae, l.time_to_event_min
        FROM features_1m f
        JOIN labels_long_1m l
          ON f.symbol = l.symbol AND f.ts = l.ts
        WHERE f.schema_version = %s
          AND l.spec_hash = %s
          AND f.symbol IN ({placeholders})
          AND f.ts >= %s
          AND f.ts < %s
        ORDER BY f.ts
        """,
        (feature_schema_version, label_spec_hash, *symbols, start_date, end_date),
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "symbol", "ts", "features", "schema_version",
            "atr", "funding_z", "btc_regime",
            "ret_net", "mae", "time_to_event_min",
        ],
    )

    # JSON features 확장
    feats = pd.json_normalize(df["features"])
    feats.columns = [f"f_{c}" for c in feats.columns]

    # object 타입을 float로 변환 (None -> NaN)
    for col in feats.columns:
        if feats[col].dtype == 'object':
            feats[col] = pd.to_numeric(feats[col], errors='coerce')

    df = pd.concat([df.drop(columns=["features"]), feats], axis=1)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # NaN 피처를 0으로 채움
    for col in feats.columns:
        df[col] = df[col].fillna(0)

    df["feature_cols"] = [feats.columns.tolist()] * len(df)
    return df.sort_values("ts")


def trade_metrics(returns: pd.Series) -> Dict[str, float]:
    """거래 메트릭 계산"""
    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    profit_factor = float(pos / abs(neg)) if neg != 0 else float("inf")
    cum = returns.cumsum()
    peak = cum.cummax()
    drawdown = (cum - peak).min()
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    return {
        "profit_factor": profit_factor,
        "max_drawdown": float(drawdown),
        "expectancy": float(returns.mean()),
        "win_rate": win_rate,
        "n_trades": int(len(returns)),
    }


def train_regressor(X: pd.DataFrame, y: pd.Series, objective: str, alpha: float | None = None) -> lgb.LGBMRegressor:
    """LightGBM 회귀 모델 훈련"""
    params = {
        "objective": objective,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
    }
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def main() -> None:
    print("=== 서브셋 모델 훈련 ===")
    print(f"종목: {TRAIN_SYMBOLS}")

    # 날짜 설정 (메모리 고려해서 180일 훈련)
    end_date = datetime(2026, 1, 31)
    val_end = end_date
    val_start = end_date - timedelta(days=30)
    train_end = val_start
    train_start = train_end - timedelta(days=180)

    print(f"Train: {train_start.date()} ~ {train_end.date()} (180일)")
    print(f"Val: {val_start.date()} ~ {val_end.date()} (30일)")

    label_spec_hash = '722ababe946d22b0'
    feature_schema_version = 1

    # Train 데이터 로드
    print("\nTrain 데이터 로드 중...")
    train_df = load_dataset_subset(
        label_spec_hash, feature_schema_version,
        TRAIN_SYMBOLS,
        train_start.strftime('%Y-%m-%d'),
        train_end.strftime('%Y-%m-%d')
    )
    print(f"  Train 샘플: {len(train_df):,}개")

    if train_df.empty:
        print("Train 데이터 없음!")
        sys.exit(1)

    # Val 데이터 로드
    print("Val 데이터 로드 중...")
    val_df = load_dataset_subset(
        label_spec_hash, feature_schema_version,
        TRAIN_SYMBOLS,
        val_start.strftime('%Y-%m-%d'),
        val_end.strftime('%Y-%m-%d')
    )
    print(f"  Val 샘플: {len(val_df):,}개")

    if val_df.empty:
        print("Val 데이터 없음!")
        sys.exit(1)

    # 피처 추출
    feature_cols = train_df.iloc[0]["feature_cols"]
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    # 타겟 및 모델 훈련
    target_map = {
        "er_long": "ret_net",
        "q05_long": "ret_net",
        "e_mae_long": "mae",
        "e_hold_long": "time_to_event_min",
    }
    models = {}
    metrics = {}

    print("\n모델 훈련 중...")
    for target, raw_target in target_map.items():
        objective = "quantile" if target.startswith("q05") else "regression"
        alpha = 0.05 if objective == "quantile" else None

        print(f"  {target}...", end=" ", flush=True)
        model = train_regressor(X_train, train_df[raw_target], objective, alpha)
        preds = model.predict(X_val)

        rmse = float(mean_squared_error(val_df[raw_target], preds, squared=False))
        mae = float(mean_absolute_error(val_df[raw_target], preds))
        metrics[target] = {"rmse": rmse, "mae": mae}
        models[target] = model
        print(f"RMSE={rmse:.6f}")

    # 거래 메트릭
    metrics["trade"] = trade_metrics(val_df["ret_net"])

    # BTC Regime별 메트릭
    regime_metrics = {}
    for regime, group in val_df.groupby("btc_regime"):
        if not group.empty:
            regime_metrics[str(int(regime))] = trade_metrics(group["ret_net"])
    metrics["regime"] = regime_metrics

    # 비용 breakdown
    metrics["cost_breakdown"] = {
        "fee_per_trade": 0.0004 * 2,
        "slippage_per_trade": 0.15 * 0.0001 * 2,
        "funding_per_trade": 0.0,
    }

    # 모델 저장
    model_id = str(uuid.uuid4())
    artifact_name = f"models/{model_id}.pkl"
    artifact_uri = upload_model(models, artifact_name)

    # Report 저장
    report = {
        "model_id": model_id,
        "targets": list(target_map.keys()),
        "feature_schema_version": feature_schema_version,
        "label_spec_hash": label_spec_hash,
        "label_spec": {
            "k_tp": 1.5,
            "k_sl": 1.0,
            "h_bars": 360,
            "fee_rate": 0.0004,
            "slippage_k": 0.15,
        },
        "split_summary": {
            "train": {"start": train_start.isoformat(), "end": train_end.isoformat(), "n_samples": len(train_df)},
            "val": {"start": val_start.isoformat(), "end": val_end.isoformat(), "n_samples": len(val_df)},
        },
        "symbols": TRAIN_SYMBOLS,
        "metrics": metrics,
    }
    report_uri = upload_json(report, f"reports/{model_id}.json")

    # DB 저장
    execute(
        """
        INSERT INTO models (model_id, algo, feature_schema_version, label_spec_hash, train_start, train_end, metrics, artifact_uri, is_production)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            model_id,
            "lgbm",
            feature_schema_version,
            label_spec_hash,
            train_start,
            train_end,
            json.dumps(metrics),
            artifact_uri,
            False,
        ),
    )

    print("\n=== 훈련 완료 ===")
    print(f"Model ID: {model_id}")
    print(f"\n성능 지표:")
    trade = metrics["trade"]
    print(f"  Profit Factor: {trade['profit_factor']:.2f}")
    print(f"  Max Drawdown: {trade['max_drawdown']:.4f}")
    print(f"  Expectancy: {trade['expectancy']:.6f}")
    print(f"  Win Rate: {trade['win_rate']:.2%}")
    print(f"  N Trades: {trade['n_trades']:,}")


if __name__ == "__main__":
    main()
