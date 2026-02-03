#!/usr/bin/env python3
"""
실제 모델 예측값 기반 백테스트

모델이 예측한 er_long 값을 기준으로 필터링하여 백테스트
"""
from __future__ import annotations

import json
import pickle
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd

from packages.common.db import fetch_all
from services.registry.storage import download_model

SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOTUSDT"]


def load_model_and_data():
    """모델과 검증 데이터 로드"""
    print("모델 및 데이터 로드 중...")

    # 최신 모델 정보 조회
    rows = fetch_all("""
        SELECT model_id, artifact_uri, feature_schema_version
        FROM models
        WHERE feature_schema_version = 3
        ORDER BY created_at DESC
        LIMIT 1
    """)

    if not rows:
        print("ERROR: 모델을 찾을 수 없습니다")
        return None, None, None

    model_id, artifact_uri, schema_version = rows[0]
    print(f"  Model ID: {model_id}")

    # 모델 다운로드 (artifact_uri에서 object_name 추출)
    # artifact_uri: s3://models/models/xxx.pkl -> object_name: models/xxx.pkl
    if artifact_uri.startswith("s3://"):
        parts = artifact_uri.replace("s3://", "").split("/", 1)
        object_name = parts[1] if len(parts) > 1 else artifact_uri
    else:
        object_name = artifact_uri

    models = download_model(object_name)
    er_model = models.get("er_long")

    if er_model is None:
        print("ERROR: er_long 모델을 찾을 수 없습니다")
        return None, None, None

    # 검증 데이터 로드 (멀티 타임프레임)
    all_data = []

    for symbol in SYMBOLS:
        # 1분 피처 + 라벨
        query_1m = """
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
        rows_1m = fetch_all(query_1m, (symbol,))
        if not rows_1m:
            continue

        df_1m = pd.DataFrame(rows_1m, columns=["symbol", "ts", "features", "ret_net", "y"])
        df_1m["ts"] = pd.to_datetime(df_1m["ts"], utc=True)

        # 15분 피처
        query_15m = """
            SELECT ts, features FROM features_15m
            WHERE symbol = %s AND ts >= '2026-01-02' AND ts <= '2026-02-01'
            ORDER BY ts
        """
        rows_15m = fetch_all(query_15m, (symbol,))

        # 1시간 피처
        query_1h = """
            SELECT ts, features FROM features_1h
            WHERE symbol = %s AND ts >= '2026-01-02' AND ts <= '2026-02-01'
            ORDER BY ts
        """
        rows_1h = fetch_all(query_1h, (symbol,))

        # 피처 파싱 및 결합
        def parse_features(f, prefix=""):
            if isinstance(f, str):
                d = json.loads(f)
            elif isinstance(f, (bytes, bytearray)):
                d = json.loads(f.decode())
            else:
                d = f or {}
            return {f"{prefix}{k}": v for k, v in d.items()}

        # 1분 피처 파싱
        features_list = []
        for _, row in df_1m.iterrows():
            f1m = parse_features(row["features"], "f_1m_")
            features_list.append(f1m)

        df_1m_features = pd.DataFrame(features_list)

        # 15분 피처 조인
        if rows_15m:
            df_15m = pd.DataFrame(rows_15m, columns=["ts", "features"])
            df_15m["ts"] = pd.to_datetime(df_15m["ts"], utc=True)
            df_15m_features = pd.DataFrame([parse_features(f, "f_15m_") for f in df_15m["features"]])
            df_15m_features["ts"] = df_15m["ts"].values
            df_15m_features = df_15m_features.set_index("ts")

            # Forward fill로 조인 (시간대 통일)
            df_1m = df_1m.set_index("ts")
            df_1m.index = df_1m.index.tz_convert("UTC") if df_1m.index.tz else df_1m.index.tz_localize("UTC")
            df_15m_features.index = df_15m_features.index.tz_convert("UTC") if df_15m_features.index.tz else df_15m_features.index.tz_localize("UTC")
            for col in df_15m_features.columns:
                df_1m[col] = df_15m_features[col].reindex(df_1m.index, method="ffill").values
            df_1m = df_1m.reset_index()

        # 1시간 피처 조인
        if rows_1h:
            df_1h = pd.DataFrame(rows_1h, columns=["ts", "features"])
            df_1h["ts"] = pd.to_datetime(df_1h["ts"], utc=True)
            df_1h_features = pd.DataFrame([parse_features(f, "f_1h_") for f in df_1h["features"]])
            df_1h_features["ts"] = df_1h["ts"].values
            df_1h_features = df_1h_features.set_index("ts")

            df_1m = df_1m.set_index("ts")
            df_1m.index = df_1m.index.tz_convert("UTC") if df_1m.index.tz else df_1m.index.tz_localize("UTC")
            df_1h_features.index = df_1h_features.index.tz_convert("UTC") if df_1h_features.index.tz else df_1h_features.index.tz_localize("UTC")
            for col in df_1h_features.columns:
                df_1m[col] = df_1h_features[col].reindex(df_1m.index, method="ffill").values
            df_1m = df_1m.reset_index()

        # 1분 피처 추가
        for col in df_1m_features.columns:
            df_1m[col] = df_1m_features[col].values

        all_data.append(df_1m)

    if not all_data:
        return None, None, None

    df = pd.concat(all_data, ignore_index=True)
    print(f"  → {len(df):,}개 샘플 로드")

    # 피처 컬럼 추출
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    return er_model, df, feature_cols


def calc_metrics(returns: pd.Series) -> Dict[str, float]:
    """거래 지표 계산"""
    if len(returns) == 0:
        return {"pf": 0, "expectancy": 0, "trades": 0, "win_rate": 0, "max_dd": 0}

    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    pf = float(pos / abs(neg)) if neg != 0 else float("inf")

    win_rate = len(returns[returns > 0]) / len(returns) * 100 if len(returns) > 0 else 0

    cum = returns.cumsum()
    peak = cum.cummax()
    max_dd = float((cum - peak).min())

    return {
        "pf": pf,
        "expectancy": float(returns.mean()),
        "trades": len(returns),
        "win_rate": win_rate,
        "max_dd": max_dd,
    }


def backtest_with_predictions(df: pd.DataFrame, er_model, feature_cols: list,
                               er_threshold: float = 0,
                               session: str = None,
                               exclude_hours: list = None) -> Dict[str, Any]:
    """모델 예측값 기반 백테스트"""
    # 피처 준비
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # er_long 예측
    er_pred = er_model.predict(X)
    df = df.copy()
    df["er_pred"] = er_pred

    # 시간 피처 추출
    df["hour"] = df["ts"].dt.hour
    df["is_asia"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
    df["is_us"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)

    # 필터 적용
    filtered = df.copy()

    # er_long threshold
    if er_threshold != 0:
        filtered = filtered[filtered["er_pred"] > er_threshold]

    # 세션 필터
    if session == "asia":
        filtered = filtered[filtered["is_asia"] == 1]
    elif session == "us":
        filtered = filtered[filtered["is_us"] == 1]

    # 시간 제외 필터
    if exclude_hours:
        filtered = filtered[~filtered["hour"].isin(exclude_hours)]

    return calc_metrics(filtered["ret_net"]), len(filtered)


def main():
    er_model, df, feature_cols = load_model_and_data()

    if er_model is None:
        return

    print(f"\n피처 수: {len(feature_cols)}")

    print("\n" + "=" * 70)
    print("1. er_long Threshold 분석 (모델 예측값 기반)")
    print("=" * 70)

    thresholds = [-0.002, -0.001, 0, 0.0005, 0.001, 0.0015, 0.002, 0.003]
    print(f"\n{'Threshold':<12} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10} {'Max DD':<10}")
    print("-" * 70)

    for th in thresholds:
        metrics, _ = backtest_with_predictions(df, er_model, feature_cols, er_threshold=th)
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{th:<12.4f} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%      {metrics['max_dd']:.4f}")

    print("\n" + "=" * 70)
    print("2. 세션별 분석 (er_long > 0)")
    print("=" * 70)

    sessions = [("All", None), ("Asia (0-8 UTC)", "asia"), ("US (13-21 UTC)", "us")]
    print(f"\n{'Session':<20} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 70)

    for name, session in sessions:
        metrics, _ = backtest_with_predictions(df, er_model, feature_cols, er_threshold=0, session=session)
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{name:<20} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%")

    print("\n" + "=" * 70)
    print("3. 시간대 제외 분석 (worst hours 제외)")
    print("=" * 70)

    # 시간대별 성과 먼저 계산
    hour_metrics = {}
    for hour in range(24):
        hour_df = df[df["ts"].dt.hour == hour]
        X_hour = hour_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        er_pred = er_model.predict(X_hour)
        filtered = hour_df[er_pred > 0]
        if len(filtered) > 0:
            m = calc_metrics(filtered["ret_net"])
            hour_metrics[hour] = m

    # 정렬하여 worst hours 찾기
    sorted_hours = sorted(hour_metrics.items(), key=lambda x: x[1]["expectancy"])
    worst_hours = [h for h, _ in sorted_hours[:4]]  # worst 4 hours

    print(f"\nWorst 4 hours (UTC): {worst_hours}")

    exclude_configs = [
        ("None", []),
        ("Worst 2", worst_hours[:2]),
        ("Worst 4", worst_hours[:4]),
        ("16-17 UTC", [16, 17]),
    ]

    print(f"\n{'Exclude':<20} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 70)

    for name, exclude in exclude_configs:
        metrics, _ = backtest_with_predictions(df, er_model, feature_cols, er_threshold=0, exclude_hours=exclude)
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{name:<20} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%")

    print("\n" + "=" * 70)
    print("4. 최적 조합 테스트")
    print("=" * 70)

    combinations = [
        {"name": "Baseline (er>0)", "threshold": 0, "session": None, "exclude": []},
        {"name": "er>0.001", "threshold": 0.001, "session": None, "exclude": []},
        {"name": "er>0.001 + US", "threshold": 0.001, "session": "us", "exclude": []},
        {"name": "er>0.001 + No 16-17", "threshold": 0.001, "session": None, "exclude": [16, 17]},
        {"name": "er>0.001 + US + No 16-17", "threshold": 0.001, "session": "us", "exclude": [16, 17]},
        {"name": "er>0.002 + US", "threshold": 0.002, "session": "us", "exclude": []},
        {"name": "er>0.002 + US + No 16-17", "threshold": 0.002, "session": "us", "exclude": [16, 17]},
    ]

    print(f"\n{'Config':<30} {'PF':<8} {'Expectancy':<12} {'Trades':<10} {'Win Rate':<10}")
    print("-" * 80)

    best_config = None
    best_expectancy = -999

    for combo in combinations:
        metrics, trades = backtest_with_predictions(
            df, er_model, feature_cols,
            er_threshold=combo["threshold"],
            session=combo.get("session"),
            exclude_hours=combo.get("exclude", [])
        )
        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"{combo['name']:<30} {pf_str:<8} {metrics['expectancy']:.4f}       {metrics['trades']:<10,} {metrics['win_rate']:.1f}%")

        if metrics["expectancy"] > best_expectancy and metrics["trades"] >= 1000:
            best_expectancy = metrics["expectancy"]
            best_config = combo
            best_config["metrics"] = metrics

    print("\n" + "=" * 70)
    print("5. 심볼별 최적 조합")
    print("=" * 70)

    for symbol in SYMBOLS:
        print(f"\n[{symbol}]")
        sym_df = df[df["symbol"] == symbol]

        best_sym = None
        best_sym_exp = -999

        for th in [0, 0.001, 0.002]:
            for session in [None, "us"]:
                for exclude in [[], [16, 17]]:
                    metrics, trades = backtest_with_predictions(
                        sym_df, er_model, feature_cols,
                        er_threshold=th, session=session, exclude_hours=exclude
                    )
                    if trades >= 500 and metrics["expectancy"] > best_sym_exp:
                        best_sym_exp = metrics["expectancy"]
                        best_sym = {"th": th, "session": session, "exclude": exclude, "metrics": metrics}

        if best_sym:
            m = best_sym["metrics"]
            session_str = best_sym["session"] or "all"
            exclude_str = str(best_sym["exclude"]) if best_sym["exclude"] else "none"
            pf_str = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "inf"
            print(f"  Best: threshold={best_sym['th']}, session={session_str}, exclude={exclude_str}")
            print(f"  PF={pf_str}, Expectancy={m['expectancy']:.4f}, Trades={m['trades']:,}, WinRate={m['win_rate']:.1f}%")

    # 최종 권장 설정
    print("\n" + "=" * 70)
    print("최종 권장 설정")
    print("=" * 70)

    if best_config:
        m = best_config["metrics"]
        print(f"\n  Config: {best_config['name']}")
        print(f"  - er_long threshold: {best_config['threshold']}")
        print(f"  - Session: {best_config.get('session') or 'all'}")
        print(f"  - Exclude hours: {best_config.get('exclude', [])}")
        print(f"\n  Results:")
        pf_str = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "inf"
        print(f"  - Profit Factor: {pf_str}")
        print(f"  - Expectancy: {m['expectancy']:.4f} ({m['expectancy']*100:.2f}%)")
        print(f"  - Trades: {m['trades']:,}")
        print(f"  - Win Rate: {m['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
