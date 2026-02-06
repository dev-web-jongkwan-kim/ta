"""Backtest different ER filter thresholds with compounding."""
import sys
sys.path.insert(0, '/Users/jongkwankim/my-work/ta')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from packages.common.db import fetch_all
from services.inference.predictor import Predictor

# Settings
INITIAL_CAPITAL = 100.0
POSITION_SIZE_PCT = 0.1  # 10% per trade
LEVERAGE = 10
FEE_RATE = 0.0004
SLIPPAGE = 0.0002

# ER filter thresholds to test
ER_THRESHOLDS = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]

# Model to test
MODEL_ID = 'c87b3140-ec29-444a-9b11-6823cd0a64c9'

# Date range - last 7 days
END_DATE = datetime.now(timezone.utc).date() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=7)

print(f"백테스트 기간: {START_DATE} ~ {END_DATE}")
print(f"초기 자본: ${INITIAL_CAPITAL}")
print(f"포지션 크기: {POSITION_SIZE_PCT*100}% (레버리지 {LEVERAGE}x)")
print()

# Load predictor
predictor = Predictor(MODEL_ID)
print(f"모델 로드 완료: {MODEL_ID[:8]}...")

# Get all symbols
symbols = fetch_all("SELECT DISTINCT symbol FROM features_1m")
symbols = [s[0] for s in symbols]
print(f"심볼 수: {len(symbols)}")

# Load features and labels for each symbol
all_data = []

for symbol in symbols:
    # Load 1m features
    rows_1m = fetch_all("""
        SELECT ts, features, atr
        FROM features_1m
        WHERE symbol = %s AND ts >= %s AND ts < %s
        ORDER BY ts
    """, (symbol, START_DATE, END_DATE + timedelta(days=1)))

    if not rows_1m:
        continue

    df_1m = pd.DataFrame(rows_1m, columns=['ts', 'features_1m', 'atr_1m'])
    df_1m['ts'] = pd.to_datetime(df_1m['ts'], utc=True)
    df_1m['symbol'] = symbol

    # Load 15m features
    rows_15m = fetch_all("""
        SELECT ts, features, atr
        FROM features_15m
        WHERE symbol = %s AND ts >= %s AND ts < %s
        ORDER BY ts
    """, (symbol, START_DATE, END_DATE + timedelta(days=1)))

    if rows_15m:
        df_15m = pd.DataFrame(rows_15m, columns=['ts', 'features_15m', 'atr_15m'])
        df_15m['ts'] = pd.to_datetime(df_15m['ts'], utc=True)
        df_1m = pd.merge_asof(df_1m.sort_values('ts'), df_15m.sort_values('ts'),
                              on='ts', direction='backward')
    else:
        df_1m['features_15m'] = None
        df_1m['atr_15m'] = None

    # Load 1h features
    rows_1h = fetch_all("""
        SELECT ts, features
        FROM features_1h
        WHERE symbol = %s AND ts >= %s AND ts < %s
        ORDER BY ts
    """, (symbol, START_DATE, END_DATE + timedelta(days=1)))

    if rows_1h:
        df_1h = pd.DataFrame(rows_1h, columns=['ts', 'features_1h'])
        df_1h['ts'] = pd.to_datetime(df_1h['ts'], utc=True)
        df_1m = pd.merge_asof(df_1m.sort_values('ts'), df_1h.sort_values('ts'),
                              on='ts', direction='backward')
    else:
        df_1m['features_1h'] = None

    # Load long labels
    rows_long = fetch_all("""
        SELECT ts, y, ret_net, mae, time_to_event_min
        FROM labels_long_1m
        WHERE symbol = %s AND ts >= %s AND ts < %s AND spec_hash = '8c8d570e2343c185'
        ORDER BY ts
    """, (symbol, START_DATE, END_DATE + timedelta(days=1)))

    if rows_long:
        df_long = pd.DataFrame(rows_long, columns=['ts', 'y_long', 'ret_long', 'mae_long', 'hold_long'])
        df_long['ts'] = pd.to_datetime(df_long['ts'], utc=True)
        df_1m = pd.merge(df_1m, df_long, on='ts', how='left')
    else:
        df_1m['y_long'] = None
        df_1m['ret_long'] = None
        df_1m['mae_long'] = None
        df_1m['hold_long'] = None

    # Load short labels
    rows_short = fetch_all("""
        SELECT ts, y, ret_net, mae, time_to_event_min
        FROM labels_short_1m
        WHERE symbol = %s AND ts >= %s AND ts < %s AND spec_hash = '8c8d570e2343c185'
        ORDER BY ts
    """, (symbol, START_DATE, END_DATE + timedelta(days=1)))

    if rows_short:
        df_short = pd.DataFrame(rows_short, columns=['ts', 'y_short', 'ret_short', 'mae_short', 'hold_short'])
        df_short['ts'] = pd.to_datetime(df_short['ts'], utc=True)
        df_1m = pd.merge(df_1m, df_short, on='ts', how='left')
    else:
        df_1m['y_short'] = None
        df_1m['ret_short'] = None
        df_1m['mae_short'] = None
        df_1m['hold_short'] = None

    all_data.append(df_1m)

print(f"데이터 로드 완료")

# Combine all data
df = pd.concat(all_data, ignore_index=True)
df = df.sort_values(['ts', 'symbol']).reset_index(drop=True)
print(f"총 샘플 수: {len(df):,}")

# Run predictions for all samples
print("\n예측 실행 중...")

predictions = []
for idx, row in df.iterrows():
    if idx % 50000 == 0:
        print(f"  {idx:,} / {len(df):,}")

    # Build multi-tf features
    model_features = {}

    # 1m features
    if row['features_1m']:
        f1m = row['features_1m'] if isinstance(row['features_1m'], dict) else {}
        for k, v in f1m.items():
            model_features[f'f_1m_{k}'] = v

    # 15m features
    if row.get('features_15m'):
        f15m = row['features_15m'] if isinstance(row['features_15m'], dict) else {}
        for k, v in f15m.items():
            model_features[f'f_15m_{k}'] = v

    # 1h features
    if row.get('features_1h'):
        f1h = row['features_1h'] if isinstance(row['features_1h'], dict) else {}
        for k, v in f1h.items():
            model_features[f'f_1h_{k}'] = v

    if not model_features:
        predictions.append({'er_long': 0, 'er_short': 0})
        continue

    try:
        preds = predictor.predict(model_features)
        predictions.append(preds)
    except Exception:
        predictions.append({'er_long': 0, 'er_short': 0})

df['er_long_pred'] = [p.get('er_long', 0) or 0 for p in predictions]
df['er_short_pred'] = [p.get('er_short', 0) or 0 for p in predictions]

print("예측 완료!")

# Backtest for each ER threshold
print("\n" + "="*70)
print("ER 필터별 백테스트 결과 (복리, $100 시작)")
print("="*70)

results = []

for threshold in ER_THRESHOLDS:
    capital = INITIAL_CAPITAL
    trades = 0
    wins = 0
    losses = 0
    total_return = 0

    # Filter signals
    long_signals = df[(df['er_long_pred'] > threshold) & (df['ret_long'].notna())].copy()
    short_signals = df[(df['er_short_pred'] > threshold) & (df['ret_short'].notna())].copy()

    # Process long trades
    for _, row in long_signals.iterrows():
        position_value = capital * POSITION_SIZE_PCT * LEVERAGE
        ret = row['ret_long'] - FEE_RATE * 2 - SLIPPAGE * 2
        pnl = position_value * ret
        capital += pnl
        total_return += ret
        trades += 1
        if ret > 0:
            wins += 1
        else:
            losses += 1

    # Process short trades
    for _, row in short_signals.iterrows():
        position_value = capital * POSITION_SIZE_PCT * LEVERAGE
        ret = row['ret_short'] - FEE_RATE * 2 - SLIPPAGE * 2
        pnl = position_value * ret
        capital += pnl
        total_return += ret
        trades += 1
        if ret > 0:
            wins += 1
        else:
            losses += 1

    win_rate = wins / trades * 100 if trades > 0 else 0
    pnl_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    results.append({
        'threshold': threshold,
        'trades': trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'pnl_pct': pnl_pct,
        'long_signals': len(long_signals),
        'short_signals': len(short_signals),
    })

    print(f"\ner>{threshold}:")
    print(f"  거래 수: {trades:,} (Long: {len(long_signals):,}, Short: {len(short_signals):,})")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  최종 자본: ${capital:.2f}")
    print(f"  수익률: {pnl_pct:+.1f}%")

# Summary table
print("\n" + "="*70)
print("요약")
print("="*70)
print(f"{'ER Filter':<12} {'Trades':>10} {'Win Rate':>10} {'Final $':>12} {'Return':>10}")
print("-"*56)
for r in results:
    print(f"er>{r['threshold']:<10} {r['trades']:>10,} {r['win_rate']:>9.1f}% ${r['final_capital']:>10.2f} {r['pnl_pct']:>+9.1f}%")

print("\n추천: 승률과 수익률의 균형을 보고 선택하세요.")
