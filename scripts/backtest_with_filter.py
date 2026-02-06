"""ER 필터 적용 백테스트"""
import sys
sys.path.insert(0, '/Users/jongkwankim/my-work/ta')

import pandas as pd
import numpy as np
from packages.common.db import fetch_all

INITIAL = 100.0
POSITION_SIZE = 30.0
LEVERAGE = 10
FEE = 0.0004
SLIPPAGE = 0.0002

print("데이터 로딩 중...")

# 라벨 + 예측 조인
query = """
SELECT
    l.symbol,
    l.ts,
    'LONG' as direction,
    l.ret_net,
    l.y,
    COALESCE((f.features->>'er_long')::float, 0) as er_pred
FROM labels_long_1m l
JOIN features_1m f ON l.symbol = f.symbol AND l.ts = f.ts
WHERE l.ts >= '2026-01-29'
  AND l.spec_hash = '8c8d570e2343c185'
UNION ALL
SELECT
    l.symbol,
    l.ts,
    'SHORT' as direction,
    l.ret_net,
    l.y,
    COALESCE((f.features->>'er_short')::float, 0) as er_pred
FROM labels_short_1m l
JOIN features_1m f ON l.symbol = f.symbol AND l.ts = f.ts
WHERE l.ts >= '2026-01-29'
  AND l.spec_hash = '8c8d570e2343c185'
ORDER BY ts
"""

rows = fetch_all(query)
print(f"데이터: {len(rows):,}개")

df = pd.DataFrame(rows, columns=['symbol', 'ts', 'direction', 'ret_net', 'y', 'er_pred'])
df['ts'] = pd.to_datetime(df['ts'], utc=True)
df['er_pred'] = df['er_pred'].fillna(0).astype(float)

print(f"기간: {df['ts'].min().date()} ~ {df['ts'].max().date()}")
days = (df['ts'].max() - df['ts'].min()).days + 1
print(f"일수: {days}일")

# ER 필터별 백테스트
ER_THRESHOLDS = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]

print("\n" + "="*80)
print(f"ER 필터별 백테스트 ($100 시작, 포지션 $30, 레버리지 10x)")
print("="*80)

results = []

for threshold in ER_THRESHOLDS:
    # 필터 적용
    filtered = df[df['er_pred'] > threshold].copy()

    if len(filtered) == 0:
        continue

    # 결과 계산
    long_df = filtered[filtered['direction'] == 'LONG']
    short_df = filtered[filtered['direction'] == 'SHORT']

    # PF 계산
    long_profit = long_df[long_df['ret_net'] > 0]['ret_net'].sum()
    long_loss = abs(long_df[long_df['ret_net'] < 0]['ret_net'].sum())
    long_pf = long_profit / long_loss if long_loss > 0 else 0

    short_profit = short_df[short_df['ret_net'] > 0]['ret_net'].sum()
    short_loss = abs(short_df[short_df['ret_net'] < 0]['ret_net'].sum())
    short_pf = short_profit / short_loss if short_loss > 0 else 0

    # 복리 백테스트
    capital = INITIAL
    trades = 0
    wins = 0

    for _, row in filtered.sort_values('ts').iterrows():
        ret = row['ret_net'] - (FEE + SLIPPAGE) * 2
        pnl = POSITION_SIZE * LEVERAGE * ret
        capital += pnl
        trades += 1
        if ret > 0:
            wins += 1

    win_rate = wins / trades * 100 if trades > 0 else 0
    pnl_pct = (capital - INITIAL) / INITIAL * 100
    trades_per_day = trades / days

    results.append({
        'threshold': threshold,
        'trades': trades,
        'trades_per_day': trades_per_day,
        'long_pf': long_pf,
        'short_pf': short_pf,
        'win_rate': win_rate,
        'final': capital,
        'pnl_pct': pnl_pct,
    })

print(f"\n{'Filter':<10} {'Trades':>10} {'/Day':>8} {'LongPF':>8} {'ShortPF':>8} {'WinRate':>8} {'Final$':>12} {'Return':>10}")
print("-"*80)

for r in results:
    print(f"er>{r['threshold']:<8} {r['trades']:>10,} {r['trades_per_day']:>7.0f} {r['long_pf']:>8.2f} {r['short_pf']:>8.2f} {r['win_rate']:>7.1f}% ${r['final']:>10.2f} {r['pnl_pct']:>+9.1f}%")

# 추천
print("\n" + "="*80)
print("분석 및 추천")
print("="*80)

for r in results:
    if r['threshold'] == 0.002:
        print(f"""
현재 설정 (er>0.002):
  - 거래 수: {r['trades']:,}개 ({r['trades_per_day']:.0f}/일)
  - Long PF: {r['long_pf']:.2f}
  - Short PF: {r['short_pf']:.2f}
  - 승률: {r['win_rate']:.1f}%
  - {days}일 후 자본: ${r['final']:.2f}
  - 수익률: {r['pnl_pct']:+.1f}%
""")

# 수익 가능 필터만 표시
print("수익 가능한 필터:")
for r in results:
    if r['pnl_pct'] > 0:
        print(f"  er>{r['threshold']}: ${r['final']:.2f} ({r['pnl_pct']:+.1f}%)")
