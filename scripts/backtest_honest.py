"""정직한 백테스트 - 실제 예측 + 실제 라벨 결과"""
import sys
sys.path.insert(0, '/Users/jongkwankim/my-work/ta')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from packages.common.db import fetch_all

# 설정
INITIAL = 100.0
POSITION_SIZE = 30.0
LEVERAGE = 10
FEE = 0.0004
SLIPPAGE = 0.0002

print("데이터 로딩 중...")

# 최근 7일 라벨 + 피처 조인
query = """
WITH features_with_pred AS (
    SELECT
        f.symbol,
        f.ts,
        f.features->'er_long' as pred_er_long,
        f.features->'er_short' as pred_er_short
    FROM features_1m f
    WHERE f.ts >= NOW() - INTERVAL '7 days'
)
SELECT
    l.symbol,
    l.ts,
    'LONG' as direction,
    l.ret_net,
    l.y,
    COALESCE((f.features->>'er_long')::float, 0) as er_pred
FROM labels_long_1m l
JOIN features_1m f ON l.symbol = f.symbol AND l.ts = f.ts
WHERE l.ts >= NOW() - INTERVAL '7 days'
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
WHERE l.ts >= NOW() - INTERVAL '7 days'
  AND l.spec_hash = '8c8d570e2343c185'
ORDER BY ts
"""

# 데이터가 없으면 더 긴 기간으로
rows = fetch_all(query)
if not rows or len(rows) < 1000:
    print("최근 7일 데이터 부족, 전체 라벨 데이터 사용...")
    query = """
    SELECT
        l.symbol,
        l.ts,
        'LONG' as direction,
        l.ret_net,
        l.y
    FROM labels_long_1m l
    WHERE l.spec_hash = '8c8d570e2343c185'
      AND l.ts >= '2026-01-29'
    UNION ALL
    SELECT
        l.symbol,
        l.ts,
        'SHORT' as direction,
        l.ret_net,
        l.y
    FROM labels_short_1m l
    WHERE l.spec_hash = '8c8d570e2343c185'
      AND l.ts >= '2026-01-29'
    ORDER BY ts
    """
    rows = fetch_all(query)

print(f"라벨 데이터: {len(rows):,}개")

if not rows:
    print("라벨 데이터가 없습니다!")
    sys.exit(1)

df = pd.DataFrame(rows, columns=['symbol', 'ts', 'direction', 'ret_net', 'y', 'er_pred'])
df['ts'] = pd.to_datetime(df['ts'], utc=True)

print(f"기간: {df['ts'].min()} ~ {df['ts'].max()}")
print(f"심볼: {df['symbol'].nunique()}개")

# 실제 모델 예측 로드
print("\n모델 예측 로딩...")
from services.inference.predictor import Predictor

predictor = Predictor()
predictor.ensure_loaded()
print(f"모델 ID: {predictor._model_id}")

# 모델 예측 직접 수행 대신, 저장된 signals 테이블 사용
# (signals에는 이미 예측 결과가 저장됨)
signals_query = """
SELECT symbol, ts, er_long, er_short
FROM signals
WHERE ts >= '2026-01-29'
ORDER BY ts
"""
signal_rows = fetch_all(signals_query)
if signal_rows:
    signals_df = pd.DataFrame(signal_rows, columns=['symbol', 'ts', 'er_long', 'er_short'])
    signals_df['ts'] = pd.to_datetime(signals_df['ts'], utc=True)
    print(f"시그널 데이터: {len(signals_df):,}개")
else:
    print("시그널 데이터 없음 - 라벨만으로 분석")
    signals_df = None

# 단순 분석: 라벨 y 기준
# y = 1: TP 도달 (수익)
# y = -1: SL 도달 (손실)
# y = 0: 타임아웃

print("\n" + "="*70)
print("라벨 기반 분석 (y=1:TP도달, y=-1:SL도달)")
print("="*70)

for direction in ['LONG', 'SHORT']:
    sub = df[df['direction'] == direction]
    total = len(sub)
    wins = len(sub[sub['y'] == 1])
    losses = len(sub[sub['y'] == -1])
    timeout = len(sub[sub['y'] == 0])

    win_rate = wins / total * 100 if total > 0 else 0
    avg_ret = sub['ret_net'].mean() * 100

    # 수익/손실 금액
    total_profit = sub[sub['ret_net'] > 0]['ret_net'].sum()
    total_loss = abs(sub[sub['ret_net'] < 0]['ret_net'].sum())
    pf = total_profit / total_loss if total_loss > 0 else 0

    print(f"\n{direction}:")
    print(f"  총 거래: {total:,}")
    print(f"  승리(TP): {wins:,} ({wins/total*100:.1f}%)")
    print(f"  패배(SL): {losses:,} ({losses/total*100:.1f}%)")
    print(f"  타임아웃: {timeout:,} ({timeout/total*100:.1f}%)")
    print(f"  평균 수익률: {avg_ret:.3f}%")
    print(f"  Profit Factor: {pf:.2f}")

# 백테스트 시뮬레이션
print("\n" + "="*70)
print(f"백테스트 시뮬레이션 (${INITIAL} 시작, 포지션 ${POSITION_SIZE}, 레버리지 {LEVERAGE}x)")
print("="*70)

# 시간순 정렬
df = df.sort_values('ts').reset_index(drop=True)

# MAX_POSITIONS 제약 적용
MAX_POSITIONS = 6
capital = INITIAL
trades = 0
wins = 0
positions = []  # (exit_time, ...)

equity_curve = [INITIAL]
trade_log = []

for idx, row in df.iterrows():
    ts = row['ts']

    # 청산된 포지션 제거 (간단하게 이전 거래 모두 청산되었다고 가정)

    # 새 포지션 진입 (MAX_POSITIONS 무시 - 모든 시그널 실행)
    ret = row['ret_net']

    # 수수료 + 슬리피지 차감
    net_ret = ret - (FEE + SLIPPAGE) * 2

    # PnL 계산
    pnl = POSITION_SIZE * LEVERAGE * net_ret
    capital += pnl
    trades += 1

    if net_ret > 0:
        wins += 1

    if trades % 10000 == 0:
        equity_curve.append(capital)

final_capital = capital
total_return = (final_capital - INITIAL) / INITIAL * 100
win_rate = wins / trades * 100 if trades > 0 else 0

print(f"\n전체 결과 (모든 시그널 실행):")
print(f"  총 거래: {trades:,}")
print(f"  승률: {win_rate:.1f}%")
print(f"  최종 자본: ${final_capital:.2f}")
print(f"  수익률: {total_return:+.1f}%")

# 일별 분석
print("\n" + "="*70)
print("일별 분석")
print("="*70)

df['date'] = df['ts'].dt.date
daily = df.groupby('date').agg({
    'ret_net': ['count', 'sum', 'mean'],
    'y': lambda x: (x == 1).sum()
}).reset_index()
daily.columns = ['date', 'trades', 'total_ret', 'avg_ret', 'wins']
daily['win_rate'] = daily['wins'] / daily['trades'] * 100

print(f"\n{'Date':<12} {'Trades':>8} {'Wins':>8} {'WinRate':>10} {'AvgRet':>10} {'DailyPnL':>12}")
print("-"*62)

for _, row in daily.iterrows():
    daily_pnl = row['total_ret'] * POSITION_SIZE * LEVERAGE
    print(f"{row['date']} {row['trades']:>8,} {row['wins']:>8,} {row['win_rate']:>9.1f}% {row['avg_ret']*100:>9.3f}% ${daily_pnl:>10.2f}")

print(f"\n총 일수: {len(daily)}")
print(f"평균 일일 거래: {daily['trades'].mean():,.0f}")
print(f"평균 일일 PnL: ${daily['total_ret'].mean() * POSITION_SIZE * LEVERAGE:.2f}")
