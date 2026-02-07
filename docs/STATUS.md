# Current Status

## System State

| 항목 | 상태 |
|------|------|
| Mode | **Shadow** |
| Worker | `realtime_worker_ws.py` 실행 중 |
| WebSocket | 연결됨 (22 심볼 × 5 스트림 = 110 스트림) |
| Model | Production 배포됨 (Feature Schema v5) |
| UI Updates | **SSE 실시간** (polling 제거) |

## Production Model

| 항목 | 값 |
|------|-----|
| Model ID | `e0925468-bda8-4711-86d8-9b1866b7fc70` |
| Algorithm | LightGBM |
| Feature Schema | v5 (Multi-TF: 1m+15m+1h = 105 features) |
| Label Spec | `97ed8e99c6d9f2ea` |
| Training Period | 2024-01-01 ~ 2026-01-30 (25개월) |
| Trained Symbols | 21개 |

### Feature Schema v5 변경사항
- 제거: 13개 피처 (funding, OI, L/S ratio 관련)
- 추가: 4개 피처 (trend_direction, swing_structure, momentum_divergence, volume_confirmation)
- 최종: 105개 (35개 × 3 타임프레임)

### Training Parameters

- TP: 15m ATR × 1.5
- SL: 15m ATR × 1.0
- Max Hold: 360분 (6시간)
- Fee Rate: 0.04%
- Slippage: 0.15

## Current Settings

### Position Sizing (복리 시스템)

| 항목 | 값 | 계산 |
|------|-----|------|
| 초기 자본 | $300 | INITIAL_CAPITAL |
| 포지션 비율 | 10% | 현재 자본의 10% |
| 레버리지 | **20x** | LEVERAGE |
| 첫 거래 노셔널 | **$600** | $300 × 10% × 20 |
| 최대 포지션 | 6개 | MAX_POSITIONS |
| 방향별 최대 | **4개** | MAX_DIRECTIONAL_POSITIONS |

**복리 계산**: `포지션 = 현재 자본 × 10% × 레버리지(20)`

### Entry Filters

| 필터 | 값 | 설명 |
|------|-----|------|
| EV_MIN | 0.001 | er_long/short > 0.1% |
| Q05_MIN | -0.02 | q05 > -2% |
| MAE_MAX | 0.05 | e_mae < 5% |

### Market Condition Filters (신규)

| 필터 | 조건 | 효과 |
|------|------|------|
| Market Direction | btc_ret_60 > 1.5% | SHORT 차단 (EXTREME_PUMP) |
| | btc_ret_60 < -1.5% | LONG 차단 (EXTREME_DUMP) |
| Volatility | atr_percentile > 90% & EV < 0.2% | 차단 (HIGH_VOL_LOW_EV) |
| Consecutive Loss | 3연패 이상 & EV < 0.3% | 차단 (RECOVERY_MODE) |

## Active Universe (22 Symbols)

```
BTCUSDT   ETHUSDT   BNBUSDT   SOLUSDT   XRPUSDT   DOGEUSDT
ADAUSDT   AVAXUSDT  DOTUSDT   LINKUSDT  LTCUSDT   BCHUSDT
SUIUSDT   AAVEUSDT  1000PEPEUSDT  FILUSDT  AXSUSDT  ENAUSDT
ZKUSDT    ZECUSDT   TRUMPUSDT NEARUSDT
```

### Excluded Symbols (8개)

| Symbol | 제외 이유 |
|--------|----------|
| XAUUSDT | 금 - 다른 자산 클래스 |
| XAGUSDT | 은 - 다른 자산 클래스 |
| PAXGUSDT | 금 토큰 - 다른 자산 클래스 |
| HYPEUSDT | 데이터 부족 |
| ZORAUSDT | 신규 상장 - 데이터 부족 |
| PUMPUSDT | 신규 상장 - 데이터 부족 |
| ALPHAUSDT | 데이터 품질 이슈 |
| BNXUSDT | 데이터 품질 이슈 |

## Infrastructure

| Service | Status | Port |
|---------|--------|------|
| PostgreSQL | Running | 5433 |
| Redis | Running | 6380 |
| MinIO | Running | 9000 |
| API | Running | 7101 |
| Web | Running | 7100 |
| worker_realtime | Running | - |
| worker_batch | Running | - |

## Frontend Updates

### Real-time Components (SSE)
- **TradingSummaryLive**: 통계 실시간 업데이트
- **OpenPositionsLive**: 포지션 실시간 업데이트
- **TradeHistoryTableLive**: 거래 기록 실시간 업데이트

### TradingEventsContext
- SSE 연결 공유
- 자동 재연결 (3초 후)
- 이벤트 타입: position_opened, position_closed, stats_update, trade_completed

## Binance Connection

| 항목 | 상태 |
|------|------|
| API Keys | 설정됨 |
| Network | **Mainnet** (BINANCE_TESTNET=false) |
| WebSocket | 연결됨 |
