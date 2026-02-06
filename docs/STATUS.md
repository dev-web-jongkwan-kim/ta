# Current Status

## System State

| 항목 | 상태 |
|------|------|
| Mode | **Shadow** |
| Worker | `realtime_worker_ws.py` 실행 중 |
| WebSocket | 연결됨 (22 심볼 × 5 스트림 = 110 스트림) |
| Model | Production 배포됨 |

## Trading Statistics (2026-02-06 기준)

| 항목 | 값 |
|------|-----|
| 열린 포지션 | 6개 (MAX) |
| 완료된 거래 | 19개 |
| 총 PnL | **+$52.61** |
| 승률 | ~85% |

## Production Model

| 항목 | 값 |
|------|-----|
| Model ID | `c87b3140-ec29-444a-9b11-6823cd0a64c9` |
| Algorithm | LightGBM |
| Feature Schema | v4 (Multi-TF: 1m+15m+1h = 132 features) |
| Label Spec | `8c8d570e2343c185` |
| Training Period | 2025-02-02 ~ 2026-02-05 |
| Trained Symbols | 22개 |

### Training Parameters

- TP: 15m ATR × 1.5
- SL: 15m ATR × 1.0
- Max Hold: 360분 (6시간)
- Fee Rate: 0.04%
- Slippage: 0.15

## Current Settings

### Position Sizing

| 항목 | 값 | 계산 |
|------|-----|------|
| 총 자본 | $300 | INITIAL_CAPITAL |
| 포지션 마진 | $30 | POSITION_SIZE |
| 레버리지 | 10x | LEVERAGE |
| 명목가치 | $300/trade | $30 × 10 |
| 최대 포지션 | 6개 | MAX_POSITIONS |
| 최대 노출 | $1,800 | $300 × 6 |

### Entry Filters

| 필터 | 값 | 설명 |
|------|-----|------|
| EV_MIN | 0.002 | er_long/short > 0.2% |
| Q05_MIN | -0.02 | q05 > -2% |
| MAE_MAX | 0.05 | e_mae < 5% |

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
| PostgreSQL | ✅ Running | 5433 |
| Redis | ✅ Running | 6380 |
| MinIO | ✅ Running | 9000 |
| API | ✅ Running | 7101 |
| Web | ✅ Running | 7100 |
| worker_realtime | ✅ Running | - |
| worker_batch | ✅ Running | - |

## Binance Connection

| 항목 | 상태 |
|------|------|
| API Keys | ✅ 설정됨 |
| Network | **Mainnet** (BINANCE_TESTNET=false) |
| USDT Balance | $0 (미입금) |
| WebSocket | ✅ 연결됨 |

## Shadow Trading Performance (Today)

### Summary

- 거래 시간: ~4시간
- 총 거래: 19개
- 승리: 16개 (84%)
- 패배: 3개 (16%)
- 총 PnL: **+$52.61**
- Profit Factor: ~6.0

### 특이사항

1. **모든 승리가 LONG**: 상승장 편향
2. **SHORT 검증 부족**: 모델의 절반 미검증
3. **단기 샘플**: 통계적 신뢰도 낮음 (50개+ 필요)
