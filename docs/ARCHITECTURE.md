# System Architecture

## Overview

Binance USDT-M Futures 자동매매 시스템. LightGBM 모델 기반 시그널 생성 → 실시간 Shadow/Live 거래.

## Directory Structure

```
ta/
├── apps/
│   ├── api/              # FastAPI 백엔드 (포트 7101)
│   └── web/              # Next.js 프론트엔드 (포트 7100)
├── services/
│   ├── realtime_worker_ws.py   # 메인 실시간 워커 (WebSocket 기반)
│   ├── realtime_worker.py      # 레거시 워커 (미사용)
│   ├── batch_worker.py         # 배치 작업 (학습, 라벨링)
│   ├── collector/              # 데이터 수집
│   ├── features/               # 피처 계산
│   ├── inference/              # 모델 추론
│   ├── labeling/               # Triple Barrier 라벨링
│   ├── training/               # 모델 학습
│   ├── policy/                 # 거래 결정 로직
│   ├── engine/                 # 포지션/세션 관리
│   ├── execution/              # 실제 주문 실행
│   ├── websocket/              # WebSocket 핸들러
│   ├── risk/                   # 리스크 관리
│   └── simulator/              # Shadow 거래 시뮬레이터
├── packages/
│   └── common/                 # 공통 유틸리티
├── scripts/                    # 유틸리티 스크립트
├── infra/
│   └── migrations/             # DB 마이그레이션
└── docker-compose.yml
```

## Services

### 1. worker_realtime (realtime_worker_ws.py)

**역할**: 실시간 데이터 수집 → 피처 계산 → 추론 → 거래 실행

**WebSocket 스트림**:
- `{symbol}@kline_1m` - 1분봉
- `{symbol}@kline_15m` - 15분봉
- `{symbol}@kline_1h` - 1시간봉
- `{symbol}@markPrice@1s` - Mark Price (SL/TP 체크용)
- `{symbol}@bookTicker` - 호가 (체결가 결정)

**주요 컴포넌트**:
- `KlineHandler`: 캔들 데이터 처리, 캔들 close 시 inference 트리거
- `MarkPriceHandler`: SL/TP 실시간 체크, 포지션 청산
- `BookTickerHandler`: 최적 bid/ask 가격 캐싱
- `PositionManager`: 포지션 생명주기 관리
- `Predictor`: LightGBM 모델 추론

### 2. worker_batch (batch_worker.py)

**역할**: 학습 작업 처리

**처리 작업**:
- 라벨링 (Triple Barrier)
- 모델 학습 (LightGBM)
- 모델 등록 및 평가

### 3. API (apps/api)

**포트**: 7101

**주요 엔드포인트**:
- `GET /api/status` - 시스템 상태
- `GET /api/signals/latest` - 최근 시그널
- `GET /api/trades` - 거래 내역
- `POST /api/trading/start` - 거래 시작
- `POST /api/trading/stop` - 거래 중지
- `POST /api/models/{id}/promote` - 모델 프로덕션 승격

### 4. Web (apps/web)

**포트**: 7100

Next.js 기반 대시보드.

## Infrastructure

### Docker Services

```yaml
postgres:     # PostgreSQL 16, 포트 5433
redis:        # Redis 7, 포트 6380
minio:        # MinIO (모델 저장소), 포트 9000/9001
api:          # FastAPI, 포트 7101
worker_realtime:  # 실시간 워커
worker_batch:     # 배치 워커
web:          # Next.js, 포트 7100
```

### Database Tables

**Core**:
- `instruments` - 거래 심볼
- `candles_1m`, `candles_15m`, `candles_1h` - OHLCV
- `premium_index` - Mark/Index/Funding
- `features_1m`, `features_15m`, `features_1h` - 계산된 피처
- `labels_long_1m`, `labels_short_1m` - Triple Barrier 라벨

**Trading**:
- `signals` - 시그널 기록
- `positions` - 포지션 이벤트 (entry/FINAL)
- `orders` - 주문
- `fills` - 체결

**Model**:
- `models` - 학습된 모델 메타데이터
- `training_jobs` - 학습 작업

## Model Architecture

### Feature Schema (v4)

**Multi-timeframe**: 1m + 15m + 1h = 132개 피처

각 타임프레임별 44개 피처:
- 가격 지표: ret_1, ret_5, ret_15, ret_60, log_ret
- 변동성: atr, atr_pct, volatility, range_pct
- 추세: adx, trend, ma_cross, ma_distance
- 모멘텀: rsi, rsi_z, stoch_k, stoch_d, macd, macd_signal, macd_hist
- 거래량: volume_ma_ratio, volume_std, vwap_distance
- 시장 구조: oi_change, funding_z, btc_regime, btc_corr_60

### LightGBM Model

**출력 (8개 타겟)**:
- `er_long`, `er_short` - Expected Return
- `q05_long`, `q05_short` - 5th Percentile Return
- `e_mae_long`, `e_mae_short` - Expected MAE
- `e_hold_long`, `e_hold_short` - Expected Hold Time

### Triple Barrier Labeling

- TP: 15m ATR × 1.5
- SL: 15m ATR × 1.0
- Max Hold: 360분 (6시간)
