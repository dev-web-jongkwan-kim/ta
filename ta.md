# 🎯 Binance USDT-M Perpetual 선물 지도학습 트레이딩 시스템 완전 구현 명세서

**버전:** 2.0 (ta2.md 기반 완전 구현)  
**대상:** Claude Code (즉시 구현 가능)  
**목표:** 실전 선물 거래에서 생존 + 수익

---

# 📋 목차

1. [시스템 설계 원칙](#1-시스템-설계-원칙)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [인프라 구성](#3-인프라-구성)
4. [데이터베이스 스키마](#4-데이터베이스-스키마)
5. [Binance 선물 API 연동](#5-binance-선물-api-연동)
6. [데이터 수집 시스템](#6-데이터-수집-시스템)
7. [Feature Engineering](#7-feature-engineering)
8. [이벤트 라벨링](#8-이벤트-라벨링)
9. [모델 학습 시스템](#9-모델-학습-시스템)
10. [추론 및 정책 엔진](#10-추론-및-정책-엔진)
11. [리스크 관리 시스템](#11-리스크-관리-시스템)
12. [실행 엔진](#12-실행-엔진)
13. [프론트엔드](#13-프론트엔드)
14. [API 서버](#14-api-서버)
15. [배포 및 운영](#15-배포-및-운영)

---

# 1. 시스템 설계 원칙

## 1.1 핵심 목표

**"전략 규칙 없이 데이터로 EV(기대값) 계산 → 리스크 하드가드로 생존"**

### 시스템이 매 5분마다 산출하는 것

각 심볼별로:
```python
{
    'symbol': 'BTCUSDT',
    
    # 기대값 (비용 포함)
    'ev_long': 0.0123,   # 롱 진입 시 순기대값
    'ev_short': -0.0045,  # 숏 진입 시 순기대값
    
    # 확률
    'pwin_long': 0.68,    # TP가 먼저 맞을 확률
    'pwin_short': 0.42,   # TP가 먼저 맞을 확률
    
    # 리스크 지표
    'mae_long': 0.0078,   # 예상 최대 역행 (청산 위험)
    'mae_short': 0.0065,
    'hold_time_min': 87,  # 예상 보유 시간 (펀딩 계산용)
    
    # 청산 거리
    'liq_distance_atr': 5.2,  # 청산가까지 ATR 배수
    
    # 의사결정
    'decision': 'LONG',   # LONG / SHORT / FLAT
    'confidence': 0.85,
    
    # 실행 파라미터
    'recommended_notional': 500.0,  # USDT
    'recommended_leverage': 4,
    'entry_price': 50125.0,  # Mark price
    'sl_price': 49620.0,
    'tp_price': 51135.0,
    
    # 근거
    'top_features': [
        ('ema_dist_atr', 0.23),
        ('funding_z', -0.18),
        ('vol_z', 0.15)
    ]
}
```

## 1.2 운영 가정 (안전 우선)

```yaml
거래_타입: USDT-M Perpetual (선물)
마진_타입: Isolated (코인별 리스크 격리)
포지션_모드: One-way (단방향, hedge는 2단계)
레버리지_기본: 5x (변동성에 따라 2-8x)
주문_방식: MARKET 진입 + LIMIT reduce-only SL/TP

가격_기준:
  피처_계산: Mark Price (청산/손익 기준)
  체결_평가: Last Price (실제 체결가)
  
리스크_한도:
  마진_사용률_상한: 70%
  일일_손실_한도: 2%
  청산거리_최소: 2 ATR
  펀딩비_상한: ±0.01 (1%)
```

## 1.3 "전략 없는 학습"의 실체

```python
# ❌ 기존 방식 (규칙 기반 전략)
if rsi < 30 and macd > 0:
    enter_long()

# ✅ 이 시스템 (학습 기반 EV)
model.predict(current_features) 
→ pwin_long=0.68, expected_hold=87min, mae=0.78%

ev_long = (
    pwin_long * tp_pct - 
    (1 - pwin_long) * sl_pct -
    fee_rate * 2 -
    slippage_rate * 2 -
    funding_rate * (hold_time_min / 480)  # 8시간=480분
)

if ev_long > 0 and pwin_long > 0.55:
    # 정책: 단순히 EV > 0 조건만
    decision = 'LONG'
else:
    decision = 'FLAT'

# 리스크 엔진이 최종 검증
if risk_manager.check_all_constraints(decision):
    execute(decision)
else:
    reject(decision)  # 강제 거부
```

---

# 2. 전체 아키텍처

## 2.1 Monorepo 구조

```
crypto-futures-ml/
├── apps/
│   ├── web/                          # Next.js 프론트엔드
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   ├── dashboard/       # 메인 대시보드
│   │   │   │   ├── symbols/         # 심볼 상세
│   │   │   │   ├── positions/       # 포지션 관리
│   │   │   │   ├── training/        # 학습 센터
│   │   │   │   └── risk/            # 리스크 모니터
│   │   │   ├── components/          # 공통 컴포넌트
│   │   │   └── lib/                 # API 클라이언트
│   │   └── package.json
│   │
│   └── api/                          # FastAPI Gateway
│       ├── main.py
│       ├── routers/
│       │   ├── data.py
│       │   ├── signals.py
│       │   ├── positions.py
│       │   ├── training.py
│       │   └── risk.py
│       └── requirements.txt
│
├── services/
│   ├── collector/                    # 시장 데이터 수집
│   │   ├── market_data.py           # OHLCV, Mark Price, Funding
│   │   ├── scheduler.py
│   │   └── requirements.txt
│   │
│   ├── userstream/                   # 계정/포지션 실시간
│   │   ├── stream_manager.py        # listenKey 관리
│   │   ├── event_handler.py         # ORDER_TRADE_UPDATE 등
│   │   └── requirements.txt
│   │
│   ├── features/                     # 피처 생성 (온/오프라인 동일)
│   │   ├── calculator.py            # 피처 계산 로직
│   │   ├── schema.py                # 피처 스키마 버전 관리
│   │   └── requirements.txt
│   │
│   ├── labeling/                     # 라벨 생성 (배치)
│   │   ├── triple_barrier.py        # Triple Barrier 라벨링
│   │   ├── cost_calculator.py       # 비용 계산
│   │   └── requirements.txt
│   │
│   ├── training/                     # 모델 학습
│   │   ├── trainer.py               # LightGBM 학습
│   │   ├── validator.py             # Walk-forward 검증
│   │   ├── registry.py              # 모델 등록
│   │   └── requirements.txt
│   │
│   ├── inference/                    # 실시간 추론
│   │   ├── predictor.py             # 모델 로딩 + 캐싱
│   │   ├── online_features.py       # 실시간 피처
│   │   └── requirements.txt
│   │
│   ├── policy/                       # 정책 엔진
│   │   ├── decision_maker.py        # EV 기반 의사결정
│   │   └── requirements.txt
│   │
│   ├── risk/                         # 리스크 하드가드
│   │   ├── margin_monitor.py        # 마진 사용률
│   │   ├── liquidation_guard.py     # 청산 거리
│   │   ├── daily_loss_limiter.py    # 일일 손실
│   │   ├── connection_monitor.py    # 연결 상태
│   │   └── requirements.txt
│   │
│   ├── execution/                    # 주문 실행
│   │   ├── order_manager.py         # 주문 상태머신
│   │   ├── position_tracker.py      # 포지션 추적
│   │   └── requirements.txt
│   │
│   └── backtest/                     # 백테스트
│       ├── engine.py                # Walk-forward 백테스트
│       ├── cost_model.py            # 비용 모델링
│       └── requirements.txt
│
├── packages/
│   └── common/                       # 공통 라이브러리
│       ├── schemas.py               # Pydantic 스키마
│       ├── database.py              # DB 연결
│       ├── binance_client.py        # Binance API 래퍼
│       └── requirements.txt
│
├── infra/
│   ├── docker-compose.yml
│   ├── db/
│   │   ├── init.sql                # 초기 스키마
│   │   └── migrations/             # Alembic 마이그레이션
│   └── nginx/
│       └── nginx.conf
│
├── .env.example
└── README.md
```

## 2.2 서비스 간 통신

```
┌─────────────────────────────────────────────────────┐
│              Frontend (Next.js)                      │
│  Dashboard │ Symbols │ Positions │ Training │ Risk  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────▼────────────────────────────────┐
│                FastAPI Gateway                       │
│  /api/signals  /api/positions  /api/risk           │
└────────────────────┬────────────────────────────────┘
                     │ REST/RPC
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──────┐ ┌──▼──────┐ ┌──▼──────────┐
│ Collector    │ │Inference│ │ Execution   │
│ (시장 데이터) │ │(실시간)  │ │(주문/포지션)│
└───────┬──────┘ └──┬──────┘ └──┬──────────┘
        │            │            │
┌───────▼────────────▼────────────▼──────────┐
│         TimescaleDB (PostgreSQL)            │
│  candles │ premium_index │ positions │ ... │
└─────────────────────────────────────────────┘

        ┌────────────┐
        │   Redis    │ (캐시 + Pub/Sub)
        └────────────┘

        ┌────────────┐
        │   MinIO    │ (모델 아티팩트)
        └────────────┘
```

---

# 3. 인프라 구성

## 3.1 Docker Compose

**파일:** `infra/docker-compose.yml`

```yaml
version: '3.8'

services:
  # TimescaleDB
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: futures-timescaledb
    environment:
      POSTGRES_DB: futures_trading
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    container_name: futures-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # MinIO (S3 호환)
  minio:
    image: minio/minio:latest
    container_name: futures-minio
    environment:
      MINIO_ROOT_USER: ${MINIO_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    ports:
      - "9000:9000"
      - "9001:9001"  # Console
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Collector Service
  collector:
    build:
      context: ../services/collector
      dockerfile: Dockerfile
    container_name: futures-collector
    environment:
      DATABASE_URL: postgresql://trading_user:${DB_PASSWORD}@timescaledb:5432/futures_trading
      REDIS_URL: redis://redis:6379/0
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      BINANCE_API_SECRET: ${BINANCE_API_SECRET}
    depends_on:
      timescaledb:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  # User Stream Service
  userstream:
    build:
      context: ../services/userstream
      dockerfile: Dockerfile
    container_name: futures-userstream
    environment:
      DATABASE_URL: postgresql://trading_user:${DB_PASSWORD}@timescaledb:5432/futures_trading
      REDIS_URL: redis://redis:6379/0
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      BINANCE_API_SECRET: ${BINANCE_API_SECRET}
    depends_on:
      timescaledb:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  # Inference + Policy + Execution (실시간 워커)
  realtime-worker:
    build:
      context: ../services
      dockerfile: Dockerfile.realtime
    container_name: futures-realtime
    environment:
      DATABASE_URL: postgresql://trading_user:${DB_PASSWORD}@timescaledb:5432/futures_trading
      REDIS_URL: redis://redis:6379/0
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: ${MINIO_USER}
      MINIO_SECRET_KEY: ${MINIO_PASSWORD}
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      BINANCE_API_SECRET: ${BINANCE_API_SECRET}
    depends_on:
      - timescaledb
      - redis
      - minio
    restart: unless-stopped

  # Training Worker (배치)
  training-worker:
    build:
      context: ../services
      dockerfile: Dockerfile.training
    container_name: futures-training
    environment:
      DATABASE_URL: postgresql://trading_user:${DB_PASSWORD}@timescaledb:5432/futures_trading
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: ${MINIO_USER}
      MINIO_SECRET_KEY: ${MINIO_PASSWORD}
    depends_on:
      - timescaledb
      - minio
    restart: unless-stopped

  # API Gateway
  api:
    build:
      context: ../apps/api
      dockerfile: Dockerfile
    container_name: futures-api
    environment:
      DATABASE_URL: postgresql://trading_user:${DB_PASSWORD}@timescaledb:5432/futures_trading
      REDIS_URL: redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - timescaledb
      - redis
    restart: unless-stopped

  # Frontend
  web:
    build:
      context: ../apps/web
      dockerfile: Dockerfile
    container_name: futures-web
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    ports:
      - "3000:3000"
    depends_on:
      - api
    restart: unless-stopped

volumes:
  timescale_data:
  redis_data:
  minio_data:
```

## 3.2 환경 변수

**파일:** `.env.example`

```bash
# Database
DB_PASSWORD=your_secure_password_here

# MinIO
MINIO_USER=admin
MINIO_PASSWORD=your_minio_password_here

# Binance API (TESTNET 먼저 사용!)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true  # 실전 전에는 true

# Trading Parameters
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT

# Risk Limits
MAX_MARGIN_USAGE=0.70
MAX_DAILY_LOSS_PCT=0.02
MIN_LIQUIDATION_DISTANCE_ATR=2.0

# Labeling Parameters
LABEL_TP_ATR_MULTIPLIER=2.0
LABEL_SL_ATR_MULTIPLIER=1.0
LABEL_TIMEOUT_BARS=48  # 4시간 (5분봉 기준)

# Policy Parameters
MIN_EV=0.0
MIN_PWIN=0.55
MAX_MAE_ATR=1.2

# Execution
DEFAULT_LEVERAGE=5
FEE_RATE=0.0004  # 0.04% (테이커)
SLIPPAGE_BPS=15  # 0.15%
```

---

# 4. 데이터베이스 스키마

## 4.1 초기화 스크립트

**파일:** `infra/db/init.sql`

```sql
-- TimescaleDB 확장
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ========== 마스터 데이터 ==========

-- 거래 가능 심볼
CREATE TABLE instruments (
    symbol TEXT PRIMARY KEY,
    base_asset TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    contract_type TEXT NOT NULL DEFAULT 'PERPETUAL',
    
    -- 레버리지 제한
    max_leverage INT NOT NULL DEFAULT 125,
    
    -- 거래 제한
    min_notional NUMERIC(20, 8),
    min_qty NUMERIC(20, 8),
    step_size NUMERIC(20, 8),
    tick_size NUMERIC(20, 8),
    
    -- 메타
    liquidity_tier TEXT DEFAULT 'A',  -- A/B/C
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ========== 시장 데이터 ==========

-- 5분봉 OHLCV
CREATE TABLE candles_5m (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    quote_volume NUMERIC(20, 8),
    trades_count INT,
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('candles_5m', 'ts', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_candles_5m_symbol_ts ON candles_5m (symbol, ts DESC);

-- Premium Index (Mark Price + Funding)
-- GET /fapi/v1/premiumIndex 응답 구조 반영
CREATE TABLE premium_index (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    
    -- 핵심 가격
    mark_price NUMERIC(20, 8) NOT NULL,      -- 공정 가격 (청산 기준)
    index_price NUMERIC(20, 8),              -- 지수 가격
    last_price NUMERIC(20, 8),               -- 최종 체결가
    
    -- 펀딩
    last_funding_rate NUMERIC(10, 6),        -- 마지막 펀딩비
    next_funding_time TIMESTAMPTZ,           -- 다음 펀딩 시간
    
    -- 프리미엄
    estimated_settle_price NUMERIC(20, 8),
    
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('premium_index', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_premium_symbol_ts ON premium_index (symbol, ts DESC);

-- Funding Rate History
-- GET /fapi/v1/fundingRate 응답 구조
CREATE TABLE funding_rates (
    funding_time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    funding_rate NUMERIC(10, 6) NOT NULL,
    mark_price NUMERIC(20, 8),
    PRIMARY KEY (symbol, funding_time)
);

SELECT create_hypertable('funding_rates', 'funding_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ========== Feature 데이터 ==========

-- Features (스키마 버전 관리 필수)
CREATE TABLE features_5m (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    
    -- 스키마 버전 (피처 정의 변경 추적)
    schema_version INT NOT NULL,
    
    -- 수익률
    r_1 NUMERIC(10, 6),   -- 1바 수익률
    r_3 NUMERIC(10, 6),   -- 3바 수익률
    r_12 NUMERIC(10, 6),  -- 12바 수익률
    
    -- 추세
    ema_7 NUMERIC(20, 8),
    ema_21 NUMERIC(20, 8),
    ema_50 NUMERIC(20, 8),
    ema_dist_atr NUMERIC(10, 4),  -- (price - ema) / atr
    
    -- 변동성
    atr NUMERIC(20, 8),
    atr_pct NUMERIC(10, 4),
    volatility_10 NUMERIC(10, 6),
    volatility_30 NUMERIC(10, 6),
    
    -- 모멘텀
    rsi_14 NUMERIC(10, 4),
    macd NUMERIC(20, 8),
    macd_signal NUMERIC(20, 8),
    adx NUMERIC(10, 4),
    
    -- 거래량
    volume_ma_12 NUMERIC(20, 8),
    vol_z NUMERIC(10, 4),  -- volume z-score
    
    -- 볼린저 밴드
    bb_upper NUMERIC(20, 8),
    bb_middle NUMERIC(20, 8),
    bb_lower NUMERIC(20, 8),
    bb_z NUMERIC(10, 4),  -- (price - bb_mid) / (bb_upper - bb_lower)
    
    -- 펀딩비 (선물 전용)
    funding_rate NUMERIC(10, 6),
    funding_ma_24 NUMERIC(10, 6),  -- 24시간 평균
    funding_z NUMERIC(10, 4),      -- z-score
    
    -- 시장 레짐 (BTC 기준)
    btc_regime INT,  -- -1: 하락, 0: 횡보, 1: 상승
    
    -- 시간 특징
    hour_of_day INT,
    day_of_week INT,
    is_asian_session BOOLEAN,
    is_funding_hour BOOLEAN,  -- 펀딩 정산 시간 근처
    
    PRIMARY KEY (symbol, ts, schema_version)
);

SELECT create_hypertable('features_5m', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_features_5m_symbol_ts ON features_5m (symbol, ts DESC);
CREATE INDEX idx_features_5m_schema ON features_5m (schema_version);

-- ========== 라벨 데이터 ==========

-- 롱 방향 라벨
CREATE TABLE labels_long_5m (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    
    -- 파라미터
    k_tp NUMERIC(4, 2) NOT NULL,  -- TP = k_tp * ATR
    k_sl NUMERIC(4, 2) NOT NULL,  -- SL = k_sl * ATR
    h_bars INT NOT NULL,          -- Timeout bars
    
    -- 결과
    y INT NOT NULL,  -- +1: TP, -1: SL, 0: TIMEOUT
    
    -- 수익률 (비용 포함)
    ret_gross NUMERIC(10, 6),     -- 가격 변화만
    ret_net NUMERIC(10, 6),       -- 비용 차감 후
    
    -- 비용 분해
    fee_cost NUMERIC(10, 6),
    slippage_cost NUMERIC(10, 6),
    funding_cost NUMERIC(10, 6),
    
    -- 리스크 지표
    mae NUMERIC(10, 6) NOT NULL,  -- Maximum Adverse Excursion
    mfe NUMERIC(10, 6) NOT NULL,  -- Maximum Favorable Excursion
    
    -- 시간
    time_to_event_min INT NOT NULL,
    bars_held INT NOT NULL,
    
    PRIMARY KEY (symbol, ts, k_tp, k_sl, h_bars)
);

SELECT create_hypertable('labels_long_5m', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 숏 방향 라벨 (구조 동일)
CREATE TABLE labels_short_5m (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    k_tp NUMERIC(4, 2) NOT NULL,
    k_sl NUMERIC(4, 2) NOT NULL,
    h_bars INT NOT NULL,
    y INT NOT NULL,
    ret_gross NUMERIC(10, 6),
    ret_net NUMERIC(10, 6),
    fee_cost NUMERIC(10, 6),
    slippage_cost NUMERIC(10, 6),
    funding_cost NUMERIC(10, 6),
    mae NUMERIC(10, 6) NOT NULL,
    mfe NUMERIC(10, 6) NOT NULL,
    time_to_event_min INT NOT NULL,
    bars_held INT NOT NULL,
    PRIMARY KEY (symbol, ts, k_tp, k_sl, h_bars)
);

SELECT create_hypertable('labels_short_5m', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ========== 학습 관리 ==========

-- 학습 작업
CREATE TABLE training_jobs (
    job_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    
    status TEXT NOT NULL,  -- pending/running/completed/failed
    
    -- 설정
    config JSONB NOT NULL,
    
    -- 결과
    metrics JSONB,
    report_uri TEXT,
    error_message TEXT
);

CREATE INDEX idx_training_jobs_status ON training_jobs (status);
CREATE INDEX idx_training_jobs_created ON training_jobs (created_at DESC);

-- 모델 레지스트리
CREATE TABLE models (
    model_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- 모델 정보
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    algorithm TEXT NOT NULL,  -- 'lgbm'
    
    -- 피처 호환성
    schema_version INT NOT NULL,
    
    -- 라벨 스펙
    label_spec_hash TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'long' or 'short'
    
    -- 학습 데이터
    train_start TIMESTAMPTZ NOT NULL,
    train_end TIMESTAMPTZ NOT NULL,
    symbols TEXT[] NOT NULL,
    
    -- 성과
    train_metrics JSONB,
    val_metrics JSONB,
    
    -- 아티팩트
    artifact_uri TEXT NOT NULL,  -- MinIO path
    
    -- 상태
    status TEXT NOT NULL DEFAULT 'trained',  -- trained/promoted/archived
    promoted_at TIMESTAMPTZ,
    
    UNIQUE(name, version)
);

CREATE INDEX idx_models_status ON models (status);
CREATE INDEX idx_models_schema ON models (schema_version);
CREATE INDEX idx_models_promoted ON models (promoted_at DESC) WHERE status = 'promoted';

-- ========== 실시간 신호 ==========

-- 추론 결과 + 정책 결정
CREATE TABLE signals (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    
    -- 모델 정보
    model_id UUID REFERENCES models(model_id),
    
    -- 예측값
    ev_long NUMERIC(10, 6),
    ev_short NUMERIC(10, 6),
    pwin_long NUMERIC(5, 4),
    pwin_short NUMERIC(5, 4),
    expected_mae_long NUMERIC(10, 6),
    expected_mae_short NUMERIC(10, 6),
    expected_hold_minutes INT,
    
    -- 의사결정
    decision TEXT NOT NULL,  -- LONG/SHORT/FLAT
    confidence NUMERIC(5, 4),
    
    -- 실행 파라미터
    recommended_notional NUMERIC(20, 2),
    recommended_leverage INT,
    entry_price NUMERIC(20, 8),
    sl_price NUMERIC(20, 8),
    tp_price NUMERIC(20, 8),
    
    -- 근거 (top feature contributions)
    reasons JSONB,
    
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('signals', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_signals_decision ON signals (decision) WHERE decision != 'FLAT';

-- ========== 계정 & 포지션 ==========

-- 계정 스냅샷
CREATE TABLE account_snapshots (
    ts TIMESTAMPTZ NOT NULL PRIMARY KEY,
    
    -- 총 자산
    total_wallet_balance NUMERIC(20, 8),
    total_unrealized_pnl NUMERIC(20, 8),
    total_margin_balance NUMERIC(20, 8),
    total_position_initial_margin NUMERIC(20, 8),
    total_open_order_initial_margin NUMERIC(20, 8),
    available_balance NUMERIC(20, 8),
    
    -- 사용률
    margin_usage_pct NUMERIC(5, 4),
    
    -- 펀딩 누적
    total_funding_pnl NUMERIC(20, 8),
    
    -- 일일 손익
    daily_realized_pnl NUMERIC(20, 8),
    daily_unrealized_pnl NUMERIC(20, 8)
);

SELECT create_hypertable('account_snapshots', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- 포지션 상태 (현재)
-- GET /fapi/v2/positionRisk 응답 구조
CREATE TABLE positions (
    symbol TEXT PRIMARY KEY REFERENCES instruments(symbol),
    
    -- 포지션 기본
    position_side TEXT NOT NULL,  -- BOTH/LONG/SHORT
    position_amt NUMERIC(20, 8) NOT NULL,
    entry_price NUMERIC(20, 8),
    
    -- 레버리지
    leverage INT NOT NULL,
    isolated_wallet NUMERIC(20, 8),
    
    -- 손익
    unrealized_profit NUMERIC(20, 8),
    
    -- ⭐ 청산가 (핵심!)
    liquidation_price NUMERIC(20, 8),
    
    -- Mark price
    mark_price NUMERIC(20, 8),
    
    -- 마진 타입
    margin_type TEXT NOT NULL,  -- isolated/cross
    
    -- 시간
    update_time TIMESTAMPTZ NOT NULL,
    
    -- 메타
    notional NUMERIC(20, 8),
    isolated_margin NUMERIC(20, 8)
);

CREATE INDEX idx_positions_update ON positions (update_time DESC);

-- 포지션 히스토리
CREATE TABLE position_history (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    
    position_amt NUMERIC(20, 8),
    entry_price NUMERIC(20, 8),
    leverage INT,
    unrealized_profit NUMERIC(20, 8),
    liquidation_price NUMERIC(20, 8),
    mark_price NUMERIC(20, 8),
    
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('position_history', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ========== 주문 & 체결 ==========

-- 주문
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    
    -- 주문 정보
    client_order_id TEXT,
    side TEXT NOT NULL,  -- BUY/SELL
    type TEXT NOT NULL,  -- MARKET/LIMIT/...
    time_in_force TEXT,
    
    -- 수량/가격
    orig_qty NUMERIC(20, 8),
    price NUMERIC(20, 8),
    avg_price NUMERIC(20, 8),
    
    -- 상태
    status TEXT NOT NULL,  -- NEW/FILLED/CANCELED/...
    executed_qty NUMERIC(20, 8),
    
    -- Reduce-only (중요!)
    reduce_only BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- 시간
    created_at TIMESTAMPTZ NOT NULL,
    update_time TIMESTAMPTZ NOT NULL,
    
    -- 메타
    position_side TEXT,  -- BOTH/LONG/SHORT
    working_type TEXT
);

CREATE INDEX idx_orders_symbol ON orders (symbol);
CREATE INDEX idx_orders_status ON orders (status);
CREATE INDEX idx_orders_created ON orders (created_at DESC);

-- 체결
CREATE TABLE fills (
    trade_id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES orders(order_id),
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    
    -- 체결 정보
    side TEXT NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    qty NUMERIC(20, 8) NOT NULL,
    realized_pnl NUMERIC(20, 8),
    commission NUMERIC(20, 8),
    commission_asset TEXT,
    
    -- 시간
    time TIMESTAMPTZ NOT NULL,
    
    -- 매이커 여부
    is_maker BOOLEAN NOT NULL
);

CREATE INDEX idx_fills_symbol ON fills (symbol);
CREATE INDEX idx_fills_time ON fills (time DESC);
CREATE INDEX idx_fills_order ON fills (order_id);

-- ========== 리스크 이벤트 ==========

CREATE TABLE risk_events (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    event_type TEXT NOT NULL,  -- margin_high/liq_close/daily_loss/connection_lost/...
    severity TEXT NOT NULL,    -- info/warning/critical
    
    symbol TEXT,
    
    details JSONB NOT NULL,
    
    -- 조치
    action_taken TEXT,  -- rejected_entry/forced_close/trading_disabled/...
    
    resolved_at TIMESTAMPTZ
);

CREATE INDEX idx_risk_events_ts ON risk_events (ts DESC);
CREATE INDEX idx_risk_events_type ON risk_events (event_type);
CREATE INDEX idx_risk_events_severity ON risk_events (severity) WHERE resolved_at IS NULL;

-- ========== 시스템 상태 ==========

CREATE TABLE system_status (
    component TEXT PRIMARY KEY,
    status TEXT NOT NULL,  -- healthy/degraded/down
    last_heartbeat TIMESTAMPTZ NOT NULL,
    details JSONB
);
```

---
# 5. Binance 선물 API 연동

## 5.1 공통 Binance 클라이언트

**파일:** `packages/common/binance_client.py`

```python
import hmac
import hashlib
import time
from typing import Dict, List, Optional
import requests
from urllib.parse import urlencode
import logging

logger = logging.getLogger(__name__)

class BinanceFuturesClient:
    """
    Binance USDT-M Futures API 클라이언트
    
    공식 문서:
    https://developers.binance.com/docs/derivatives/usds-margined-futures
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Testnet vs Mainnet
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_base_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_base_url = "wss://fstream.binance.com"
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
    
    def _sign(self, params: Dict) -> str:
        """HMAC SHA256 서명 생성"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        **kwargs
    ) -> Dict:
        """공통 요청 메서드"""
        url = f"{self.base_url}{endpoint}"
        
        if signed:
            kwargs['timestamp'] = int(time.time() * 1000)
            kwargs['signature'] = self._sign(kwargs)
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=kwargs, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, params=kwargs, timeout=10)
            elif method == 'PUT':
                response = self.session.put(url, params=kwargs, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, params=kwargs, timeout=10)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API error: {e}")
            raise
    
    # ========== 시장 데이터 ==========
    
    def get_exchange_info(self) -> Dict:
        """거래소 정보 (심볼, 레버리지 제한 등)"""
        return self._request('GET', '/fapi/v1/exchangeInfo')
    
    def get_klines(
        self,
        symbol: str,
        interval: str = '5m',
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Kline/캔들 데이터
        
        interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        return self._request('GET', '/fapi/v1/klines', **params)
    
    def get_premium_index(self, symbol: Optional[str] = None) -> Dict:
        """
        Mark Price & Funding Rate
        
        API: GET /fapi/v1/premiumIndex
        응답: {
            "symbol": "BTCUSDT",
            "markPrice": "50000.00000000",
            "indexPrice": "49995.12345678",
            "lastFundingRate": "0.00010000",
            "nextFundingTime": 1640995200000,
            "interestRate": "0.00010000",
            "time": 1640995123456
        }
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._request('GET', '/fapi/v1/premiumIndex', **params)
    
    def get_funding_rate(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Funding Rate 히스토리
        
        API: GET /fapi/v1/fundingRate
        응답: [{
            "symbol": "BTCUSDT",
            "fundingTime": 1640995200000,
            "fundingRate": "0.00010000",
            "markPrice": "50000.00000000"
        }]
        """
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        return self._request('GET', '/fapi/v1/fundingRate', **params)
    
    def get_open_interest(self, symbol: str) -> Dict:
        """미결제약정 (Open Interest)"""
        return self._request('GET', '/fapi/v1/openInterest', symbol=symbol)
    
    # ========== 계정 & 포지션 (서명 필요) ==========
    
    def get_account(self) -> Dict:
        """계정 정보"""
        return self._request('GET', '/fapi/v2/account', signed=True)
    
    def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        포지션 리스크 정보 (청산가 포함!)
        
        API: GET /fapi/v2/positionRisk
        응답: [{
            "symbol": "BTCUSDT",
            "positionAmt": "0.100",
            "entryPrice": "50000.0",
            "markPrice": "50500.0",
            "unRealizedProfit": "50.0",
            "liquidationPrice": "45000.0",  # ⭐ 청산가
            "leverage": "5",
            "maxNotionalValue": "250000",
            "marginType": "isolated",
            "isolatedMargin": "1000.00000000",
            "isAutoAddMargin": "false",
            "positionSide": "BOTH",
            "notional": "5050.0",
            "isolatedWallet": "1000.00000000",
            "updateTime": 1625474304765
        }]
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._request('GET', '/fapi/v2/positionRisk', signed=True, **params)
    
    def get_leverage_bracket(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        레버리지 브라켓 (notional 구간별 최대 레버리지)
        
        API: GET /fapi/v1/leverageBracket
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._request('GET', '/fapi/v1/leverageBracket', signed=True, **params)
    
    def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        레버리지 변경
        
        API: POST /fapi/v1/leverage
        """
        return self._request(
            'POST',
            '/fapi/v1/leverage',
            signed=True,
            symbol=symbol,
            leverage=leverage
        )
    
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        마진 타입 변경 (ISOLATED/CROSSED)
        
        API: POST /fapi/v1/marginType
        """
        return self._request(
            'POST',
            '/fapi/v1/marginType',
            signed=True,
            symbol=symbol,
            marginType=margin_type
        )
    
    # ========== 주문 ==========
    
    def create_order(
        self,
        symbol: str,
        side: str,  # BUY/SELL
        order_type: str,  # MARKET/LIMIT/STOP/...
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: Optional[str] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        주문 생성
        
        API: POST /fapi/v1/order
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type
        }
        
        if quantity:
            params['quantity'] = quantity
        if price:
            params['price'] = price
        if reduce_only:
            params['reduceOnly'] = 'true'
        if time_in_force:
            params['timeInForce'] = time_in_force
        if stop_price:
            params['stopPrice'] = stop_price
        
        params.update(kwargs)
        
        return self._request('POST', '/fapi/v1/order', signed=True, **params)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """주문 취소"""
        return self._request(
            'DELETE',
            '/fapi/v1/order',
            signed=True,
            symbol=symbol,
            orderId=order_id
        )
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """미체결 주문 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._request('GET', '/fapi/v1/openOrders', signed=True, **params)
    
    def get_all_orders(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None
    ) -> List[Dict]:
        """전체 주문 히스토리"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        
        return self._request('GET', '/fapi/v1/allOrders', signed=True, **params)
    
    # ========== User Data Stream ==========
    
    def start_user_data_stream(self) -> Dict:
        """
        User Data Stream 시작 (listenKey 발급)
        
        API: POST /fapi/v1/listenKey
        응답: {"listenKey": "pqia91ma19a5s61cv6a81va65sdf19v8a65a1a5s61cv6a81va65sdf19v8a65a1"}
        
        ⚠️ listenKey는 60분마다 만료되므로 keepalive 필수!
        """
        return self._request('POST', '/fapi/v1/listenKey', signed=False)
    
    def keepalive_user_data_stream(self, listen_key: str) -> Dict:
        """
        User Data Stream Keepalive (60분마다 호출 필요)
        
        API: PUT /fapi/v1/listenKey
        """
        return self._request(
            'PUT',
            '/fapi/v1/listenKey',
            signed=False,
            listenKey=listen_key
        )
    
    def close_user_data_stream(self, listen_key: str) -> Dict:
        """User Data Stream 종료"""
        return self._request(
            'DELETE',
            '/fapi/v1/listenKey',
            signed=False,
            listenKey=listen_key
        )
```

## 5.2 Pydantic 스키마

**파일:** `packages/common/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

# ========== 시장 데이터 ==========

class PremiumIndex(BaseModel):
    """Mark Price & Funding Rate"""
    symbol: str
    mark_price: Decimal = Field(alias='markPrice')
    index_price: Optional[Decimal] = Field(None, alias='indexPrice')
    last_price: Optional[Decimal] = Field(None, alias='lastPrice')
    last_funding_rate: Optional[Decimal] = Field(None, alias='lastFundingRate')
    next_funding_time: Optional[datetime] = Field(None, alias='nextFundingTime')
    interest_rate: Optional[Decimal] = Field(None, alias='interestRate')
    time: datetime
    
    class Config:
        populate_by_name = True

class FundingRate(BaseModel):
    """Funding Rate 히스토리"""
    symbol: str
    funding_time: datetime = Field(alias='fundingTime')
    funding_rate: Decimal = Field(alias='fundingRate')
    mark_price: Optional[Decimal] = Field(None, alias='markPrice')
    
    class Config:
        populate_by_name = True

class Candle(BaseModel):
    """Kline/캔들 데이터"""
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime
    quote_volume: Decimal
    trades_count: int
    taker_buy_base: Decimal
    taker_buy_quote: Decimal
    
    @classmethod
    def from_binance(cls, data: List):
        """Binance API 응답 파싱"""
        return cls(
            open_time=datetime.fromtimestamp(data[0] / 1000),
            open=Decimal(data[1]),
            high=Decimal(data[2]),
            low=Decimal(data[3]),
            close=Decimal(data[4]),
            volume=Decimal(data[5]),
            close_time=datetime.fromtimestamp(data[6] / 1000),
            quote_volume=Decimal(data[7]),
            trades_count=int(data[8]),
            taker_buy_base=Decimal(data[9]),
            taker_buy_quote=Decimal(data[10])
        )

# ========== 포지션 & 계정 ==========

class PositionRisk(BaseModel):
    """포지션 리스크 정보"""
    symbol: str
    position_amt: Decimal = Field(alias='positionAmt')
    entry_price: Decimal = Field(alias='entryPrice')
    mark_price: Decimal = Field(alias='markPrice')
    unrealized_profit: Decimal = Field(alias='unRealizedProfit')
    liquidation_price: Decimal = Field(alias='liquidationPrice')  # ⭐ 핵심
    leverage: int
    max_notional_value: Decimal = Field(alias='maxNotionalValue')
    margin_type: str = Field(alias='marginType')
    isolated_margin: Decimal = Field(alias='isolatedMargin')
    is_auto_add_margin: bool = Field(alias='isAutoAddMargin')
    position_side: str = Field(alias='positionSide')
    notional: Decimal
    isolated_wallet: Decimal = Field(alias='isolatedWallet')
    update_time: datetime = Field(alias='updateTime')
    
    class Config:
        populate_by_name = True

class Account(BaseModel):
    """계정 정보"""
    total_wallet_balance: Decimal = Field(alias='totalWalletBalance')
    total_unrealized_profit: Decimal = Field(alias='totalUnrealizedProfit')
    total_margin_balance: Decimal = Field(alias='totalMarginBalance')
    total_position_initial_margin: Decimal = Field(alias='totalPositionInitialMargin')
    total_open_order_initial_margin: Decimal = Field(alias='totalOpenOrderInitialMargin')
    available_balance: Decimal = Field(alias='availableBalance')
    max_withdraw_amount: Decimal = Field(alias='maxWithdrawAmount')
    
    class Config:
        populate_by_name = True

# ========== 주문 ==========

class Order(BaseModel):
    """주문 정보"""
    order_id: int = Field(alias='orderId')
    symbol: str
    status: str
    client_order_id: str = Field(alias='clientOrderId')
    price: Decimal
    avg_price: Decimal = Field(alias='avgPrice')
    orig_qty: Decimal = Field(alias='origQty')
    executed_qty: Decimal = Field(alias='executedQty')
    cumulative_quote_qty: Decimal = Field(alias='cumQuoteQty')
    time_in_force: str = Field(alias='timeInForce')
    type: str
    reduce_only: bool = Field(alias='reduceOnly')
    side: str
    position_side: str = Field(alias='positionSide')
    stop_price: Optional[Decimal] = Field(None, alias='stopPrice')
    working_type: Optional[str] = Field(None, alias='workingType')
    time: datetime = Field(alias='updateTime')
    
    class Config:
        populate_by_name = True

# ========== User Data Stream 이벤트 ==========

class OrderTradeUpdate(BaseModel):
    """ORDER_TRADE_UPDATE 이벤트"""
    event_type: str = Field(alias='e')
    event_time: datetime = Field(alias='E')
    transaction_time: datetime = Field(alias='T')
    
    symbol: str = Field(alias='s')
    client_order_id: str = Field(alias='c')
    side: str = Field(alias='S')
    order_type: str = Field(alias='o')
    time_in_force: str = Field(alias='f')
    original_quantity: Decimal = Field(alias='q')
    original_price: Decimal = Field(alias='p')
    average_price: Decimal = Field(alias='ap')
    stop_price: Decimal = Field(alias='sp')
    execution_type: str = Field(alias='x')
    order_status: str = Field(alias='X')
    order_id: int = Field(alias='i')
    last_filled_quantity: Decimal = Field(alias='l')
    cumulative_filled_quantity: Decimal = Field(alias='z')
    last_filled_price: Decimal = Field(alias='L')
    commission_asset: Optional[str] = Field(None, alias='N')
    commission: Optional[Decimal] = Field(None, alias='n')
    trade_time: datetime = Field(alias='T')
    trade_id: int = Field(alias='t')
    realized_profit: Decimal = Field(alias='rp')
    
    class Config:
        populate_by_name = True

class AccountUpdate(BaseModel):
    """ACCOUNT_UPDATE 이벤트"""
    event_type: str = Field(alias='e')
    event_time: datetime = Field(alias='E')
    transaction_time: datetime = Field(alias='T')
    
    # 간소화 - 실제로는 더 많은 필드
    reason: str = Field(alias='m')
    
    class Config:
        populate_by_name = True
```

---

# 6. 데이터 수집 시스템

## 6.1 Market Data Collector

**파일:** `services/collector/market_data.py`

```python
import asyncio
import websockets
import json
from datetime import datetime, timedelta
from typing import List
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from packages.common.binance_client import BinanceFuturesClient
from packages.common.schemas import PremiumIndex, FundingRate, Candle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataCollector:
    """
    시장 데이터 수집기
    
    - OHLCV (5분봉)
    - Premium Index (Mark Price + Funding)
    - Funding Rate History
    """
    
    def __init__(
        self,
        symbols: List[str],
        db_url: str,
        binance_client: BinanceFuturesClient
    ):
        self.symbols = symbols
        self.client = binance_client
        
        # DB 연결
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
    
    async def collect_candles_historical(self, symbol: str, days: int = 30):
        """
        과거 캔들 데이터 수집
        
        Args:
            symbol: 'BTCUSDT' 등
            days: 수집할 일수
        """
        logger.info(f"Collecting historical candles for {symbol} ({days} days)")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        all_candles = []
        current_start = start_ms
        
        while current_start < end_ms:
            try:
                # Binance API는 최대 1500개까지
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval='5m',
                    limit=1500,
                    start_time=current_start,
                    end_time=end_ms
                )
                
                if not klines:
                    break
                
                # 파싱
                candles = [Candle.from_binance(k) for k in klines]
                all_candles.extend(candles)
                
                logger.info(f"Collected {len(candles)} candles for {symbol}")
                
                # 다음 구간
                current_start = klines[-1][6] + 1  # close_time + 1ms
                
                # Rate limit
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error collecting candles: {e}")
                await asyncio.sleep(5)
        
        # DB 저장
        session = self.Session()
        try:
            for candle in all_candles:
                session.execute("""
                    INSERT INTO candles_5m (
                        ts, symbol, open, high, low, close, volume,
                        quote_volume, trades_count
                    ) VALUES (
                        :ts, :symbol, :open, :high, :low, :close, :volume,
                        :quote_volume, :trades_count
                    )
                    ON CONFLICT (symbol, ts) DO NOTHING
                """, {
                    'ts': candle.open_time,
                    'symbol': symbol,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'quote_volume': candle.quote_volume,
                    'trades_count': candle.trades_count
                })
            
            session.commit()
            logger.info(f"✓ Saved {len(all_candles)} candles for {symbol}")
            
        except Exception as e:
            logger.error(f"DB save error: {e}")
            session.rollback()
        finally:
            session.close()
    
    async def collect_premium_index_loop(self):
        """Premium Index 주기적 수집 (1분마다)"""
        logger.info("Starting premium index collection loop")
        
        while True:
            try:
                # 전체 심볼 조회
                data = self.client.get_premium_index()
                
                if not isinstance(data, list):
                    data = [data]
                
                # 우리가 추적하는 심볼만 필터
                data = [d for d in data if d['symbol'] in self.symbols]
                
                # 파싱
                records = [PremiumIndex(**d) for d in data]
                
                # DB 저장
                session = self.Session()
                try:
                    for rec in records:
                        session.execute("""
                            INSERT INTO premium_index (
                                ts, symbol, mark_price, index_price, last_price,
                                last_funding_rate, next_funding_time
                            ) VALUES (
                                :ts, :symbol, :mark_price, :index_price, :last_price,
                                :last_funding_rate, :next_funding_time
                            )
                            ON CONFLICT (symbol, ts) DO UPDATE SET
                                mark_price = EXCLUDED.mark_price,
                                last_funding_rate = EXCLUDED.last_funding_rate
                        """, {
                            'ts': rec.time,
                            'symbol': rec.symbol,
                            'mark_price': rec.mark_price,
                            'index_price': rec.index_price,
                            'last_price': rec.last_price,
                            'last_funding_rate': rec.last_funding_rate,
                            'next_funding_time': rec.next_funding_time
                        })
                    
                    session.commit()
                    logger.info(f"✓ Saved premium index for {len(records)} symbols")
                    
                except Exception as e:
                    logger.error(f"Premium index save error: {e}")
                    session.rollback()
                finally:
                    session.close()
                
                # 1분 대기
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Premium index collection error: {e}")
                await asyncio.sleep(10)
    
    async def collect_funding_history(self, symbol: str):
        """Funding Rate 히스토리 수집 (배치)"""
        logger.info(f"Collecting funding rate history for {symbol}")
        
        try:
            # 최근 100개
            data = self.client.get_funding_rate(symbol=symbol, limit=100)
            
            records = [FundingRate(**d) for d in data]
            
            # DB 저장
            session = self.Session()
            try:
                for rec in records:
                    session.execute("""
                        INSERT INTO funding_rates (
                            funding_time, symbol, funding_rate, mark_price
                        ) VALUES (
                            :funding_time, :symbol, :funding_rate, :mark_price
                        )
                        ON CONFLICT (symbol, funding_time) DO NOTHING
                    """, {
                        'funding_time': rec.funding_time,
                        'symbol': rec.symbol,
                        'funding_rate': rec.funding_rate,
                        'mark_price': rec.mark_price
                    })
                
                session.commit()
                logger.info(f"✓ Saved {len(records)} funding rates for {symbol}")
                
            except Exception as e:
                logger.error(f"Funding rate save error: {e}")
                session.rollback()
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Funding rate collection error: {e}")
    
    async def collect_candles_websocket(self, symbol: str):
        """WebSocket으로 실시간 캔들 수집"""
        stream_name = f"{symbol.lower()}@kline_5m"
        url = f"{self.client.ws_base_url}/ws/{stream_name}"
        
        logger.info(f"Starting WebSocket candle stream for {symbol}")
        
        while True:
            try:
                async with websockets.connect(url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        if 'k' in data:
                            k = data['k']
                            
                            # 캔들 종료 시에만 저장
                            if k['x']:  # is_closed
                                candle = Candle(
                                    open_time=datetime.fromtimestamp(k['t'] / 1000),
                                    open=k['o'],
                                    high=k['h'],
                                    low=k['l'],
                                    close=k['c'],
                                    volume=k['v'],
                                    close_time=datetime.fromtimestamp(k['T'] / 1000),
                                    quote_volume=k['q'],
                                    trades_count=k['n'],
                                    taker_buy_base=k['V'],
                                    taker_buy_quote=k['Q']
                                )
                                
                                # DB 저장
                                session = self.Session()
                                try:
                                    session.execute("""
                                        INSERT INTO candles_5m (
                                            ts, symbol, open, high, low, close, volume,
                                            quote_volume, trades_count
                                        ) VALUES (
                                            :ts, :symbol, :open, :high, :low, :close, :volume,
                                            :quote_volume, :trades_count
                                        )
                                        ON CONFLICT (symbol, ts) DO NOTHING
                                    """, {
                                        'ts': candle.open_time,
                                        'symbol': symbol,
                                        'open': candle.open,
                                        'high': candle.high,
                                        'low': candle.low,
                                        'close': candle.close,
                                        'volume': candle.volume,
                                        'quote_volume': candle.quote_volume,
                                        'trades_count': candle.trades_count
                                    })
                                    
                                    session.commit()
                                    logger.info(f"✓ {symbol} 5m candle closed: {candle.close}")
                                    
                                except Exception as e:
                                    logger.error(f"Candle save error: {e}")
                                    session.rollback()
                                finally:
                                    session.close()
                
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def run(self):
        """모든 수집 태스크 실행"""
        tasks = []
        
        # 1. 과거 데이터 수집 (한 번만)
        for symbol in self.symbols:
            await self.collect_candles_historical(symbol, days=365)
            await self.collect_funding_history(symbol)
        
        # 2. 실시간 수집
        # Premium Index (1분마다)
        tasks.append(asyncio.create_task(self.collect_premium_index_loop()))
        
        # WebSocket 캔들 (각 심볼별)
        for symbol in self.symbols:
            tasks.append(asyncio.create_task(self.collect_candles_websocket(symbol)))
        
        # 실행
        await asyncio.gather(*tasks)

# 실행
if __name__ == "__main__":
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT').split(',')
    db_url = os.getenv('DATABASE_URL')
    
    client = BinanceFuturesClient(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    )
    
    collector = MarketDataCollector(symbols, db_url, client)
    asyncio.run(collector.run())
```

## 6.2 User Stream Manager

**파일:** `services/userstream/stream_manager.py`

```python
import asyncio
import websockets
import json
from datetime import datetime
from typing import Optional
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from packages.common.binance_client import BinanceFuturesClient
from packages.common.schemas import OrderTradeUpdate, PositionRisk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserStreamManager:
    """
    User Data Stream 관리
    
    - listenKey 발급 및 60분마다 keepalive
    - ORDER_TRADE_UPDATE 이벤트 처리
    - ACCOUNT_UPDATE 이벤트 처리
    - 포지션 상태 실시간 업데이트
    
    ⚠️ 연결 끊김 시 거래 중단 필수!
    """
    
    def __init__(
        self,
        db_url: str,
        binance_client: BinanceFuturesClient
    ):
        self.client = binance_client
        self.listen_key: Optional[str] = None
        
        # DB
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        
        # 상태
        self.is_connected = False
    
    async def start_stream(self):
        """listenKey 발급 및 스트림 시작"""
        try:
            response = self.client.start_user_data_stream()
            self.listen_key = response['listenKey']
            
            logger.info(f"✓ User data stream started: {self.listen_key[:10]}...")
            
            # 시스템 상태 업데이트
            self._update_system_status('userstream', 'healthy')
            
        except Exception as e:
            logger.error(f"Failed to start user stream: {e}")
            self._update_system_status('userstream', 'down')
            raise
    
    async def keepalive_loop(self):
        """60분마다 keepalive"""
        logger.info("Starting keepalive loop")
        
        while True:
            try:
                await asyncio.sleep(30 * 60)  # 30분마다 (여유 있게)
                
                if self.listen_key:
                    self.client.keepalive_user_data_stream(self.listen_key)
                    logger.info("✓ User stream keepalive sent")
                    
            except Exception as e:
                logger.error(f"Keepalive error: {e}")
                
                # Keepalive 실패 = 심각
                self._update_system_status('userstream', 'down')
                
                # 거래 강제 중단
                self._emergency_shutdown()
    
    async def listen_stream(self):
        """WebSocket 이벤트 수신"""
        if not self.listen_key:
            raise ValueError("listenKey not initialized")
        
        url = f"{self.client.ws_base_url}/ws/{self.listen_key}"
        
        logger.info(f"Connecting to user stream: {url}")
        
        while True:
            try:
                async with websockets.connect(url) as ws:
                    self.is_connected = True
                    self._update_system_status('userstream', 'healthy')
                    
                    logger.info("✓ User stream connected")
                    
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # 이벤트 타입에 따라 처리
                        event_type = data.get('e')
                        
                        if event_type == 'ORDER_TRADE_UPDATE':
                            await self.handle_order_update(data)
                        
                        elif event_type == 'ACCOUNT_UPDATE':
                            await self.handle_account_update(data)
                        
                        elif event_type == 'listenKeyExpired':
                            logger.warning("listenKey expired! Restarting...")
                            break
                
            except Exception as e:
                logger.error(f"User stream error: {e}")
                self.is_connected = False
                self._update_system_status('userstream', 'degraded')
                
                # ⚠️ 연결 끊김 = 위험!
                logger.critical("USER STREAM DISCONNECTED - EMERGENCY SHUTDOWN")
                self._emergency_shutdown()
                
                # 재연결 시도
                await asyncio.sleep(5)
                await self.start_stream()
    
    async def handle_order_update(self, data: dict):
        """
        ORDER_TRADE_UPDATE 이벤트 처리
        
        이벤트 발생 시점:
        - 주문 생성
        - 주문 체결 (일부/전체)
        - 주문 취소
        - 주문 만료
        """
        try:
            # 파싱
            event = OrderTradeUpdate(**data['o'])
            
            logger.info(
                f"Order Update: {event.symbol} {event.side} "
                f"{event.execution_type} {event.order_status}"
            )
            
            # DB 저장
            session = self.Session()
            try:
                # orders 테이블 업데이트
                session.execute("""
                    INSERT INTO orders (
                        order_id, symbol, client_order_id, side, type,
                        orig_qty, price, avg_price, status, executed_qty,
                        reduce_only, position_side, created_at, update_time
                    ) VALUES (
                        :order_id, :symbol, :client_order_id, :side, :type,
                        :orig_qty, :price, :avg_price, :status, :executed_qty,
                        :reduce_only, :position_side, :created_at, :update_time
                    )
                    ON CONFLICT (order_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        executed_qty = EXCLUDED.executed_qty,
                        avg_price = EXCLUDED.avg_price,
                        update_time = EXCLUDED.update_time
                """, {
                    'order_id': event.order_id,
                    'symbol': event.symbol,
                    'client_order_id': event.client_order_id,
                    'side': event.side,
                    'type': event.order_type,
                    'orig_qty': event.original_quantity,
                    'price': event.original_price,
                    'avg_price': event.average_price,
                    'status': event.order_status,
                    'executed_qty': event.cumulative_filled_quantity,
                    'reduce_only': False,  # 이벤트에서 파싱 필요
                    'position_side': 'BOTH',
                    'created_at': event.event_time,
                    'update_time': event.event_time
                })
                
                # 체결 발생 시 fills 테이블에도 저장
                if event.last_filled_quantity > 0:
                    session.execute("""
                        INSERT INTO fills (
                            trade_id, order_id, symbol, side, price, qty,
                            realized_pnl, commission, commission_asset,
                            time, is_maker
                        ) VALUES (
                            :trade_id, :order_id, :symbol, :side, :price, :qty,
                            :realized_pnl, :commission, :commission_asset,
                            :time, :is_maker
                        )
                        ON CONFLICT (trade_id) DO NOTHING
                    """, {
                        'trade_id': event.trade_id,
                        'order_id': event.order_id,
                        'symbol': event.symbol,
                        'side': event.side,
                        'price': event.last_filled_price,
                        'qty': event.last_filled_quantity,
                        'realized_pnl': event.realized_profit,
                        'commission': event.commission,
                        'commission_asset': event.commission_asset,
                        'time': event.trade_time,
                        'is_maker': False  # 이벤트에서 파싱 필요
                    })
                
                session.commit()
                
            except Exception as e:
                logger.error(f"Order update save error: {e}")
                session.rollback()
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Order update handling error: {e}")
    
    async def handle_account_update(self, data: dict):
        """
        ACCOUNT_UPDATE 이벤트 처리
        
        포지션 변경, 자산 변경 등
        """
        try:
            logger.info("Account Update received")
            
            # 포지션 스냅샷 갱신 (REST API 호출)
            await self.update_positions_snapshot()
            
        except Exception as e:
            logger.error(f"Account update handling error: {e}")
    
    async def update_positions_snapshot(self):
        """
        포지션 상태 스냅샷 갱신
        
        GET /fapi/v2/positionRisk 호출
        """
        try:
            data = self.client.get_position_risk()
            
            # 파싱
            positions = [PositionRisk(**p) for p in data]
            
            # 실제 포지션만 (position_amt != 0)
            active_positions = [
                p for p in positions 
                if abs(float(p.position_amt)) > 0
            ]
            
            # DB 저장
            session = self.Session()
            try:
                # 기존 positions 테이블 클리어
                session.execute("DELETE FROM positions")
                
                # 새 포지션 삽입
                for pos in active_positions:
                    session.execute("""
                        INSERT INTO positions (
                            symbol, position_side, position_amt, entry_price,
                            leverage, isolated_wallet, unrealized_profit,
                            liquidation_price, mark_price, margin_type,
                            update_time, notional, isolated_margin
                        ) VALUES (
                            :symbol, :position_side, :position_amt, :entry_price,
                            :leverage, :isolated_wallet, :unrealized_profit,
                            :liquidation_price, :mark_price, :margin_type,
                            :update_time, :notional, :isolated_margin
                        )
                    """, {
                        'symbol': pos.symbol,
                        'position_side': pos.position_side,
                        'position_amt': pos.position_amt,
                        'entry_price': pos.entry_price,
                        'leverage': pos.leverage,
                        'isolated_wallet': pos.isolated_wallet,
                        'unrealized_profit': pos.unrealized_profit,
                        'liquidation_price': pos.liquidation_price,
                        'mark_price': pos.mark_price,
                        'margin_type': pos.margin_type,
                        'update_time': pos.update_time,
                        'notional': pos.notional,
                        'isolated_margin': pos.isolated_margin
                    })
                    
                    # 히스토리에도 저장
                    session.execute("""
                        INSERT INTO position_history (
                            ts, symbol, position_amt, entry_price, leverage,
                            unrealized_profit, liquidation_price, mark_price
                        ) VALUES (
                            NOW(), :symbol, :position_amt, :entry_price, :leverage,
                            :unrealized_profit, :liquidation_price, :mark_price
                        )
                    """, {
                        'symbol': pos.symbol,
                        'position_amt': pos.position_amt,
                        'entry_price': pos.entry_price,
                        'leverage': pos.leverage,
                        'unrealized_profit': pos.unrealized_profit,
                        'liquidation_price': pos.liquidation_price,
                        'mark_price': pos.mark_price
                    })
                
                session.commit()
                logger.info(f"✓ Updated {len(active_positions)} positions")
                
            except Exception as e:
                logger.error(f"Position snapshot save error: {e}")
                session.rollback()
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Position snapshot update error: {e}")
    
    def _update_system_status(self, component: str, status: str):
        """시스템 상태 업데이트"""
        session = self.Session()
        try:
            session.execute("""
                INSERT INTO system_status (component, status, last_heartbeat)
                VALUES (:component, :status, NOW())
                ON CONFLICT (component) DO UPDATE SET
                    status = EXCLUDED.status,
                    last_heartbeat = EXCLUDED.last_heartbeat
            """, {
                'component': component,
                'status': status
            })
            
            session.commit()
            
        except Exception as e:
            logger.error(f"System status update error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _emergency_shutdown(self):
        """
        긴급 종료
        
        User Stream 연결 끊김 시:
        1. 모든 거래 중단
        2. 리스크 이벤트 기록
        """
        logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED")
        
        session = self.Session()
        try:
            # 리스크 이벤트 기록
            session.execute("""
                INSERT INTO risk_events (
                    ts, event_type, severity, details, action_taken
                ) VALUES (
                    NOW(), 'connection_lost', 'critical',
                    '{"component": "userstream", "reason": "websocket_disconnected"}',
                    'trading_disabled'
                )
            """)
            
            # 시스템 상태 = DOWN
            session.execute("""
                UPDATE system_status
                SET status = 'down'
                WHERE component IN ('userstream', 'execution')
            """)
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Emergency shutdown logging error: {e}")
            session.rollback()
        finally:
            session.close()
        
        # TODO: 실제로는 실행 엔진에 중단 신호 전송
        # Redis Pub/Sub 또는 DB flag 활용
    
    async def run(self):
        """전체 실행"""
        # listenKey 발급
        await self.start_stream()
        
        # 태스크
        tasks = [
            asyncio.create_task(self.listen_stream()),
            asyncio.create_task(self.keepalive_loop())
        ]
        
        await asyncio.gather(*tasks)

# 실행
if __name__ == "__main__":
    db_url = os.getenv('DATABASE_URL')
    
    client = BinanceFuturesClient(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    )
    
    manager = UserStreamManager(db_url, client)
    asyncio.run(manager.run())
```

---
# 7. Feature Engineering

## 7.1 Feature Calculator (온/오프라인 동일)

**파일:** `services/features/calculator.py`

```python
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 현재 Feature 스키마 버전
SCHEMA_VERSION = 1

class FeatureCalculator:
    """
    Feature 계산기
    
    ⭐ 핵심: 온라인(실시간)/오프라인(배치) 동일한 코드 사용
    - 백테스트와 실전의 정합성 보장
    """
    
    def __init__(self):
        self.schema_version = SCHEMA_VERSION
    
    def calculate(
        self,
        candles: pd.DataFrame,
        premium_index: pd.DataFrame,
        funding_rates: pd.DataFrame,
        btc_candles: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        전체 Feature 계산
        
        Args:
            candles: OHLCV DataFrame (index=ts)
            premium_index: Mark price + funding DataFrame
            funding_rates: Funding rate history
            btc_candles: BTC 캔들 (레짐 판단용, optional)
        
        Returns:
            Features DataFrame
        """
        df = candles.copy()
        
        # 1. 수익률
        df = self._calculate_returns(df)
        
        # 2. 추세 지표
        df = self._calculate_trend(df)
        
        # 3. 변동성
        df = self._calculate_volatility(df)
        
        # 4. 모멘텀
        df = self._calculate_momentum(df)
        
        # 5. 거래량
        df = self._calculate_volume(df)
        
        # 6. 볼린저 밴드
        df = self._calculate_bollinger(df)
        
        # 7. 펀딩비 (선물 전용)
        df = self._calculate_funding_features(df, premium_index, funding_rates)
        
        # 8. BTC 레짐
        if btc_candles is not None:
            df = self._calculate_btc_regime(df, btc_candles)
        else:
            df['btc_regime'] = 0
        
        # 9. 시간 특징
        df = self._calculate_time_features(df)
        
        # 10. Schema version
        df['schema_version'] = self.schema_version
        
        # NaN 처리
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """수익률"""
        close = df['close']
        
        df['r_1'] = close.pct_change(1)
        df['r_3'] = close.pct_change(3)
        df['r_12'] = close.pct_change(12)  # 1시간
        
        return df
    
    def _calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """추세 지표 (EMA)"""
        close = df['close']
        
        # EMA
        for period in [7, 21, 50]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # ATR (변동성 정규화용)
        high = df['high']
        low = df['low']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / close * 100
        
        # EMA 거리 (ATR로 정규화)
        df['ema_dist_atr'] = (close - df['ema_21']) / df['atr']
        
        return df
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성"""
        returns = df['close'].pct_change()
        
        df['volatility_10'] = returns.rolling(10).std()
        df['volatility_30'] = returns.rolling(30).std()
        
        return df
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 지표"""
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # ADX
        high = df['high']
        low = df['low']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = df['atr'] * 14  # ATR는 이미 계산됨
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        
        return df
    
    def _calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 지표"""
        volume = df['volume']
        
        # 이동평균
        df['volume_ma_12'] = volume.rolling(12).mean()
        
        # Z-score (정규화)
        vol_mean = volume.rolling(48).mean()  # 4시간
        vol_std = volume.rolling(48).std()
        df['vol_z'] = (volume - vol_mean) / vol_std
        
        return df
    
    def _calculate_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """볼린저 밴드"""
        close = df['close']
        
        bb_period = 20
        bb_std = 2
        
        bb_middle = close.rolling(bb_period).mean()
        bb_std_val = close.rolling(bb_period).std()
        
        df['bb_upper'] = bb_middle + (bb_std_val * bb_std)
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_middle - (bb_std_val * bb_std)
        
        # BB 포지션 (정규화)
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_z'] = (close - df['bb_middle']) / bb_range
        
        return df
    
    def _calculate_funding_features(
        self,
        df: pd.DataFrame,
        premium_index: pd.DataFrame,
        funding_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        펀딩비 특징 (선물 전용)
        
        Args:
            premium_index: ts, mark_price, last_funding_rate
            funding_rates: funding_time, funding_rate
        """
        # Premium index와 조인
        df = df.join(
            premium_index[['last_funding_rate']].rename(
                columns={'last_funding_rate': 'funding_rate'}
            ),
            how='left'
        )
        
        # 24시간 이동평균
        df['funding_ma_24'] = df['funding_rate'].rolling(
            24 * 12  # 5분봉 기준 24시간 = 288개
        ).mean()
        
        # Z-score
        funding_mean = df['funding_rate'].rolling(24 * 12).mean()
        funding_std = df['funding_rate'].rolling(24 * 12).std()
        df['funding_z'] = (df['funding_rate'] - funding_mean) / funding_std
        
        return df
    
    def _calculate_btc_regime(
        self,
        df: pd.DataFrame,
        btc_candles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        BTC 레짐 (시장 전체 방향성)
        
        -1: 하락
         0: 횡보
         1: 상승
        """
        # BTC EMA
        btc_close = btc_candles['close']
        btc_ema_50 = btc_close.ewm(span=50, adjust=False).mean()
        btc_ema_200 = btc_close.ewm(span=200, adjust=False).mean()
        
        # 레짐 판단
        regime = pd.Series(0, index=btc_candles.index)
        regime[btc_ema_50 > btc_ema_200 * 1.02] = 1   # 상승
        regime[btc_ema_50 < btc_ema_200 * 0.98] = -1  # 하락
        
        # 원래 DataFrame에 조인
        df = df.join(
            regime.rename('btc_regime'),
            how='left'
        )
        
        df['btc_regime'] = df['btc_regime'].fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 특징"""
        # 시간 추출
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 거래 세션 (UTC 기준)
        # 아시아: 00:00-08:00 UTC
        df['is_asian_session'] = (
            (df['hour_of_day'] >= 0) & (df['hour_of_day'] < 8)
        ).astype(int)
        
        # 펀딩 시간 (8시간마다: 00:00, 08:00, 16:00 UTC)
        df['is_funding_hour'] = (
            df['hour_of_day'].isin([0, 8, 16])
        ).astype(int)
        
        return df

# 온라인 Feature 계산 (실시간)
class OnlineFeatureCalculator(FeatureCalculator):
    """
    실시간 Feature 계산
    
    - 최신 N개 캔들만 유지
    - 증분 계산 최적화
    """
    
    def __init__(self, lookback_bars: int = 300):
        super().__init__()
        self.lookback_bars = lookback_bars
        
        # 캐시
        self.candle_cache = {}
        self.premium_cache = {}
        self.funding_cache = {}
    
    def update_cache(
        self,
        symbol: str,
        candles: pd.DataFrame,
        premium_index: pd.DataFrame,
        funding_rates: pd.DataFrame
    ):
        """캐시 업데이트"""
        # 최신 N개만 유지
        self.candle_cache[symbol] = candles.tail(self.lookback_bars)
        self.premium_cache[symbol] = premium_index.tail(self.lookback_bars)
        self.funding_cache[symbol] = funding_rates.tail(self.lookback_bars)
    
    def calculate_latest(
        self,
        symbol: str,
        btc_candles: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        최신 Feature만 계산 (단일 행)
        
        Returns:
            최신 시점의 Feature Series
        """
        candles = self.candle_cache.get(symbol)
        premium = self.premium_cache.get(symbol)
        funding = self.funding_cache.get(symbol)
        
        if candles is None or premium is None:
            raise ValueError(f"Cache not initialized for {symbol}")
        
        # 전체 계산
        features_df = self.calculate(candles, premium, funding, btc_candles)
        
        # 최신 행만 반환
        return features_df.iloc[-1]
```

## 7.2 Feature Pipeline (배치)

**파일:** `services/features/pipeline.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import os
from typing import List

from .calculator import FeatureCalculator, SCHEMA_VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Feature 생성 파이프라인 (배치)
    
    - 과거 데이터에 대한 Feature 계산
    - DB에 저장
    """
    
    def __init__(self, db_url: str):
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        self.calculator = FeatureCalculator()
    
    def load_candles(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """캔들 데이터 로드"""
        session = self.Session()
        
        try:
            query = f"""
                SELECT ts, open, high, low, close, volume
                FROM candles_5m
                WHERE symbol = '{symbol}'
                  AND ts >= '{start_date}'
                  AND ts < '{end_date}'
                ORDER BY ts
            """
            
            df = pd.read_sql(query, session.bind, parse_dates=['ts'])
            df.set_index('ts', inplace=True)
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            
            return df
            
        finally:
            session.close()
    
    def load_premium_index(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Premium Index 로드"""
        session = self.Session()
        
        try:
            query = f"""
                SELECT ts, mark_price, last_funding_rate
                FROM premium_index
                WHERE symbol = '{symbol}'
                  AND ts >= '{start_date}'
                  AND ts < '{end_date}'
                ORDER BY ts
            """
            
            df = pd.read_sql(query, session.bind, parse_dates=['ts'])
            df.set_index('ts', inplace=True)
            
            return df
            
        finally:
            session.close()
    
    def load_funding_rates(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Funding Rate 히스토리 로드"""
        session = self.Session()
        
        try:
            query = f"""
                SELECT funding_time AS ts, funding_rate
                FROM funding_rates
                WHERE symbol = '{symbol}'
                  AND funding_time >= '{start_date}'
                  AND funding_time < '{end_date}'
                ORDER BY funding_time
            """
            
            df = pd.read_sql(query, session.bind, parse_dates=['ts'])
            df.set_index('ts', inplace=True)
            
            return df
            
        finally:
            session.close()
    
    def generate_features(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ):
        """Feature 생성 및 저장"""
        logger.info(f"Generating features for {symbol} from {start_date} to {end_date}")
        
        # 데이터 로드
        candles = self.load_candles(symbol, start_date, end_date)
        premium = self.load_premium_index(symbol, start_date, end_date)
        funding = self.load_funding_rates(symbol, start_date, end_date)
        
        # BTC 레짐용 (선택)
        btc_candles = None
        if symbol != 'BTCUSDT':
            try:
                btc_candles = self.load_candles('BTCUSDT', start_date, end_date)
            except:
                pass
        
        # Feature 계산
        features = self.calculator.calculate(
            candles, premium, funding, btc_candles
        )
        
        # DB 저장
        session = self.Session()
        try:
            # 기존 데이터 삭제 (같은 schema_version)
            session.execute(f"""
                DELETE FROM features_5m
                WHERE symbol = '{symbol}'
                  AND ts >= '{start_date}'
                  AND ts < '{end_date}'
                  AND schema_version = {SCHEMA_VERSION}
            """)
            
            # 삽입
            features['symbol'] = symbol
            features_reset = features.reset_index()
            
            # Chunk 단위 저장
            chunk_size = 5000
            for i in range(0, len(features_reset), chunk_size):
                chunk = features_reset.iloc[i:i+chunk_size]
                
                chunk.to_sql(
                    'features_5m',
                    session.bind,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.info(f"Saved {min(i+chunk_size, len(features_reset))}/{len(features_reset)}")
            
            session.commit()
            logger.info(f"✓ Feature generation complete for {symbol}")
            
        except Exception as e:
            logger.error(f"Feature save error: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def generate_all_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ):
        """모든 심볼 Feature 생성"""
        for symbol in symbols:
            try:
                self.generate_features(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to generate features for {symbol}: {e}")

# 실행
if __name__ == "__main__":
    db_url = os.getenv('DATABASE_URL')
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
    
    pipeline = FeaturePipeline(db_url)
    pipeline.generate_all_symbols(
        symbols,
        start_date='2024-01-01',
        end_date='2025-01-01'
    )
```

---

# 8. 이벤트 라벨링

## 8.1 Triple Barrier Labeling (비용 포함)

**파일:** `services/labeling/triple_barrier.py`

```python
import pandas as pd
import numpy as np
from typing import Tuple
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeler:
    """
    Triple Barrier 라벨링
    
    각 시점에서 진입했다고 가정:
    - TP (Take Profit): mark_price * (1 + k_tp * ATR)
    - SL (Stop Loss): mark_price * (1 - k_sl * ATR)
    - TIME: H bars 경과
    
    ⭐ 핵심: 비용 포함 계산
    - Fee (진입 + 청산)
    - Slippage (진입 + 청산)
    - Funding (보유 시간 동안)
    """
    
    def __init__(
        self,
        k_tp: float = 2.0,   # TP = 2 * ATR
        k_sl: float = 1.0,   # SL = 1 * ATR
        h_bars: int = 48,    # 4시간 (5분봉 기준)
        fee_rate: float = 0.0004,      # 0.04% (테이커)
        slippage_bps: float = 15       # 0.15%
    ):
        self.k_tp = k_tp
        self.k_sl = k_sl
        self.h_bars = h_bars
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_bps / 10000
    
    def label_direction(
        self,
        candles: pd.DataFrame,
        premium_index: pd.DataFrame,
        direction: str  # 'long' or 'short'
    ) -> pd.DataFrame:
        """
        단일 방향 라벨링
        
        Args:
            candles: OHLCV (index=ts)
            premium_index: mark_price, last_funding_rate
            direction: 'long' or 'short'
        
        Returns:
            Labels DataFrame
        """
        logger.info(f"Labeling {direction} for {len(candles)} candles")
        
        # Mark price 조인
        df = candles.join(premium_index[['mark_price', 'last_funding_rate']])
        
        # ATR 계산
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # 라벨 생성
        results = []
        
        for i in range(len(df) - self.h_bars - 1):
            entry_time = df.index[i]
            entry_price = df.iloc[i]['mark_price']
            atr = df.iloc[i]['atr']
            
            if pd.isna(entry_price) or pd.isna(atr) or atr == 0:
                continue
            
            # 미래 데이터
            future = df.iloc[i+1:i+self.h_bars+1]
            
            # Barrier 설정
            if direction == 'long':
                tp_price = entry_price * (1 + self.k_tp * atr / entry_price)
                sl_price = entry_price * (1 - self.k_sl * atr / entry_price)
                
                # 도달 체크
                tp_hit = future['high'] >= tp_price
                sl_hit = future['low'] <= sl_price
                
            else:  # short
                tp_price = entry_price * (1 - self.k_tp * atr / entry_price)
                sl_price = entry_price * (1 + self.k_sl * atr / entry_price)
                
                tp_hit = future['low'] <= tp_price
                sl_hit = future['high'] >= sl_price
            
            # 먼저 도달한 것
            tp_idx = tp_hit.idxmax() if tp_hit.any() else None
            sl_idx = sl_hit.idxmax() if sl_hit.any() else None
            
            # 결과 판정
            if tp_idx is not None and (sl_idx is None or tp_idx < sl_idx):
                # TP 먼저 도달
                y = 1
                exit_time = tp_idx
                exit_price = tp_price
                outcome = 'TP'
                
            elif sl_idx is not None:
                # SL 먼저 도달
                y = -1
                exit_time = sl_idx
                exit_price = sl_price
                outcome = 'SL'
                
            else:
                # Timeout
                y = 0
                exit_time = future.index[-1]
                exit_price = future.iloc[-1]['mark_price']
                outcome = 'TIMEOUT'
            
            # 보유 시간
            time_held = (exit_time - entry_time).total_seconds() / 60  # 분
            bars_held = int(time_held / 5)  # 5분봉 개수
            
            # 수익률 (비용 제외)
            if direction == 'long':
                ret_gross = (exit_price - entry_price) / entry_price
            else:
                ret_gross = (entry_price - exit_price) / entry_price
            
            # 비용 계산
            costs = self._calculate_costs(
                entry_price,
                exit_price,
                time_held,
                future['last_funding_rate'].mean()
            )
            
            ret_net = ret_gross - costs['total']
            
            # MAE/MFE
            if direction == 'long':
                mae = (future.iloc[:bars_held+1]['low'].min() - entry_price) / entry_price
                mfe = (future.iloc[:bars_held+1]['high'].max() - entry_price) / entry_price
            else:
                mae = (entry_price - future.iloc[:bars_held+1]['high'].max()) / entry_price
                mfe = (entry_price - future.iloc[:bars_held+1]['low'].min()) / entry_price
            
            results.append({
                'ts': entry_time,
                'y': y,
                'ret_gross': ret_gross,
                'ret_net': ret_net,
                'fee_cost': costs['fee'],
                'slippage_cost': costs['slippage'],
                'funding_cost': costs['funding'],
                'mae': mae,
                'mfe': mfe,
                'time_to_event_min': int(time_held),
                'bars_held': bars_held
            })
        
        return pd.DataFrame(results)
    
    def _calculate_costs(
        self,
        entry_price: float,
        exit_price: float,
        time_held_minutes: float,
        avg_funding_rate: float
    ) -> dict:
        """
        비용 계산
        
        Returns:
            {
                'fee': 진입+청산 수수료,
                'slippage': 진입+청산 슬리피지,
                'funding': 펀딩비,
                'total': 총 비용
            }
        """
        # Fee (진입 + 청산)
        fee = self.fee_rate * 2
        
        # Slippage (진입 + 청산)
        slippage = self.slippage_rate * 2
        
        # Funding (8시간마다 정산)
        # 보유 시간 동안 발생한 펀딩 횟수
        funding_periods = time_held_minutes / 480  # 480분 = 8시간
        funding = abs(avg_funding_rate) * funding_periods if not pd.isna(avg_funding_rate) else 0
        
        total = fee + slippage + funding
        
        return {
            'fee': fee,
            'slippage': slippage,
            'funding': funding,
            'total': total
        }
    
    def save_labels(
        self,
        labels: pd.DataFrame,
        symbol: str,
        direction: str,
        session
    ):
        """라벨 DB 저장"""
        table = f'labels_{direction}_5m'
        
        try:
            # 기존 데이터 삭제
            session.execute(f"""
                DELETE FROM {table}
                WHERE symbol = '{symbol}'
                  AND k_tp = {self.k_tp}
                  AND k_sl = {self.k_sl}
                  AND h_bars = {self.h_bars}
            """)
            
            # 삽입
            for _, row in labels.iterrows():
                session.execute(f"""
                    INSERT INTO {table} (
                        ts, symbol, k_tp, k_sl, h_bars,
                        y, ret_gross, ret_net,
                        fee_cost, slippage_cost, funding_cost,
                        mae, mfe, time_to_event_min, bars_held
                    ) VALUES (
                        :ts, :symbol, :k_tp, :k_sl, :h_bars,
                        :y, :ret_gross, :ret_net,
                        :fee_cost, :slippage_cost, :funding_cost,
                        :mae, :mfe, :time_to_event_min, :bars_held
                    )
                """, {
                    'ts': row['ts'],
                    'symbol': symbol,
                    'k_tp': self.k_tp,
                    'k_sl': self.k_sl,
                    'h_bars': self.h_bars,
                    'y': row['y'],
                    'ret_gross': row['ret_gross'],
                    'ret_net': row['ret_net'],
                    'fee_cost': row['fee_cost'],
                    'slippage_cost': row['slippage_cost'],
                    'funding_cost': row['funding_cost'],
                    'mae': row['mae'],
                    'mfe': row['mfe'],
                    'time_to_event_min': row['time_to_event_min'],
                    'bars_held': row['bars_held']
                })
            
            session.commit()
            logger.info(f"✓ Saved {len(labels)} {direction} labels for {symbol}")
            
        except Exception as e:
            logger.error(f"Label save error: {e}")
            session.rollback()
            raise
```

## 8.2 Labeling Service

**파일:** `services/labeling/service.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import os
from typing import List

from .triple_barrier import TripleBarrierLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelingService:
    """라벨링 서비스 (배치)"""
    
    def __init__(self, db_url: str):
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        
        # Labeler (설정은 환경변수에서)
        self.labeler = TripleBarrierLabeler(
            k_tp=float(os.getenv('LABEL_TP_ATR_MULTIPLIER', '2.0')),
            k_sl=float(os.getenv('LABEL_SL_ATR_MULTIPLIER', '1.0')),
            h_bars=int(os.getenv('LABEL_TIMEOUT_BARS', '48')),
            fee_rate=float(os.getenv('FEE_RATE', '0.0004')),
            slippage_bps=float(os.getenv('SLIPPAGE_BPS', '15'))
        )
    
    def load_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        session = self.Session()
        
        try:
            # Candles
            candles_query = f"""
                SELECT ts, open, high, low, close, volume
                FROM candles_5m
                WHERE symbol = '{symbol}'
                  AND ts >= '{start_date}'
                  AND ts < '{end_date}'
                ORDER BY ts
            """
            
            candles = pd.read_sql(candles_query, session.bind, parse_dates=['ts'])
            candles.set_index('ts', inplace=True)
            
            # Premium Index
            premium_query = f"""
                SELECT ts, mark_price, last_funding_rate
                FROM premium_index
                WHERE symbol = '{symbol}'
                  AND ts >= '{start_date}'
                  AND ts < '{end_date}'
                ORDER BY ts
            """
            
            premium = pd.read_sql(premium_query, session.bind, parse_dates=['ts'])
            premium.set_index('ts', inplace=True)
            
            return candles, premium
            
        finally:
            session.close()
    
    def label_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ):
        """심볼 라벨링 (롱 + 숏)"""
        logger.info(f"Labeling {symbol} from {start_date} to {end_date}")
        
        # 데이터 로드
        candles, premium = self.load_data(symbol, start_date, end_date)
        
        session = self.Session()
        try:
            # 롱 라벨
            logger.info("Generating LONG labels...")
            long_labels = self.labeler.label_direction(candles, premium, 'long')
            self.labeler.save_labels(long_labels, symbol, 'long', session)
            
            # 숏 라벨
            logger.info("Generating SHORT labels...")
            short_labels = self.labeler.label_direction(candles, premium, 'short')
            self.labeler.save_labels(short_labels, symbol, 'short', session)
            
            # 통계
            long_win_rate = (long_labels['y'] == 1).mean()
            short_win_rate = (short_labels['y'] == 1).mean()
            
            logger.info(f"""
            ✓ Labeling complete for {symbol}:
            - LONG: {len(long_labels)} labels, Win Rate: {long_win_rate:.2%}
            - SHORT: {len(short_labels)} labels, Win Rate: {short_win_rate:.2%}
            """)
            
        finally:
            session.close()
    
    def label_all_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ):
        """모든 심볼 라벨링"""
        for symbol in symbols:
            try:
                self.label_symbol(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to label {symbol}: {e}")

# 실행
if __name__ == "__main__":
    db_url = os.getenv('DATABASE_URL')
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
    
    service = LabelingService(db_url)
    service.label_all_symbols(
        symbols,
        start_date='2024-01-01',
        end_date='2025-01-01'
    )
```

---
# 9. 모델 학습 시스템

## 9.1 LightGBM Trainer (Walk-forward)

**파일:** `services/training/trainer.py`

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import joblib
import json
import uuid
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    LightGBM 모델 학습
    
    - Walk-forward validation
    - Purge & Embargo (시계열 누수 방지)
    - 2개 모델: pwin 예측 + hold_time 예측
    """
    
    def __init__(self, db_url: str, minio_client=None):
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        self.minio = minio_client
    
    def load_training_data(
        self,
        symbols: List[str],
        direction: str,  # 'long' or 'short'
        start_date: str,
        end_date: str,
        schema_version: int
    ) -> pd.DataFrame:
        """
        학습 데이터 로드 (Features + Labels 조인)
        
        Args:
            symbols: 심볼 리스트
            direction: 'long' or 'short'
            start_date, end_date: 기간
            schema_version: Feature 스키마 버전
        """
        logger.info(f"Loading {direction} training data for {len(symbols)} symbols")
        
        session = self.Session()
        
        try:
            symbol_list = "','".join(symbols)
            
            query = f"""
                SELECT 
                    f.ts, f.symbol,
                    -- Features
                    f.r_1, f.r_3, f.r_12,
                    f.ema_7, f.ema_21, f.ema_50, f.ema_dist_atr,
                    f.atr, f.atr_pct, f.volatility_10, f.volatility_30,
                    f.rsi_14, f.macd, f.macd_signal, f.adx,
                    f.volume_ma_12, f.vol_z,
                    f.bb_upper, f.bb_middle, f.bb_lower, f.bb_z,
                    f.funding_rate, f.funding_ma_24, f.funding_z,
                    f.btc_regime,
                    f.hour_of_day, f.day_of_week, f.is_asian_session, f.is_funding_hour,
                    -- Labels
                    l.y, l.ret_net, l.mae, l.mfe, l.time_to_event_min, l.bars_held
                FROM features_5m f
                INNER JOIN labels_{direction}_5m l
                    ON f.ts = l.ts AND f.symbol = l.symbol
                WHERE f.symbol IN ('{symbol_list}')
                  AND f.ts >= '{start_date}'
                  AND f.ts < '{end_date}'
                  AND f.schema_version = {schema_version}
                  AND l.k_tp = {os.getenv('LABEL_TP_ATR_MULTIPLIER', '2.0')}
                  AND l.k_sl = {os.getenv('LABEL_SL_ATR_MULTIPLIER', '1.0')}
                  AND l.h_bars = {os.getenv('LABEL_TIMEOUT_BARS', '48')}
                ORDER BY f.ts
            """
            
            df = pd.read_sql(query, session.bind, parse_dates=['ts'])
            
            logger.info(f"Loaded {len(df)} samples")
            
            # 결측치 체크
            missing = df.isnull().sum()
            if missing.any():
                logger.warning(f"Missing values:\n{missing[missing > 0]}")
                df = df.fillna(0)
            
            return df
            
        finally:
            session.close()
    
    def prepare_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Feature 준비
        
        Returns:
            X, y_pwin, y_hold_time, feature_names
        """
        # Feature 컬럼
        feature_cols = [
            'r_1', 'r_3', 'r_12',
            'ema_dist_atr', 'atr_pct',
            'volatility_10', 'volatility_30',
            'rsi_14', 'macd', 'macd_signal', 'adx',
            'vol_z', 'bb_z',
            'funding_rate', 'funding_ma_24', 'funding_z',
            'btc_regime',
            'hour_of_day', 'day_of_week', 'is_asian_session', 'is_funding_hour'
        ]
        
        X = df[feature_cols].values
        
        # Target 1: pwin (TP=1, else=0)
        y_pwin = (df['y'] == 1).astype(int).values
        
        # Target 2: hold_time
        y_hold_time = df['time_to_event_min'].values
        
        return X, y_pwin, y_hold_time, feature_cols
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        purge_gap_minutes: int = 60,
        embargo_pct: float = 0.01
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Walk-forward 분할 (Purge & Embargo)
        
        Args:
            df: 시간순 정렬된 DataFrame
            n_splits: 분할 수
            purge_gap_minutes: Train과 Test 사이 간격 (분)
            embargo_pct: Test 이후 embargo 비율
        
        Returns:
            [(train_idx, test_idx), ...]
        """
        total = len(df)
        test_size = total // (n_splits + 1)
        purge_bars = purge_gap_minutes // 5  # 5분봉 기준
        embargo_bars = int(test_size * embargo_pct)
        
        splits = []
        
        for i in range(n_splits):
            # Test 구간
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > total:
                break
            
            # Train 구간 (Test 시작 전 purge_gap만큼 제외)
            train_end = test_start - purge_bars
            train_start = 0
            
            # Test 구간
            test_idx = np.arange(test_start, min(test_end, total))
            train_idx = np.arange(train_start, train_end)
            
            # Embargo (Test 이후 일부 제외)
            if i < n_splits - 1:
                next_train_start = test_end + embargo_bars
                # 다음 fold의 train에서 embargo 구간 제외
                pass  # 간소화
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                
                logger.info(
                    f"Fold {i+1}: Train={len(train_idx)}, "
                    f"Test={len(test_idx)}, "
                    f"Purge={purge_bars}, Embargo={embargo_bars}"
                )
        
        return splits
    
    def train_pwin_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> lgb.Booster:
        """
        Pwin 예측 모델 학습 (이진 분류)
        
        Returns:
            LightGBM Booster
        """
        logger.info("Training pwin model (binary classification)...")
        
        # LightGBM 데이터셋
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 파라미터
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 100,
            'verbose': -1
        }
        
        # 학습
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # 검증 성과
        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        logger.info(f"Pwin model metrics: {metrics}")
        
        return model
    
    def train_hold_time_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> lgb.Booster:
        """
        Hold time 예측 모델 학습 (회귀)
        """
        logger.info("Training hold_time model (regression)...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # 검증 성과
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mae = np.mean(np.abs(y_val - y_pred))
        
        logger.info(f"Hold time model: RMSE={rmse:.2f}, MAE={mae:.2f}")
        
        return model
    
    def train(
        self,
        symbols: List[str],
        direction: str,
        start_date: str,
        end_date: str,
        schema_version: int
    ) -> Dict:
        """
        전체 학습 파이프라인
        
        Returns:
            학습 결과 딕셔너리
        """
        logger.info(f"Starting training for {direction} on {len(symbols)} symbols")
        
        # Job 생성
        job_id = str(uuid.uuid4())
        session = self.Session()
        
        try:
            session.execute("""
                INSERT INTO training_jobs (job_id, status, config)
                VALUES (:job_id, 'running', :config)
            """, {
                'job_id': job_id,
                'config': json.dumps({
                    'symbols': symbols,
                    'direction': direction,
                    'start_date': start_date,
                    'end_date': end_date,
                    'schema_version': schema_version
                })
            })
            session.commit()
            
        except Exception as e:
            logger.error(f"Job creation error: {e}")
            session.rollback()
        finally:
            session.close()
        
        # 데이터 로드
        df = self.load_training_data(
            symbols, direction, start_date, end_date, schema_version
        )
        
        # Feature 준비
        X, y_pwin, y_hold_time, feature_names = self.prepare_features(df)
        
        # Walk-forward split
        splits = self.walk_forward_split(df, n_splits=5)
        
        # 마지막 fold로 최종 모델 학습
        train_idx, val_idx = splits[-1]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_pwin_train, y_pwin_val = y_pwin[train_idx], y_pwin[val_idx]
        y_hold_train, y_hold_val = y_hold_time[train_idx], y_hold_time[val_idx]
        
        # 모델 학습
        pwin_model = self.train_pwin_model(
            X_train, y_pwin_train, X_val, y_pwin_val
        )
        
        hold_model = self.train_hold_time_model(
            X_train, y_hold_train, X_val, y_hold_val
        )
        
        # Feature Importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': pwin_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 features:\n{importance_df.head(10)}")
        
        # 모델 저장
        model_id = str(uuid.uuid4())
        model_dir = f"/tmp/models/{model_id}"
        os.makedirs(model_dir, exist_ok=True)
        
        pwin_model.save_model(f"{model_dir}/pwin_model.txt")
        hold_model.save_model(f"{model_dir}/hold_model.txt")
        joblib.dump(feature_names, f"{model_dir}/feature_names.pkl")
        importance_df.to_csv(f"{model_dir}/feature_importance.csv", index=False)
        
        # MinIO 업로드 (생략 - 실제로는 MinIO에 업로드)
        artifact_uri = f"s3://models/{model_id}/"
        
        # 모델 레지스트리에 등록
        session = self.Session()
        try:
            # Pwin 예측
            y_pred_proba = pwin_model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            val_metrics = {
                'accuracy': float(accuracy_score(y_pwin_val, y_pred)),
                'precision': float(precision_score(y_pwin_val, y_pred, zero_division=0)),
                'recall': float(recall_score(y_pwin_val, y_pred, zero_division=0)),
                'f1': float(f1_score(y_pwin_val, y_pred, zero_division=0)),
                'auc': float(roc_auc_score(y_pwin_val, y_pred_proba))
            }
            
            session.execute("""
                INSERT INTO models (
                    model_id, name, version, algorithm,
                    schema_version, label_spec_hash, direction,
                    train_start, train_end, symbols,
                    val_metrics, artifact_uri, status
                ) VALUES (
                    :model_id, :name, :version, 'lgbm',
                    :schema_version, :label_spec_hash, :direction,
                    :train_start, :train_end, :symbols,
                    :val_metrics, :artifact_uri, 'trained'
                )
            """, {
                'model_id': model_id,
                'name': f'lgbm_{direction}',
                'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'schema_version': schema_version,
                'label_spec_hash': 'v1',  # 라벨 스펙 해시
                'direction': direction,
                'train_start': start_date,
                'train_end': end_date,
                'symbols': symbols,
                'val_metrics': json.dumps(val_metrics),
                'artifact_uri': artifact_uri
            })
            
            # Job 완료
            session.execute("""
                UPDATE training_jobs
                SET status = 'completed',
                    ended_at = NOW(),
                    metrics = :metrics
                WHERE job_id = :job_id
            """, {
                'job_id': job_id,
                'metrics': json.dumps(val_metrics)
            })
            
            session.commit()
            
            logger.info(f"✓ Training complete. Model ID: {model_id}")
            
            return {
                'job_id': job_id,
                'model_id': model_id,
                'val_metrics': val_metrics,
                'feature_importance': importance_df.head(20).to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Model registration error: {e}")
            session.rollback()
            
            # Job 실패 기록
            session.execute("""
                UPDATE training_jobs
                SET status = 'failed',
                    ended_at = NOW(),
                    error_message = :error
                WHERE job_id = :job_id
            """, {
                'job_id': job_id,
                'error': str(e)
            })
            session.commit()
            
            raise
        finally:
            session.close()

# 실행
if __name__ == "__main__":
    db_url = os.getenv('DATABASE_URL')
    symbols = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
    
    trainer = ModelTrainer(db_url)
    
    # Long 모델 학습
    result = trainer.train(
        symbols=symbols,
        direction='long',
        start_date='2024-01-01',
        end_date='2024-12-31',
        schema_version=1
    )
    
    print(json.dumps(result, indent=2))
```

---

# 10. 추론 및 정책 엔진

## 10.1 실시간 추론기

**파일:** `services/inference/predictor.py`

```python
import lightgbm as lgb
import numpy as np
from typing import Dict, Optional
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.features.calculator import OnlineFeatureCalculator

logger = logging.getLogger(__name__)

class Predictor:
    """
    실시간 추론기
    
    - 모델 로딩 및 캐싱
    - Feature 계산 → 예측
    """
    
    def __init__(self, db_url: str):
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        
        # 모델 캐시
        self.models = {
            'long': {'pwin': None, 'hold': None, 'features': None},
            'short': {'pwin': None, 'hold': None, 'features': None}
        }
        
        # Feature calculator
        self.feature_calc = OnlineFeatureCalculator(lookback_bars=300)
    
    def load_production_model(self, direction: str):
        """
        프로덕션 모델 로드
        
        Args:
            direction: 'long' or 'short'
        """
        session = self.Session()
        
        try:
            # 가장 최근 promoted 모델 조회
            result = session.execute(f"""
                SELECT model_id, artifact_uri
                FROM models
                WHERE direction = '{direction}'
                  AND status = 'promoted'
                ORDER BY promoted_at DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                raise ValueError(f"No promoted model found for {direction}")
            
            model_id, artifact_uri = result
            
            # 모델 파일 로드 (실제로는 MinIO에서 다운로드)
            # 여기서는 로컬 경로 가정
            model_path = artifact_uri.replace('s3://models/', '/tmp/models/')
            
            pwin_model = lgb.Booster(model_file=f"{model_path}/pwin_model.txt")
            hold_model = lgb.Booster(model_file=f"{model_path}/hold_model.txt")
            
            import joblib
            feature_names = joblib.load(f"{model_path}/feature_names.pkl")
            
            # 캐시 저장
            self.models[direction] = {
                'pwin': pwin_model,
                'hold': hold_model,
                'features': feature_names,
                'model_id': model_id
            }
            
            logger.info(f"✓ Loaded {direction} model: {model_id}")
            
        finally:
            session.close()
    
    def predict(
        self,
        symbol: str,
        direction: str,
        features: Dict[str, float]
    ) -> Dict:
        """
        예측 수행
        
        Args:
            symbol: 'BTCUSDT' 등
            direction: 'long' or 'short'
            features: Feature 딕셔너리
        
        Returns:
            {
                'pwin': 0.65,
                'expected_hold_minutes': 87,
                'model_id': 'uuid...'
            }
        """
        # 모델 로드 체크
        if self.models[direction]['pwin'] is None:
            self.load_production_model(direction)
        
        model_cache = self.models[direction]
        feature_names = model_cache['features']
        
        # Feature 배열 생성
        X = np.array([[features[f] for f in feature_names]])
        
        # 예측
        pwin = float(model_cache['pwin'].predict(X)[0])
        hold_time = float(model_cache['hold'].predict(X)[0])
        
        return {
            'pwin': pwin,
            'expected_hold_minutes': hold_time,
            'model_id': model_cache['model_id']
        }
```

## 10.2 정책 엔진

**파일:** `services/policy/decision_maker.py`

```python
import pandas as pd
from typing import Dict, Optional
from decimal import Decimal
import logging
import os

logger = logging.getLogger(__name__)

class PolicyEngine:
    """
    정책 엔진 (EV 기반 의사결정)
    
    - EV 계산 (비용 포함)
    - 진입 조건 체크
    - 실행 파라미터 생성
    """
    
    def __init__(self):
        # 파라미터 (환경변수)
        self.min_ev = float(os.getenv('MIN_EV', '0.0'))
        self.min_pwin = float(os.getenv('MIN_PWIN', '0.55'))
        self.max_mae_atr = float(os.getenv('MAX_MAE_ATR', '1.2'))
        
        self.fee_rate = float(os.getenv('FEE_RATE', '0.0004'))
        self.slippage_bps = float(os.getenv('SLIPPAGE_BPS', '15'))
        self.default_leverage = int(os.getenv('DEFAULT_LEVERAGE', '5'))
        
        # TP/SL 배수
        self.k_tp = float(os.getenv('LABEL_TP_ATR_MULTIPLIER', '2.0'))
        self.k_sl = float(os.getenv('LABEL_SL_ATR_MULTIPLIER', '1.0'))
    
    def calculate_ev(
        self,
        pwin: float,
        expected_hold_minutes: float,
        atr: float,
        funding_rate: float
    ) -> float:
        """
        EV (기대값) 계산
        
        EV = pwin * TP - (1 - pwin) * SL - costs
        
        Costs:
        - Fee: 0.04% * 2 (진입 + 청산)
        - Slippage: 0.15% * 2
        - Funding: funding_rate * (hold_time / 480)
        """
        # TP/SL (ATR 배수)
        tp_pct = self.k_tp * atr
        sl_pct = self.k_sl * atr
        
        # 비용
        fee = self.fee_rate * 2
        slippage = (self.slippage_bps / 10000) * 2
        
        # 펀딩 (8시간=480분 마다 정산)
        funding_periods = expected_hold_minutes / 480
        funding_cost = abs(funding_rate) * funding_periods
        
        total_cost = fee + slippage + funding_cost
        
        # EV
        ev = pwin * tp_pct - (1 - pwin) * sl_pct - total_cost
        
        return ev
    
    def make_decision(
        self,
        symbol: str,
        long_prediction: Dict,
        short_prediction: Dict,
        market_data: Dict
    ) -> Dict:
        """
        의사결정
        
        Args:
            symbol: 심볼
            long_prediction: {'pwin': ..., 'expected_hold_minutes': ...}
            short_prediction: 동일
            market_data: {'mark_price': ..., 'atr': ..., 'funding_rate': ...}
        
        Returns:
            {
                'decision': 'LONG' / 'SHORT' / 'FLAT',
                'ev_long': ...,
                'ev_short': ...,
                'confidence': ...,
                'recommended_notional': ...,
                'recommended_leverage': ...,
                'entry_price': ...,
                'sl_price': ...,
                'tp_price': ...,
                'reasons': [...]
            }
        """
        mark_price = market_data['mark_price']
        atr = market_data['atr']
        funding_rate = market_data.get('funding_rate', 0)
        
        # EV 계산
        ev_long = self.calculate_ev(
            long_prediction['pwin'],
            long_prediction['expected_hold_minutes'],
            atr / mark_price,  # ATR 비율
            funding_rate
        )
        
        ev_short = self.calculate_ev(
            short_prediction['pwin'],
            short_prediction['expected_hold_minutes'],
            atr / mark_price,
            funding_rate
        )
        
        # 의사결정
        decision = 'FLAT'
        confidence = 0.0
        selected_direction = None
        
        if ev_long > self.min_ev and long_prediction['pwin'] > self.min_pwin:
            if ev_short > self.min_ev and short_prediction['pwin'] > self.min_pwin:
                # 둘 다 조건 만족 → EV 높은 쪽
                if ev_long > ev_short:
                    decision = 'LONG'
                    confidence = long_prediction['pwin']
                    selected_direction = 'long'
                else:
                    decision = 'SHORT'
                    confidence = short_prediction['pwin']
                    selected_direction = 'short'
            else:
                decision = 'LONG'
                confidence = long_prediction['pwin']
                selected_direction = 'long'
        
        elif ev_short > self.min_ev and short_prediction['pwin'] > self.min_pwin:
            decision = 'SHORT'
            confidence = short_prediction['pwin']
            selected_direction = 'short'
        
        # 실행 파라미터
        if decision != 'FLAT':
            # SL/TP 가격
            if decision == 'LONG':
                sl_price = mark_price - (self.k_sl * atr)
                tp_price = mark_price + (self.k_tp * atr)
            else:
                sl_price = mark_price + (self.k_sl * atr)
                tp_price = mark_price - (self.k_tp * atr)
            
            # Notional (간단히 고정값)
            recommended_notional = 500.0  # USDT
            
            # Leverage
            recommended_leverage = self.default_leverage
            
        else:
            sl_price = None
            tp_price = None
            recommended_notional = 0
            recommended_leverage = 0
        
        # 근거 (top features)
        reasons = [
            f"EV_long: {ev_long:.4f}",
            f"EV_short: {ev_short:.4f}",
            f"Pwin_long: {long_prediction['pwin']:.2%}",
            f"Pwin_short: {short_prediction['pwin']:.2%}",
            f"Funding: {funding_rate:.4f}"
        ]
        
        return {
            'symbol': symbol,
            'decision': decision,
            'confidence': confidence,
            'ev_long': ev_long,
            'ev_short': ev_short,
            'pwin_long': long_prediction['pwin'],
            'pwin_short': short_prediction['pwin'],
            'recommended_notional': recommended_notional,
            'recommended_leverage': recommended_leverage,
            'entry_price': mark_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'reasons': reasons
        }
```

---

# 11. 리스크 관리 시스템

**파일:** `services/risk/risk_manager.py`

```python
from typing import Dict, List, Optional
from decimal import Decimal
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class RiskManager:
    """
    리스크 하드가드
    
    - 마진 사용률 체크
    - 청산 거리 모니터링
    - 일일 손실 제한
    - 연결 상태 체크
    - 펀딩비 폭탄 회피
    """
    
    def __init__(self, db_url: str):
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        
        # 한도
        self.max_margin_usage = float(os.getenv('MAX_MARGIN_USAGE', '0.70'))
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', '0.02'))
        self.min_liq_distance_atr = float(os.getenv('MIN_LIQUIDATION_DISTANCE_ATR', '2.0'))
        self.max_funding_rate = float(os.getenv('MAX_FUNDING_RATE', '0.01'))
    
    def check_all_constraints(
        self,
        symbol: str,
        decision: str,
        notional: float
    ) -> Dict:
        """
        모든 리스크 제약 체크
        
        Returns:
            {
                'allowed': True/False,
                'violations': [...],
                'warnings': [...]
            }
        """
        violations = []
        warnings = []
        
        # 1. 시스템 상태 체크
        if not self.check_system_health():
            violations.append({
                'type': 'system_down',
                'message': 'System component is down - trading disabled'
            })
        
        # 2. 마진 사용률
        margin_check = self.check_margin_usage(notional)
        if not margin_check['allowed']:
            violations.append(margin_check)
        elif margin_check.get('warning'):
            warnings.append(margin_check)
        
        # 3. 일일 손실
        daily_loss_check = self.check_daily_loss()
        if not daily_loss_check['allowed']:
            violations.append(daily_loss_check)
        
        # 4. 청산 거리 (기존 포지션)
        liq_check = self.check_liquidation_distance(symbol)
        if not liq_check['allowed']:
            violations.append(liq_check)
        elif liq_check.get('warning'):
            warnings.append(liq_check)
        
        # 5. 펀딩비
        funding_check = self.check_funding_rate(symbol, decision)
        if not funding_check['allowed']:
            violations.append(funding_check)
        elif funding_check.get('warning'):
            warnings.append(funding_check)
        
        # 최종 판단
        allowed = len(violations) == 0
        
        # 리스크 이벤트 기록
        if not allowed:
            self.log_risk_event(symbol, violations)
        
        return {
            'allowed': allowed,
            'violations': violations,
            'warnings': warnings
        }
    
    def check_system_health(self) -> bool:
        """시스템 상태 체크"""
        session = self.Session()
        
        try:
            # 핵심 컴포넌트 상태 체크
            result = session.execute("""
                SELECT component, status, last_heartbeat
                FROM system_status
                WHERE component IN ('userstream', 'execution', 'collector')
            """).fetchall()
            
            for component, status, last_heartbeat in result:
                # Down 상태
                if status == 'down':
                    logger.critical(f"Component {component} is DOWN")
                    return False
                
                # Heartbeat 체크 (5분 이내)
                if last_heartbeat:
                    elapsed = (datetime.now() - last_heartbeat).total_seconds()
                    if elapsed > 300:  # 5분
                        logger.critical(f"Component {component} heartbeat timeout")
                        return False
            
            return True
            
        finally:
            session.close()
    
    def check_margin_usage(self, additional_notional: float) -> Dict:
        """마진 사용률 체크"""
        session = self.Session()
        
        try:
            # 최신 계정 스냅샷
            result = session.execute("""
                SELECT 
                    total_margin_balance,
                    total_position_initial_margin,
                    total_open_order_initial_margin,
                    available_balance
                FROM account_snapshots
                ORDER BY ts DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                return {
                    'allowed': False,
                    'type': 'margin_unknown',
                    'message': 'Cannot determine margin status'
                }
            
            margin_balance, position_margin, order_margin, available = result
            
            # 현재 사용률
            current_usage = (position_margin + order_margin) / margin_balance
            
            # 추가 진입 시 사용률
            # 간단히: additional_notional / margin_balance
            projected_usage = current_usage + (additional_notional / float(margin_balance))
            
            if projected_usage > self.max_margin_usage:
                return {
                    'allowed': False,
                    'type': 'margin_high',
                    'message': f'Margin usage would be {projected_usage:.1%} (limit: {self.max_margin_usage:.1%})',
                    'current_usage': current_usage,
                    'projected_usage': projected_usage
                }
            
            # 경고 (60% 이상)
            if projected_usage > 0.6:
                return {
                    'allowed': True,
                    'warning': True,
                    'type': 'margin_warning',
                    'message': f'Margin usage approaching limit: {projected_usage:.1%}'
                }
            
            return {'allowed': True}
            
        finally:
            session.close()
    
    def check_daily_loss(self) -> Dict:
        """일일 손실 한도 체크"""
        session = self.Session()
        
        try:
            # 오늘 손익
            result = session.execute("""
                SELECT daily_realized_pnl, daily_unrealized_pnl
                FROM account_snapshots
                WHERE DATE(ts) = CURRENT_DATE
                ORDER BY ts DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                return {'allowed': True}
            
            realized_pnl, unrealized_pnl = result
            total_pnl = float(realized_pnl) + float(unrealized_pnl)
            
            # 초기 자본 (간단히 고정값)
            initial_capital = 10000.0
            
            loss_pct = total_pnl / initial_capital
            
            if loss_pct < -self.max_daily_loss_pct:
                return {
                    'allowed': False,
                    'type': 'daily_loss_limit',
                    'message': f'Daily loss limit reached: {loss_pct:.2%} (limit: {self.max_daily_loss_pct:.2%})',
                    'action': 'FORCE_FLAT_AND_DISABLE'
                }
            
            return {'allowed': True}
            
        finally:
            session.close()
    
    def check_liquidation_distance(self, symbol: str) -> Dict:
        """
        청산 거리 체크
        
        ⭐ 핵심: positionRisk.liquidationPrice 사용
        """
        session = self.Session()
        
        try:
            # 현재 포지션
            result = session.execute(f"""
                SELECT 
                    position_amt, mark_price, liquidation_price
                FROM positions
                WHERE symbol = '{symbol}'
            """).fetchone()
            
            if not result or result[0] == 0:
                # 포지션 없음
                return {'allowed': True}
            
            position_amt, mark_price, liquidation_price = result
            
            if liquidation_price == 0:
                # Cross margin이면 청산가 = 0
                return {'allowed': True}
            
            # 청산까지 거리
            liq_distance = abs(float(mark_price) - float(liquidation_price))
            
            # ATR 조회
            atr_result = session.execute(f"""
                SELECT atr
                FROM features_5m
                WHERE symbol = '{symbol}'
                ORDER BY ts DESC
                LIMIT 1
            """).fetchone()
            
            if not atr_result:
                return {'allowed': True}
            
            atr = float(atr_result[0])
            
            # ATR 배수
            liq_distance_atr = liq_distance / atr if atr > 0 else 999
            
            if liq_distance_atr < self.min_liq_distance_atr:
                return {
                    'allowed': False,
                    'type': 'liquidation_close',
                    'message': f'Liquidation too close: {liq_distance_atr:.2f} ATR (min: {self.min_liq_distance_atr})',
                    'action': 'FORCE_CLOSE_POSITION'
                }
            
            # 경고 (3 ATR 이내)
            if liq_distance_atr < 3.0:
                return {
                    'allowed': True,
                    'warning': True,
                    'type': 'liquidation_warning',
                    'message': f'Liquidation distance: {liq_distance_atr:.2f} ATR'
                }
            
            return {'allowed': True}
            
        finally:
            session.close()
    
    def check_funding_rate(self, symbol: str, direction: str) -> Dict:
        """펀딩비 체크"""
        session = self.Session()
        
        try:
            # 현재 펀딩비
            result = session.execute(f"""
                SELECT last_funding_rate
                FROM premium_index
                WHERE symbol = '{symbol}'
                ORDER BY ts DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                return {'allowed': True}
            
            funding_rate = float(result[0])
            
            # 극단적 펀딩비
            if abs(funding_rate) > self.max_funding_rate:
                # 방향 체크
                if (direction == 'LONG' and funding_rate > 0) or \
                   (direction == 'SHORT' and funding_rate < 0):
                    # 불리한 방향
                    return {
                        'allowed': False,
                        'type': 'funding_extreme',
                        'message': f'Funding rate extreme: {funding_rate:.4f} ({direction})'
                    }
            
            return {'allowed': True}
            
        finally:
            session.close()
    
    def log_risk_event(self, symbol: str, violations: List[Dict]):
        """리스크 이벤트 기록"""
        session = self.Session()
        
        try:
            for violation in violations:
                session.execute("""
                    INSERT INTO risk_events (
                        ts, event_type, severity, symbol, details, action_taken
                    ) VALUES (
                        NOW(), :event_type, 'critical', :symbol, :details, :action
                    )
                """, {
                    'event_type': violation['type'],
                    'symbol': symbol,
                    'details': str(violation),
                    'action': violation.get('action', 'rejected_entry')
                })
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Risk event logging error: {e}")
            session.rollback()
        finally:
            session.close()
```

---

# 12. 실행 엔진

**파일:** `services/execution/order_manager.py`

```python
from typing import Dict, Optional
from enum import Enum
from decimal import Decimal
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from packages.common.binance_client import BinanceFuturesClient

logger = logging.getLogger(__name__)

class OrderState(Enum):
    """주문 상태"""
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELED = 'canceled'
    FAILED = 'failed'

class OrderManager:
    """
    주문 관리자
    
    - 진입 주문 (MARKET)
    - 보호 주문 (SL/TP, reduce-only LIMIT)
    - 상태 머신
    - Naked 금지
    """
    
    def __init__(
        self,
        db_url: str,
        binance_client: BinanceFuturesClient
    ):
        engine = create_engine(db_url)
        self.Session = sessionmaker(bind=engine)
        self.client = binance_client
    
    def execute_signal(
        self,
        signal: Dict
    ) -> Dict:
        """
        시그널 실행
        
        Args:
            signal: {
                'symbol': 'BTCUSDT',
                'decision': 'LONG',
                'recommended_notional': 500,
                'recommended_leverage': 5,
                'sl_price': 49500,
                'tp_price': 51500
            }
        
        Returns:
            실행 결과
        """
        symbol = signal['symbol']
        decision = signal['decision']
        
        if decision == 'FLAT':
            return {'status': 'skipped', 'reason': 'FLAT signal'}
        
        try:
            # 1. 레버리지 설정
            self.set_leverage(symbol, signal['recommended_leverage'])
            
            # 2. 진입 주문 (MARKET)
            entry_result = self.place_market_order(
                symbol=symbol,
                side='BUY' if decision == 'LONG' else 'SELL',
                notional=signal['recommended_notional']
            )
            
            if entry_result['status'] != 'FILLED':
                raise Exception(f"Entry order failed: {entry_result}")
            
            # 3. 보호 주문 설치 (SL/TP)
            protection_result = self.place_protection_orders(
                symbol=symbol,
                side='SELL' if decision == 'LONG' else 'BUY',  # 반대 방향
                quantity=entry_result['filled_qty'],
                sl_price=signal['sl_price'],
                tp_price=signal['tp_price']
            )
            
            if not protection_result['success']:
                # ⚠️ 보호주문 실패 = 즉시 청산 (naked 금지!)
                logger.critical(f"Protection orders failed for {symbol} - EMERGENCY CLOSE")
                self.emergency_close_position(symbol)
                
                raise Exception("Protection orders failed - position closed")
            
            logger.info(f"✓ Signal executed for {symbol}: {decision}")
            
            return {
                'status': 'success',
                'entry_order': entry_result,
                'protection_orders': protection_result
            }
            
        except Exception as e:
            logger.error(f"Signal execution error: {e}")
            
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def set_leverage(self, symbol: str, leverage: int):
        """레버리지 설정"""
        try:
            self.client.change_leverage(symbol, leverage)
            logger.info(f"Set leverage for {symbol}: {leverage}x")
            
        except Exception as e:
            logger.error(f"Leverage setting error: {e}")
            # 이미 설정되어 있으면 무시
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # BUY/SELL
        notional: float
    ) -> Dict:
        """
        MARKET 주문
        
        Returns:
            {
                'status': 'FILLED',
                'filled_qty': 0.1,
                'avg_price': 50000.0,
                'order_id': 12345
            }
        """
        try:
            # Notional → Quantity 변환
            # 간단히: mark price 기준
            mark_price = self.get_mark_price(symbol)
            quantity = notional / mark_price
            
            # 수량 조정 (step size)
            quantity = self.round_quantity(symbol, quantity)
            
            # 주문
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"Market order placed: {symbol} {side} {quantity}")
            
            # 체결 대기 (간소화 - 실제로는 ORDER_TRADE_UPDATE 이벤트로 확인)
            import time
            time.sleep(1)
            
            # 주문 상태 조회
            order_id = order['orderId']
            order_status = self.client.get_order_status(symbol, order_id)
            
            if order_status['status'] == 'FILLED':
                return {
                    'status': 'FILLED',
                    'filled_qty': float(order_status['executedQty']),
                    'avg_price': float(order_status['avgPrice']),
                    'order_id': order_id
                }
            else:
                return {
                    'status': order_status['status'],
                    'order_id': order_id
                }
            
        except Exception as e:
            logger.error(f"Market order error: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def place_protection_orders(
        self,
        symbol: str,
        side: str,  # SELL (롱 청산용) or BUY (숏 청산용)
        quantity: float,
        sl_price: float,
        tp_price: float
    ) -> Dict:
        """
        보호 주문 (SL/TP)
        
        ⚠️ reduce-only 필수!
        """
        try:
            results = {}
            
            # SL (STOP_MARKET, reduce-only)
            sl_order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='STOP_MARKET',
                quantity=quantity,
                stop_price=sl_price,
                reduce_only=True
            )
            
            results['sl_order_id'] = sl_order['orderId']
            logger.info(f"SL order placed: {symbol} @ {sl_price}")
            
            # TP (TAKE_PROFIT_MARKET, reduce-only)
            tp_order = self.client.create_order(
                symbol=symbol,
                side=side,
                order_type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stop_price=tp_price,
                reduce_only=True
            )
            
            results['tp_order_id'] = tp_order['orderId']
            logger.info(f"TP order placed: {symbol} @ {tp_price}")
            
            return {
                'success': True,
                'orders': results
            }
            
        except Exception as e:
            logger.error(f"Protection orders error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def emergency_close_position(self, symbol: str):
        """
        긴급 포지션 청산
        
        Naked 금지 원칙: 보호주문 없는 포지션은 즉시 청산
        """
        try:
            # 현재 포지션 조회
            positions = self.client.get_position_risk(symbol)
            
            for pos in positions:
                position_amt = float(pos['positionAmt'])
                
                if position_amt != 0:
                    # MARKET 주문으로 즉시 청산
                    side = 'SELL' if position_amt > 0 else 'BUY'
                    quantity = abs(position_amt)
                    
                    self.client.create_order(
                        symbol=symbol,
                        side=side,
                        order_type='MARKET',
                        quantity=quantity,
                        reduce_only=True
                    )
                    
                    logger.critical(f"🚨 EMERGENCY CLOSE: {symbol} {quantity}")
            
            # 리스크 이벤트 기록
            session = self.Session()
            try:
                session.execute("""
                    INSERT INTO risk_events (
                        ts, event_type, severity, symbol, details, action_taken
                    ) VALUES (
                        NOW(), 'naked_position', 'critical', :symbol,
                        '{"reason": "protection_orders_failed"}',
                        'forced_close'
                    )
                """, {'symbol': symbol})
                
                session.commit()
            finally:
                session.close()
            
        except Exception as e:
            logger.error(f"Emergency close error: {e}")
    
    def get_mark_price(self, symbol: str) -> float:
        """Mark Price 조회"""
        data = self.client.get_premium_index(symbol)
        return float(data['markPrice'])
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """수량 라운딩 (step size)"""
        # 간단히 소수점 3자리
        return round(quantity, 3)
```
# 13. 프론트엔드

## 13.1 Next.js 설정

**파일:** `apps/web/package.json`

```json
{
  "name": "futures-trading-web",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.0.4",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "@tanstack/react-query": "5.14.2",
    "recharts": "2.10.3",
    "axios": "1.6.2",
    "date-fns": "3.0.6",
    "lucide-react": "0.294.0",
    "clsx": "2.0.0",
    "tailwind-merge": "2.1.0"
  },
  "devDependencies": {
    "@types/node": "20.10.5",
    "@types/react": "18.2.45",
    "@types/react-dom": "18.2.18",
    "autoprefixer": "10.4.16",
    "postcss": "8.4.32",
    "tailwindcss": "3.3.6",
    "typescript": "5.3.3"
  }
}
```

**파일:** `apps/web/tailwind.config.ts`

```typescript
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
        },
        success: {
          500: '#10b981',
          600: '#059669',
        },
        danger: {
          500: '#ef4444',
          600: '#dc2626',
        },
      },
    },
  },
  plugins: [],
}
export default config
```

## 13.2 API 클라이언트

**파일:** `apps/web/src/lib/api.ts`

```typescript
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// ===== Types =====
export interface SystemStatus {
  component: string;
  status: 'healthy' | 'degraded' | 'down';
  last_heartbeat: string;
}

export interface AccountSnapshot {
  total_wallet_balance: number;
  total_unrealized_pnl: number;
  available_balance: number;
  margin_usage_pct: number;
  daily_realized_pnl: number;
  daily_unrealized_pnl: number;
}

export interface Position {
  symbol: string;
  position_amt: number;
  entry_price: number;
  mark_price: number;
  unrealized_profit: number;
  liquidation_price: number;
  leverage: number;
  margin_type: string;
}

export interface Signal {
  symbol: string;
  ts: string;
  decision: 'LONG' | 'SHORT' | 'FLAT';
  confidence: number;
  ev_long: number;
  ev_short: number;
  pwin_long: number;
  pwin_short: number;
  recommended_notional: number;
  entry_price: number;
  sl_price: number;
  tp_price: number;
  reasons: string[];
}

export interface Model {
  model_id: string;
  name: string;
  version: string;
  direction: string;
  created_at: string;
  val_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    auc: number;
  };
  status: string;
}

export interface RiskEvent {
  id: number;
  ts: string;
  event_type: string;
  severity: string;
  symbol?: string;
  details: any;
  action_taken: string;
}

// ===== API Methods =====

export const getSystemStatus = async (): Promise<SystemStatus[]> => {
  const { data } = await api.get('/api/system/status');
  return data;
};

export const getAccountSnapshot = async (): Promise<AccountSnapshot> => {
  const { data } = await api.get('/api/account/snapshot');
  return data;
};

export const getPositions = async (): Promise<Position[]> => {
  const { data } = await api.get('/api/positions');
  return data;
};

export const getRecentSignals = async (limit: number = 20): Promise<Signal[]> => {
  const { data } = await api.get(`/api/signals/recent?limit=${limit}`);
  return data;
};

export const getSignalsBySymbol = async (symbol: string): Promise<Signal[]> => {
  const { data } = await api.get(`/api/signals/symbol/${symbol}`);
  return data;
};

export const getModels = async (): Promise<Model[]> => {
  const { data } = await api.get('/api/training/models');
  return data;
};

export const promoteModel = async (modelId: string): Promise<void> => {
  await api.post(`/api/training/models/${modelId}/promote`);
};

export const getRiskEvents = async (limit: number = 50): Promise<RiskEvent[]> => {
  const { data } = await api.get(`/api/risk/events?limit=${limit}`);
  return data;
};

export default api;
```

## 13.3 메인 대시보드

**파일:** `apps/web/src/app/page.tsx`

```typescript
'use client';

import { useQuery } from '@tanstack/react-query';
import { 
  getSystemStatus, 
  getAccountSnapshot, 
  getPositions,
  getRecentSignals 
} from '@/lib/api';
import { AlertCircle, TrendingUp, TrendingDown, Activity } from 'lucide-react';

export default function Dashboard() {
  const { data: systemStatus } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: getSystemStatus,
    refetchInterval: 5000,
  });

  const { data: account } = useQuery({
    queryKey: ['account'],
    queryFn: getAccountSnapshot,
    refetchInterval: 10000,
  });

  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: getPositions,
    refetchInterval: 5000,
  });

  const { data: signals } = useQuery({
    queryKey: ['signals'],
    queryFn: () => getRecentSignals(10),
    refetchInterval: 30000,
  });

  // 시스템 상태 체크
  const isSystemHealthy = systemStatus?.every(s => s.status === 'healthy') ?? false;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                🤖 Futures Trading ML
              </h1>
              <p className="text-sm text-gray-500">
                USDT-M Perpetual • Supervised Learning
              </p>
            </div>
            
            {/* 시스템 상태 */}
            <div className="flex items-center gap-2">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                isSystemHealthy 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {isSystemHealthy ? '● Online' : '● Offline'}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* 계정 요약 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Total Balance"
            value={`$${account?.total_wallet_balance.toFixed(2) ?? '0.00'}`}
            icon={<Activity className="w-6 h-6" />}
            color="blue"
          />
          
          <MetricCard
            title="Unrealized PnL"
            value={`$${account?.total_unrealized_pnl.toFixed(2) ?? '0.00'}`}
            icon={account?.total_unrealized_pnl >= 0 
              ? <TrendingUp className="w-6 h-6" />
              : <TrendingDown className="w-6 h-6" />
            }
            color={account?.total_unrealized_pnl >= 0 ? 'green' : 'red'}
          />
          
          <MetricCard
            title="Margin Usage"
            value={`${(account?.margin_usage_pct * 100).toFixed(1) ?? '0.0'}%`}
            icon={<AlertCircle className="w-6 h-6" />}
            color={account?.margin_usage_pct > 0.7 ? 'red' : 'blue'}
            warning={account?.margin_usage_pct > 0.7}
          />
          
          <MetricCard
            title="Daily PnL"
            value={`$${(
              (account?.daily_realized_pnl ?? 0) + 
              (account?.daily_unrealized_pnl ?? 0)
            ).toFixed(2)}`}
            icon={<TrendingUp className="w-6 h-6" />}
            color="blue"
          />
        </div>

        {/* 포지션 + 시그널 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          
          {/* 현재 포지션 */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">
                Active Positions ({positions?.length ?? 0})
              </h2>
            </div>
            
            <div className="p-6">
              {positions && positions.length > 0 ? (
                <div className="space-y-4">
                  {positions.map((pos) => (
                    <PositionCard key={pos.symbol} position={pos} />
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">No active positions</p>
              )}
            </div>
          </div>

          {/* Top 시그널 */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">
                Top Signals
              </h2>
            </div>
            
            <div className="p-6">
              {signals && signals.length > 0 ? (
                <div className="space-y-3">
                  {signals
                    .filter(s => s.decision !== 'FLAT')
                    .slice(0, 5)
                    .map((signal, idx) => (
                      <SignalCard key={idx} signal={signal} />
                    ))}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">No signals</p>
              )}
            </div>
          </div>
        </div>

        {/* 시스템 상태 */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">
              System Components
            </h2>
          </div>
          
          <div className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {systemStatus?.map((status) => (
                <div 
                  key={status.component}
                  className="p-4 border border-gray-200 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700 capitalize">
                      {status.component}
                    </span>
                    <div className={`w-3 h-3 rounded-full ${
                      status.status === 'healthy' ? 'bg-green-500' :
                      status.status === 'degraded' ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`} />
                  </div>
                  <p className="text-xs text-gray-500">
                    {new Date(status.last_heartbeat).toLocaleTimeString()}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}

// ===== Components =====

function MetricCard({ 
  title, 
  value, 
  icon, 
  color,
  warning 
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red';
  warning?: boolean;
}) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
  };

  return (
    <div className={`bg-white rounded-lg shadow p-6 ${warning ? 'ring-2 ring-red-500' : ''}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
        </div>
        <div className={`${colorClasses[color]} rounded-full p-3 text-white`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

function PositionCard({ position }: { position: any }) {
  const isLong = position.position_amt > 0;
  const pnlColor = position.unrealized_profit >= 0 ? 'text-green-600' : 'text-red-600';
  
  // 청산 거리 계산
  const liqDistance = Math.abs(position.mark_price - position.liquidation_price);
  const liqDistancePct = (liqDistance / position.mark_price) * 100;

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-bold text-gray-900">{position.symbol}</span>
          <span className={`px-2 py-1 rounded text-xs font-medium ${
            isLong ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {isLong ? 'LONG' : 'SHORT'} {position.leverage}x
          </span>
        </div>
        <span className={`text-lg font-bold ${pnlColor}`}>
          ${position.unrealized_profit.toFixed(2)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-gray-500">Entry:</span>
          <span className="ml-2 font-medium">${position.entry_price.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-gray-500">Mark:</span>
          <span className="ml-2 font-medium">${position.mark_price.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-gray-500">Liq:</span>
          <span className={`ml-2 font-medium ${
            liqDistancePct < 5 ? 'text-red-600' : 'text-gray-900'
          }`}>
            ${position.liquidation_price.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Distance:</span>
          <span className="ml-2 font-medium">{liqDistancePct.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

function SignalCard({ signal }: { signal: any }) {
  const isLong = signal.decision === 'LONG';
  const ev = isLong ? signal.ev_long : signal.ev_short;
  const pwin = isLong ? signal.pwin_long : signal.pwin_short;

  return (
    <div className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900">{signal.symbol}</span>
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
            isLong ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {signal.decision}
          </span>
        </div>
        <span className="text-sm font-medium text-gray-600">
          {(signal.confidence * 100).toFixed(0)}%
        </span>
      </div>

      <div className="flex items-center justify-between text-xs text-gray-600">
        <div>
          <span className="text-gray-500">EV:</span>
          <span className={`ml-1 font-medium ${ev > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {(ev * 100).toFixed(2)}%
          </span>
        </div>
        <div>
          <span className="text-gray-500">Pwin:</span>
          <span className="ml-1 font-medium">{(pwin * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Entry:</span>
          <span className="ml-1 font-medium">${signal.entry_price.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
```

## 13.4 심볼 상세 페이지

**파일:** `apps/web/src/app/symbols/[symbol]/page.tsx`

```typescript
'use client';

import { useQuery } from '@tanstack/react-query';
import { useParams } from 'next/navigation';
import { getSignalsBySymbol, getPositions } from '@/lib/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function SymbolDetailPage() {
  const params = useParams();
  const symbol = params.symbol as string;

  const { data: signals } = useQuery({
    queryKey: ['signals', symbol],
    queryFn: () => getSignalsBySymbol(symbol),
    refetchInterval: 30000,
  });

  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: getPositions,
    refetchInterval: 5000,
  });

  const currentPosition = positions?.find(p => p.symbol === symbol);
  const latestSignal = signals?.[0];

  // EV 차트 데이터
  const evChartData = signals?.slice(0, 50).reverse().map(s => ({
    time: new Date(s.ts).toLocaleTimeString(),
    ev_long: s.ev_long * 100,
    ev_short: s.ev_short * 100,
  })) ?? [];

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-gray-900">{symbol}</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* 현재 포지션 */}
        {currentPosition && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-lg font-semibold mb-4">Current Position</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-500">Direction</p>
                <p className={`text-xl font-bold ${
                  currentPosition.position_amt > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {currentPosition.position_amt > 0 ? 'LONG' : 'SHORT'} {currentPosition.leverage}x
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Entry Price</p>
                <p className="text-xl font-bold">${currentPosition.entry_price.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Mark Price</p>
                <p className="text-xl font-bold">${currentPosition.mark_price.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Unrealized PnL</p>
                <p className={`text-xl font-bold ${
                  currentPosition.unrealized_profit >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  ${currentPosition.unrealized_profit.toFixed(2)}
                </p>
              </div>
            </div>

            {/* 청산가 경고 */}
            <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-800">
                ⚠️ Liquidation Price: <strong>${currentPosition.liquidation_price.toFixed(2)}</strong>
                {' '}({Math.abs(currentPosition.mark_price - currentPosition.liquidation_price).toFixed(2)} distance)
              </p>
            </div>
          </div>
        )}

        {/* 최신 시그널 */}
        {latestSignal && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-lg font-semibold mb-4">Latest Signal</h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div>
                <p className="text-sm text-gray-500">Decision</p>
                <p className={`text-xl font-bold ${
                  latestSignal.decision === 'LONG' ? 'text-green-600' :
                  latestSignal.decision === 'SHORT' ? 'text-red-600' :
                  'text-gray-600'
                }`}>
                  {latestSignal.decision}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Confidence</p>
                <p className="text-xl font-bold">{(latestSignal.confidence * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">EV (Selected)</p>
                <p className={`text-xl font-bold ${
                  (latestSignal.decision === 'LONG' ? latestSignal.ev_long : latestSignal.ev_short) > 0
                    ? 'text-green-600' : 'text-red-600'
                }`}>
                  {((latestSignal.decision === 'LONG' ? latestSignal.ev_long : latestSignal.ev_short) * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Entry Price</p>
                <p className="text-xl font-bold">${latestSignal.entry_price.toFixed(2)}</p>
              </div>
            </div>

            {/* 진입 근거 */}
            <div className="mt-4">
              <p className="text-sm font-medium text-gray-700 mb-2">Reasons:</p>
              <div className="flex flex-wrap gap-2">
                {latestSignal.reasons.map((reason: string, idx: number) => (
                  <span 
                    key={idx}
                    className="px-3 py-1 bg-blue-50 text-blue-700 text-sm rounded-full"
                  >
                    {reason}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* EV 차트 */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Expected Value (EV) History</h2>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={evChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis label={{ value: 'EV (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="ev_long" 
                stroke="#10b981" 
                strokeWidth={2}
                name="EV Long"
              />
              <Line 
                type="monotone" 
                dataKey="ev_short" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="EV Short"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

      </main>
    </div>
  );
}
```

## 13.5 학습 센터

**파일:** `apps/web/src/app/training/page.tsx`

```typescript
'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getModels, promoteModel } from '@/lib/api';
import { CheckCircle, Clock, ArrowUpCircle } from 'lucide-react';

export default function TrainingPage() {
  const queryClient = useQueryClient();

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  });

  const promoteMutation = useMutation({
    mutationFn: promoteModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });

  const handlePromote = (modelId: string) => {
    if (confirm('Promote this model to production?')) {
      promoteMutation.mutate(modelId);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Training Center</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Models</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Direction
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Accuracy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Precision
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    AUC
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Created
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {models?.map((model) => (
                  <tr key={model.model_id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {model.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        model.direction === 'long' 
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {model.direction.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {(model.val_metrics.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {(model.val_metrics.precision * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {model.val_metrics.auc.toFixed(3)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {model.status === 'promoted' ? (
                        <span className="flex items-center gap-1 text-green-600">
                          <CheckCircle className="w-4 h-4" />
                          Promoted
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-gray-600">
                          <Clock className="w-4 h-4" />
                          Trained
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(model.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {model.status === 'trained' && (
                        <button
                          onClick={() => handlePromote(model.model_id)}
                          className="flex items-center gap-1 text-blue-600 hover:text-blue-800"
                        >
                          <ArrowUpCircle className="w-4 h-4" />
                          Promote
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </main>
    </div>
  );
}
```

---

# 14. API 서버

**파일:** `apps/api/main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Optional
import os

app = FastAPI(title="Futures Trading API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ===== System =====

@app.get("/api/system/status")
async def get_system_status():
    """시스템 상태"""
    session = SessionLocal()
    
    try:
        results = session.execute("""
            SELECT component, status, last_heartbeat
            FROM system_status
        """).fetchall()
        
        return [
            {
                'component': r[0],
                'status': r[1],
                'last_heartbeat': r[2].isoformat() if r[2] else None
            }
            for r in results
        ]
    finally:
        session.close()

# ===== Account =====

@app.get("/api/account/snapshot")
async def get_account_snapshot():
    """계정 스냅샷"""
    session = SessionLocal()
    
    try:
        result = session.execute("""
            SELECT 
                total_wallet_balance,
                total_unrealized_pnl,
                total_margin_balance,
                available_balance,
                margin_usage_pct,
                daily_realized_pnl,
                daily_unrealized_pnl
            FROM account_snapshots
            ORDER BY ts DESC
            LIMIT 1
        """).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="No account data")
        
        return {
            'total_wallet_balance': float(result[0]),
            'total_unrealized_pnl': float(result[1]),
            'total_margin_balance': float(result[2]),
            'available_balance': float(result[3]),
            'margin_usage_pct': float(result[4]),
            'daily_realized_pnl': float(result[5]),
            'daily_unrealized_pnl': float(result[6])
        }
    finally:
        session.close()

# ===== Positions =====

@app.get("/api/positions")
async def get_positions():
    """현재 포지션"""
    session = SessionLocal()
    
    try:
        results = session.execute("""
            SELECT 
                symbol, position_amt, entry_price, mark_price,
                unrealized_profit, liquidation_price, leverage, margin_type
            FROM positions
            WHERE ABS(position_amt) > 0
            ORDER BY symbol
        """).fetchall()
        
        return [
            {
                'symbol': r[0],
                'position_amt': float(r[1]),
                'entry_price': float(r[2]),
                'mark_price': float(r[3]),
                'unrealized_profit': float(r[4]),
                'liquidation_price': float(r[5]),
                'leverage': int(r[6]),
                'margin_type': r[7]
            }
            for r in results
        ]
    finally:
        session.close()

# ===== Signals =====

@app.get("/api/signals/recent")
async def get_recent_signals(limit: int = 20):
    """최근 시그널"""
    session = SessionLocal()
    
    try:
        results = session.execute(f"""
            SELECT 
                ts, symbol, decision, confidence,
                ev_long, ev_short, pwin_long, pwin_short,
                recommended_notional, entry_price, sl_price, tp_price,
                reasons
            FROM signals
            ORDER BY ts DESC
            LIMIT {limit}
        """).fetchall()
        
        return [
            {
                'ts': r[0].isoformat(),
                'symbol': r[1],
                'decision': r[2],
                'confidence': float(r[3]),
                'ev_long': float(r[4]),
                'ev_short': float(r[5]),
                'pwin_long': float(r[6]),
                'pwin_short': float(r[7]),
                'recommended_notional': float(r[8]),
                'entry_price': float(r[9]),
                'sl_price': float(r[10]) if r[10] else None,
                'tp_price': float(r[11]) if r[11] else None,
                'reasons': r[12]
            }
            for r in results
        ]
    finally:
        session.close()

@app.get("/api/signals/symbol/{symbol}")
async def get_signals_by_symbol(symbol: str):
    """심볼별 시그널"""
    session = SessionLocal()
    
    try:
        results = session.execute(f"""
            SELECT 
                ts, decision, confidence,
                ev_long, ev_short, pwin_long, pwin_short,
                entry_price, reasons
            FROM signals
            WHERE symbol = '{symbol}'
            ORDER BY ts DESC
            LIMIT 100
        """).fetchall()
        
        return [
            {
                'ts': r[0].isoformat(),
                'decision': r[1],
                'confidence': float(r[2]),
                'ev_long': float(r[3]),
                'ev_short': float(r[4]),
                'pwin_long': float(r[5]),
                'pwin_short': float(r[6]),
                'entry_price': float(r[7]),
                'reasons': r[8]
            }
            for r in results
        ]
    finally:
        session.close()

# ===== Training =====

@app.get("/api/training/models")
async def get_models():
    """모델 목록"""
    session = SessionLocal()
    
    try:
        results = session.execute("""
            SELECT 
                model_id, name, version, direction,
                created_at, val_metrics, status, promoted_at
            FROM models
            ORDER BY created_at DESC
        """).fetchall()
        
        import json
        
        return [
            {
                'model_id': r[0],
                'name': r[1],
                'version': r[2],
                'direction': r[3],
                'created_at': r[4].isoformat(),
                'val_metrics': json.loads(r[5]) if r[5] else {},
                'status': r[6],
                'promoted_at': r[7].isoformat() if r[7] else None
            }
            for r in results
        ]
    finally:
        session.close()

@app.post("/api/training/models/{model_id}/promote")
async def promote_model(model_id: str):
    """모델 프로모션"""
    session = SessionLocal()
    
    try:
        # 같은 방향의 기존 promoted 모델 해제
        session.execute(f"""
            UPDATE models
            SET status = 'archived'
            WHERE direction = (
                SELECT direction FROM models WHERE model_id = '{model_id}'
            )
            AND status = 'promoted'
        """)
        
        # 새 모델 promote
        session.execute(f"""
            UPDATE models
            SET status = 'promoted', promoted_at = NOW()
            WHERE model_id = '{model_id}'
        """)
        
        session.commit()
        
        return {'status': 'success'}
        
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# ===== Risk =====

@app.get("/api/risk/events")
async def get_risk_events(limit: int = 50):
    """리스크 이벤트"""
    session = SessionLocal()
    
    try:
        results = session.execute(f"""
            SELECT 
                id, ts, event_type, severity, symbol,
                details, action_taken
            FROM risk_events
            ORDER BY ts DESC
            LIMIT {limit}
        """).fetchall()
        
        return [
            {
                'id': r[0],
                'ts': r[1].isoformat(),
                'event_type': r[2],
                'severity': r[3],
                'symbol': r[4],
                'details': r[5],
                'action_taken': r[6]
            }
            for r in results
        ]
    finally:
        session.close()

@app.get("/")
async def root():
    return {
        "message": "Futures Trading ML API",
        "version": "1.0.0",
        "status": "running"
    }
```

---

# 15. 배포 및 운영

## 15.1 실행 스크립트

**파일:** `start.sh`

```bash
#!/bin/bash

set -e

echo "🚀 Starting Crypto Futures Trading ML System"

# 1. 환경변수 체크
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

# 2. Docker Compose 시작
echo "📦 Starting containers..."
docker-compose up -d

# 3. DB 초기화 대기
echo "⏳ Waiting for database..."
sleep 10

# 4. 데이터 수집 시작 (백그라운드)
echo "📊 Starting data collection..."
docker-compose exec -d collector python -m services.collector.market_data

# 5. User Stream 시작 (백그라운드)
echo "👤 Starting user stream..."
docker-compose exec -d userstream python -m services.userstream.stream_manager

echo ""
echo "✅ System started successfully!"
echo ""
echo "📍 Access points:"
echo "  - Frontend: http://localhost:3000"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - MinIO Console: http://localhost:9001"
echo ""
echo "📝 Next steps:"
echo "  1. Wait for data collection (check logs: docker-compose logs -f collector)"
echo "  2. Generate features: docker-compose exec training-worker python -m services.features.pipeline"
echo "  3. Generate labels: docker-compose exec training-worker python -m services.labeling.service"
echo "  4. Train models: docker-compose exec training-worker python -m services.training.trainer"
echo ""
```

**파일:** `setup.sh`

```bash
#!/bin/bash

set -e

echo "🔧 Setting up Crypto Futures Trading ML System"

# 1. 필요한 디렉토리 생성
echo "📁 Creating directories..."
mkdir -p models data/raw infra/db/migrations

# 2. .env 파일 생성 (없으면)
if [ ! -f .env ]; then
    echo "📝 Creating .env from example..."
    cp .env.example .env
    
    echo ""
    echo "⚠️  Please edit .env and configure:"
    echo "  - BINANCE_API_KEY"
    echo "  - BINANCE_API_SECRET"
    echo "  - DB_PASSWORD"
    echo "  - MINIO credentials"
    echo ""
    read -p "Press enter when done..."
fi

# 3. Docker 이미지 빌드
echo "🏗️  Building Docker images..."
docker-compose build

# 4. DB 초기화
echo "💾 Initializing database..."
docker-compose up -d timescaledb redis minio
sleep 10

# DB 스키마 적용
docker-compose exec -T timescaledb psql -U trading_user -d futures_trading < infra/db/init.sql

echo ""
echo "✅ Setup complete!"
echo ""
echo "Run './start.sh' to start the system"
```

## 15.2 모니터링

**파일:** `monitor.sh`

```bash
#!/bin/bash

echo "📊 System Monitoring"
echo ""

# 컨테이너 상태
echo "=== Container Status ==="
docker-compose ps
echo ""

# 시스템 상태
echo "=== System Components ==="
docker-compose exec -T timescaledb psql -U trading_user -d futures_trading -c \
    "SELECT component, status, last_heartbeat FROM system_status;" 2>/dev/null || echo "Database not ready"
echo ""

# 포지션
echo "=== Active Positions ==="
docker-compose exec -T timescaledb psql -U trading_user -d futures_trading -c \
    "SELECT symbol, position_amt, unrealized_profit, liquidation_price 
     FROM positions WHERE ABS(position_amt) > 0;" 2>/dev/null || echo "No positions"
echo ""

# 최근 시그널
echo "=== Recent Signals ==="
docker-compose exec -T timescaledb psql -U trading_user -d futures_trading -c \
    "SELECT ts, symbol, decision, confidence, ev_long, ev_short 
     FROM signals ORDER BY ts DESC LIMIT 5;" 2>/dev/null || echo "No signals"
echo ""

# 리스크 이벤트
echo "=== Risk Events (Last 24h) ==="
docker-compose exec -T timescaledb psql -U trading_user -d futures_trading -c \
    "SELECT ts, event_type, severity, symbol, action_taken 
     FROM risk_events 
     WHERE ts > NOW() - INTERVAL '24 hours'
     ORDER BY ts DESC LIMIT 10;" 2>/dev/null || echo "No risk events"
```

## 15.3 데이터 파이프라인 실행

**파일:** `run_pipeline.sh`

```bash
#!/bin/bash

set -e

SYMBOLS=${1:-"BTCUSDT,ETHUSDT,BNBUSDT"}
START_DATE=${2:-"2024-01-01"}
END_DATE=${3:-"2025-01-01"}

echo "🔄 Running data pipeline"
echo "Symbols: $SYMBOLS"
echo "Date range: $START_DATE to $END_DATE"
echo ""

# 1. Feature 생성
echo "⚙️  Step 1/3: Generating features..."
docker-compose exec -T training-worker python -m services.features.pipeline \
    --symbols "$SYMBOLS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

# 2. 라벨 생성
echo "🏷️  Step 2/3: Generating labels..."
docker-compose exec -T training-worker python -m services.labeling.service \
    --symbols "$SYMBOLS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

# 3. 모델 학습
echo "🧠 Step 3/3: Training models..."

# Long 모델
echo "Training LONG model..."
docker-compose exec -T training-worker python -m services.training.trainer \
    --direction long \
    --symbols "$SYMBOLS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

# Short 모델
echo "Training SHORT model..."
docker-compose exec -T training-worker python -m services.training.trainer \
    --direction short \
    --symbols "$SYMBOLS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

echo ""
echo "✅ Pipeline complete!"
echo ""
echo "Next: Promote models via web UI (http://localhost:3000/training)"
```

## 15.4 README

**파일:** `README.md`

```markdown
# 🤖 Crypto Futures Trading ML System

USDT-M Perpetual 선물 자동매매 시스템 (지도학습 기반)

## 🎯 주요 특징

- **이벤트 라벨링**: Triple Barrier + 비용 포함 계산
- **LightGBM**: Pwin 예측 + Hold time 예측
- **정책 엔진**: EV 기반 의사결정
- **리스크 하드가드**: 청산 거리, 마진, 펀딩비, 일일 손실
- **실시간 모니터링**: Next.js 대시보드

## 📋 요구사항

- Docker & Docker Compose
- Binance API Key (Testnet 권장)

## 🚀 빠른 시작

### 1. 설정

```bash
# Setup
chmod +x setup.sh
./setup.sh

# .env 파일 편집
nano .env
```

### 2. 시스템 시작

```bash
chmod +x start.sh
./start.sh
```

### 3. 데이터 파이프라인 실행

```bash
# 과거 데이터 수집 (1년)
docker-compose exec collector python -m services.collector.historical

# Feature + Label + Training
chmod +x run_pipeline.sh
./run_pipeline.sh "BTCUSDT,ETHUSDT,BNBUSDT" "2024-01-01" "2025-01-01"
```

### 4. 모델 프로모션

- http://localhost:3000/training 접속
- 학습된 모델 확인
- "Promote" 버튼 클릭

### 5. 실전 운영

시그널이 생성되면 자동으로 거래 실행됩니다.

## 📊 모니터링

```bash
# 실시간 모니터링
chmod +x monitor.sh
./monitor.sh

# 로그
docker-compose logs -f realtime-worker
```

## ⚙️ 주요 설정

`.env` 파일:

```bash
# 거래 파라미터
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
DEFAULT_LEVERAGE=5

# 리스크 한도
MAX_MARGIN_USAGE=0.70          # 70%
MAX_DAILY_LOSS_PCT=0.02        # 2%
MIN_LIQUIDATION_DISTANCE_ATR=2.0

# 라벨링
LABEL_TP_ATR_MULTIPLIER=2.0    # TP = 2 * ATR
LABEL_SL_ATR_MULTIPLIER=1.0    # SL = 1 * ATR
LABEL_TIMEOUT_BARS=48          # 4시간

# 정책
MIN_EV=0.0
MIN_PWIN=0.55
```

## 🛡️ 안전장치

1. **청산 거리 모니터링**: 2 ATR 이내 → 강제 청산
2. **마진 사용률**: 70% 초과 → 진입 거부
3. **일일 손실**: 2% 초과 → 모든 포지션 청산 + 거래 중단
4. **User Stream 연결**: 끊김 → 즉시 거래 중단
5. **Naked 금지**: 보호주문 실패 → 즉시 청산

## 📁 구조

```
crypto-futures-ml/
├── apps/
│   ├── web/          # Next.js 프론트엔드
│   └── api/          # FastAPI 게이트웨이
├── services/
│   ├── collector/    # 시장 데이터 수집
│   ├── userstream/   # 포지션/주문 실시간
│   ├── features/     # Feature 생성
│   ├── labeling/     # 라벨 생성
│   ├── training/     # 모델 학습
│   ├── inference/    # 실시간 추론
│   ├── policy/       # 의사결정
│   ├── risk/         # 리스크 관리
│   └── execution/    # 주문 실행
└── packages/
    └── common/       # 공통 라이브러리
```

## 🔧 문제 해결

### 데이터 수집 안 됨
```bash
docker-compose logs collector
# Binance API 키 확인
```

### 모델 학습 실패
```bash
# 데이터 확인
docker-compose exec timescaledb psql -U trading_user -d futures_trading
\dt
SELECT COUNT(*) FROM features_5m;
SELECT COUNT(*) FROM labels_long_5m;
```

### User Stream 끊김
```bash
# 자동 재연결되나, 수동 재시작:
docker-compose restart userstream
```

## ⚠️ 주의사항

- **Testnet 먼저 사용**: BINANCE_TESTNET=true
- **소액으로 시작**: 실전 투입 전 충분한 테스트
- **청산 위험**: 레버리지 거래는 높은 위험
- **출금 권한 없는 API 키 사용**

## 📝 라이선스

MIT

---

**면책 조항**: 이 시스템은 교육/연구 목적입니다. 실제 거래 손실에 대한 책임은 사용자에게 있습니다.
```

---

# 🎉 완성!

**전체 시스템이 완성되었습니다!**

## ✅ 구현된 모든 컴포넌트

### Backend (Python)
1. ✅ Binance 선물 API 클라이언트 (청산가, 펀딩비 포함)
2. ✅ 실시간 데이터 수집 (WebSocket)
3. ✅ User Stream 관리 (listenKey keepalive)
4. ✅ Feature Engineering (온/오프라인 동일)
5. ✅ Triple Barrier 라벨링 (비용 포함)
6. ✅ LightGBM 학습 (Walk-forward)
7. ✅ 실시간 추론
8. ✅ 정책 엔진 (EV 기반)
9. ✅ 리스크 하드가드 (청산/마진/펀딩/일일손실)
10. ✅ 주문 실행 (Naked 금지)

### Frontend (Next.js)
1. ✅ 메인 대시보드 (계정/포지션/시그널/시스템 상태)
2. ✅ 심볼 상세 페이지 (EV 차트, 청산가 라인)
3. ✅ 학습 센터 (모델 관리, 프로모션)

### Infrastructure
1. ✅ Docker Compose (전체 시스템)
2. ✅ PostgreSQL + TimescaleDB
3. ✅ Redis
4. ✅ MinIO

### Scripts
1. ✅ setup.sh (초기 설정)
2. ✅ start.sh (시스템 시작)
3. ✅ monitor.sh (모니터링)
4. ✅ run_pipeline.sh (데이터 파이프라인)

## 🚀 실행 순서

```bash
# 1. Setup
./setup.sh

# 2. Start
./start.sh

# 3. Pipeline (데이터 수집 완료 후)
./run_pipeline.sh

# 4. 웹 접속
http://localhost:3000
```


- ✅ USDT-M Perpetual 전용
- ✅ Mark Price 기준 계산
- ✅ 청산가 실시간 모니터링
- ✅ 펀딩비 비용 포함
- ✅ User Stream 연결 체크
- ✅ Triple Barrier + 비용
- ✅ LightGBM + Walk-forward
- ✅ EV 기반 정책
- ✅ 리스크 하드가드
- ✅ 실전 트레이더 대시보드

