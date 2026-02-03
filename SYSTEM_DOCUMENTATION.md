# TA: Binance USDT-M Supervised Trading System

> **버전**: v3
> **최종 업데이트**: 2026-02-02
> **목적**: Binance 선물 시장에서 머신러닝 기반 자동매매 시스템

---

## 1. 프로젝트 개요

바이낸스 USDT-M 무기한 선물 시장에서 작동하는 **지도학습 기반 자동매매 시스템**입니다.

### 핵심 기능
- 실시간 시장 데이터 수집 및 피처 엔지니어링 (33개 피처)
- **시장 심리 데이터**: Open Interest, Long/Short Ratio, Taker Buy/Sell Ratio
- LightGBM 기반 가격 예측 및 트레이딩 결정 (Early Stopping 적용)
- **강화학습 (RL) 하이브리드**: PPO/A2C 에이전트가 LightGBM 예측 기반 최종 결정 **(v3 추가)**
- 브라켓 주문 실행 (진입 + TP + SL)
- 리스크 관리 및 드리프트 모니터링 (서킷 브레이커)
- 백테스팅, 라벨링, 모델 학습 파이프라인
- 모니터링용 웹 UI

### 고정 유니버스 (v3)
30개 심볼 고정 운영 (금/은 포함):
```
BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, DOTUSDT, LINKUSDT,
LTCUSDT, BCHUSDT, SUIUSDT, AAVEUSDT, 1000PEPEUSDT, FILUSDT, AXSUSDT, ENAUSDT, ZKUSDT, XAUUSDT,
XAGUSDT, HYPEUSDT, ZECUSDT, BNXUSDT, ALPHAUSDT, ZORAUSDT, TRUMPUSDT, PUMPUSDT, NEARUSDT, PAXGUSDT
```

### 운영 모드
| 모드 | 설명 |
|------|------|
| `OFF` | 트레이딩 비활성화 |
| `SHADOW` | 시뮬레이션 모드 (실제 주문 없음) |
| `LIVE` | 실제 주문 실행 |

---

## 2. 프로젝트 구조

```
ta/
├── apps/                          # 프론트엔드 및 API
│   ├── api/                       # FastAPI REST/WebSocket 게이트웨이
│   │   └── main.py                # API 라우트 및 웹소켓 핸들러
│   └── web/                       # Next.js React 웹 UI
│       ├── src/app/               # Next.js app 디렉토리
│       └── src/lib/               # API 클라이언트 유틸
│
├── services/                      # 핵심 트레이딩 서비스
│   ├── collector/                 # 시장 데이터 수집
│   │   ├── market_data.py         # Binance API 클라이언트
│   │   ├── buffer.py              # 배치/버퍼링
│   │   └── universe_builder.py    # Top-N 종목 선택
│   ├── features/                  # 피처 엔지니어링
│   │   └── compute.py             # 기술적 지표 계산
│   ├── inference/                 # ML 예측
│   │   └── predictor.py           # 모델 추론 래퍼
│   ├── policy/                    # 트레이딩 결정 로직
│   │   └── decide.py              # 시그널 생성 및 필터링
│   ├── risk/                      # 리스크 관리
│   │   └── guard.py               # 포트폴리오/드리프트 체크
│   ├── execution/                 # 실제 주문 실행
│   │   ├── binance_client.py      # Binance API 래퍼
│   │   └── trader.py              # 브라켓 주문 배치
│   ├── simulator/                 # 섀도우 트레이딩 시뮬레이션
│   │   ├── fill.py                # 체결 모델 및 슬리피지
│   │   └── trade_writer.py        # 트레이드 그룹 저장
│   ├── labeling/                  # 학습 데이터 생성
│   │   ├── pipeline.py            # 라벨링 오케스트레이션
│   │   └── triple_barrier.py      # Triple-barrier 라벨링
│   ├── training/                  # 모델 학습
│   │   ├── pipeline.py            # 학습 오케스트레이션
│   │   ├── train.py               # LightGBM 학습 로직
│   │   └── splits.py              # Walk-forward 검증
│   ├── registry/                  # 모델 저장소
│   │   └── storage.py             # MinIO S3 연동
│   ├── monitoring/                # 데이터 품질/드리프트 감지
│   │   ├── drift.py               # PSI, 레이턴시, 아웃라이어
│   │   └── drift_job.py           # 드리프트 작업 스케줄러
│   ├── rl/                        # 강화학습 모듈 (v3 추가)
│   │   ├── environment.py         # Gymnasium 트레이딩 환경
│   │   ├── agent.py               # PPO/A2C 에이전트 래퍼
│   │   └── decision_store.py      # RL 결정 저장/조회
│   ├── realtime_worker.py         # 메인 실시간 루프
│   ├── batch_worker.py            # 학습 작업 프로세서
│   └── universe_worker.py         # 유니버스 갱신 워커 (고정 유니버스 동기화)
│
├── packages/                      # 공유 라이브러리
│   └── common/
│       ├── config.py              # Pydantic 설정
│       ├── db.py                  # PostgreSQL 연결
│       ├── bus.py                 # Redis pub/sub 스트림
│       ├── runtime.py             # Redis 상태 관리
│       ├── portfolio.py           # 포지션/노출도 계산
│       ├── types.py               # Pydantic DTO
│       └── symbol_map.py          # 심볼 포맷 변환
│
├── infra/migrations/              # DB 마이그레이션
│   ├── 001_init.sql               # 코어 스키마
│   ├── 002_trade_group.sql        # 트레이드 그룹 지원
│   ├── 003_training_report.sql    # 학습 리포트 저장
│   ├── 004_drift_metrics.sql      # 드리프트 메트릭 테이블
│   ├── 005_market_sentiment.sql   # OI/Long-Short Ratio 테이블 (v3 추가)
│   └── 009_rl_tables.sql          # RL 모델/결정 테이블 (v3 추가)
│
├── scripts/                       # 초기화 스크립트
├── tests/                         # 테스트 스위트
├── docker-compose.yml             # 서비스 오케스트레이션
├── Dockerfile                     # Python 컨테이너 이미지
└── requirements.txt               # Python 의존성
```

---

## 3. 기술 스택

### 백엔드
| 구분 | 기술 |
|------|------|
| 언어 | Python 3.11/3.13 |
| 프레임워크 | FastAPI 0.115.8, Uvicorn |
| ML/데이터 | LightGBM 4.5.0, scikit-learn 1.5.2, pandas 2.2.3, numpy 2.1.3 |
| 강화학습 | stable-baselines3 2.3.2, gymnasium 0.29.1, PyTorch 2.0+ **(v3 추가)** |

### 프론트엔드
| 구분 | 기술 |
|------|------|
| 언어 | TypeScript 5.5.4 |
| 프레임워크 | React 18.3.1, Next.js 14.2.5 |
| 스타일 | TailwindCSS 3.4.10 |

### 데이터베이스 & 스토리지
| 구분 | 기술 | 용도 |
|------|------|------|
| PostgreSQL 16 | 주 데이터베이스 | 캔들, 피처, 라벨, 시그널, 트레이드, 모델 |
| Redis 7 | 상태 관리 | pub/sub, 모드/상태 플래그 |
| MinIO | 오브젝트 스토리지 | 모델 아티팩트, 리포트 |

### 외부 서비스
- **Binance Futures API**: 시장 데이터, 주문 실행, 포지션 스트림
- **Binance Testnet**: 테스트넷 트레이딩 모드 (선택)

---

## 4. 진입점 (Entry Points)

### API 서버
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```
- **파일**: `apps/api/main.py`
- **포트**: 8000
- **역할**: REST/WebSocket 엔드포인트 제공

### 실시간 워커
```bash
python -m services.realtime_worker
```
- **파일**: `services/realtime_worker.py`
- **역할**: 5초마다 실행되는 메인 트레이딩 루프
  - 최신 캔들/프리미엄 인덱스 수집
  - 피처 계산 → 추론 → 정책/리스크 체크
  - 시뮬레이션(shadow) 또는 실주문(live) 실행

### 배치 워커
```bash
python -m services.batch_worker
```
- **파일**: `services/batch_worker.py`
- **역할**: 비동기 학습 작업 처리
  - 라벨링 → 학습 → 모델 등록 → 리포트 생성

### 유니버스 워커
```bash
python -m services.universe_worker
```
- **파일**: `services/universe_worker.py`
- **역할**: 주기적 유니버스 갱신 (거래량 기준 Top-N 선택)

### 웹 UI
```bash
cd apps/web && npm run dev
```
- **포트**: 3000
- **역할**: 시그널, 포지션, 모델 모니터링 대시보드

---

## 5. 핵심 모듈 상세

### 5.1 데이터 레이어 (`packages/common/`)

| 모듈 | 역할 |
|------|------|
| `config.py` | Pydantic BaseSettings로 환경변수 로드 |
| `db.py` | PostgreSQL 커넥션 풀링, `fetch_all()`, `bulk_upsert()` |
| `bus.py` | Redis Streams pub/sub (`signal_updates` 스트림) |
| `runtime.py` | Redis 키 관리 (MODE, DRIFT_STATUS 등) |
| `portfolio.py` | 포트폴리오 노출도 계산 |

### 5.2 데이터 수집 (`services/collector/`)

| 모듈 | 역할 |
|------|------|
| `market_data.py` | Binance API 호출 (아래 상세) |
| `buffer.py` | 인메모리 버퍼링, 배치 DB 쓰기 (OI/L-S 포함) |
| `universe_builder.py` | 24h 거래량 기준 Top-N 종목 선택 |

**market_data.py API 함수** (v3 확장):
| 함수 | 설명 |
|------|------|
| `fetch_klines()` | OHLCV 캔들 데이터 |
| `fetch_premium_index()` | 마크가격, 펀딩레이트 |
| `fetch_funding_rates()` | 펀딩 레이트 히스토리 |
| `fetch_open_interest()` | 현재 미결제약정 **(v3 추가)** |
| `fetch_open_interest_hist()` | 미결제약정 히스토리 **(v3 추가)** |
| `fetch_long_short_ratio()` | 글로벌 롱/숏 비율 **(v3 추가)** |
| `fetch_top_long_short_ratio()` | 탑 트레이더 롱/숏 비율 **(v3 추가)** |
| `fetch_taker_buy_sell_ratio()` | 테이커 매수/매도 비율 **(v3 추가)** |

### 5.3 피처 엔지니어링 (`services/features/compute.py`)

**스키마 버전**: v2 (33개 피처, v1은 14개)

계산되는 피처:
| 카테고리 | 피처 | 설명 |
|----------|------|------|
| **수익률 (3)** | `ret_1`, `ret_5`, `ret_15` | 1/5/15봉 수익률 |
| **변동성 (4)** | `atr`, `natr`, `bb_z`, `vol_z` | ATR, Normalized ATR **(v3)**, BB z-score, 거래량 z-score |
| **모멘텀 (8)** | `ema_dist_atr`, `adx` | EMA 거리, 14봉 ADX |
| | `rsi`, `rsi_z` | RSI (14봉), RSI z-score **(v3 추가)** |
| | `macd`, `macd_signal`, `macd_hist`, `macd_z` | MACD 지표군 **(v3 추가)** |
| **거래량 (2)** | `obv`, `obv_z` | On-Balance Volume **(v3 추가)** |
| **펀딩 (3)** | `funding_rate`, `funding_z`, `basis` | 펀딩비율 및 z-score |
| **BTC 레짐 (3)** | `btc_ret_60`, `btc_vol_60`, `btc_regime` | BTC 변동성 레짐 |
| **미결제약정 (3)** | `oi_change_1h`, `oi_change_4h`, `oi_z` | OI 변화율 **(v3 추가)** |
| **롱숏비율 (7)** | `ls_ratio`, `ls_ratio_z`, `ls_extreme` | 글로벌 롱/숏 비율 **(v3 추가)** |
| | `top_ls_ratio`, `top_ls_ratio_z` | 탑 트레이더 롱/숏 **(v3 추가)** |
| | `taker_ratio`, `taker_ratio_z` | 테이커 매수/매도 **(v3 추가)** |

**v3에서 추가된 기술적 지표 함수**:
```python
def _rsi(close, window=14) -> pd.Series      # RSI (0-100)
def _macd(close, fast=12, slow=26, signal=9) # MACD, Signal, Histogram
def _obv(close, volume) -> pd.Series         # On-Balance Volume
```

### 5.4 추론 (`services/inference/predictor.py`)

- MinIO에서 프로덕션 모델 lazy-load
- LightGBM 예측 수행
- 출력: `er_long`, `er_short`, `q05_long`, `q05_short`, `e_mae_long`, `e_mae_short`

### 5.5 정책 결정 (`services/policy/decide.py`)

```
EV = expected_return - costs
결정 = max(EV_long, EV_short, 0) 중 최대값
필터 = ev_min, q05_min, mae_max 임계값 적용
```

### 5.6 리스크 관리 (`services/risk/guard.py`)

체크 항목:
1. **마진 비율**: `used_margin / equity > max_used_margin_pct`
2. **일일 손실**: `daily_loss >= daily_loss_limit_pct * equity`
3. **포지션 수**: `open_positions >= max_positions`
4. **총 노출도**: `total_notional > max_total_notional_pct * equity`
5. **방향성 노출도**: `max(long, short) > max_directional_notional_pct * equity`
6. **유저스트림 상태**: 연결 끊김 → `USERSTREAM_DOWN`
7. **데이터 신선도**: lag > 120s → `DATA_STALE`
8. **누락률**: alert_rate 초과 → `MISSING_DATA`
9. **드리프트 상태**: PSI 초과 → `DRIFT_ALERT` / `DRIFT_BLOCK`

### 5.7 시뮬레이션 (`services/simulator/`)

| 모듈 | 역할 |
|------|------|
| `fill.py` | 슬리피지 모델, SL/TP 히트 시뮬레이션 |
| `trade_writer.py` | 시그널/주문/체결 원자적 저장 |

### 5.8 학습 파이프라인 상세

학습 파이프라인은 **데이터 수집 → 피처 계산 → 라벨링 → 학습 → 모델 등록** 5단계로 진행됩니다.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ML 학습 파이프라인 전체 흐름                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  1. 데이터   │ →  │  2. 피처    │ →  │  3. 라벨링  │ →  │  4. 학습    │   │11
│  │    수집     │    │    계산     │    │             │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        ↓                  ↓                  ↓                  ↓           │
│   candles_1m         features_1m        labels_long_1m      models         │
│   premium_index      (33개 피처)        labels_short_1m     (MinIO)        │
│   open_interest                                                            │
│   long_short_ratio                                                         │
│                                                                              │
│                              ┌─────────────┐                                │
│                              │  5. 추론    │                                │
│                              │  (실시간)   │                                │
│                              └─────────────┘                                │
│                                    ↓                                        │
│                               signals                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

#### 5.8.1 데이터 수집 (Stage 1)

**실시간 워커** (`services/realtime_worker.py`)가 5초마다 수집:

| 데이터 | 테이블 | 소스 API | 주기 |
|--------|--------|----------|------|
| OHLCV 캔들 | `candles_1m` | `/fapi/v1/klines` | 1분 |
| 마크가격/펀딩 | `premium_index` | `/fapi/v1/premiumIndex` | 실시간 |
| 미결제약정 | `open_interest` | `/futures/data/openInterestHist` | 5분 |
| 롱/숏 비율 | `long_short_ratio` | `/futures/data/globalLongShortAccountRatio` | 5분 |

**과거 데이터 백필**:
```bash
# 캔들 히스토리 (최대 1500개 = ~25시간)
python scripts/bootstrap_history.py

# OI/Long-Short 히스토리 (최대 500개 = ~41시간)
python scripts/backfill_oi_ls.py
```

---

#### 5.8.2 피처 계산 (Stage 2)

**파일**: `services/features/compute.py`
**스키마 버전**: v2 (33개 피처)

피처는 실시간으로 계산되어 `features_1m` 테이블에 저장됩니다.

| 카테고리 | 피처 | 계산 방법 | 윈도우 |
|----------|------|-----------|--------|
| **수익률 (3)** | `ret_1`, `ret_5`, `ret_15` | `close.pct_change(n)` | 1/5/15봉 |
| **변동성 (4)** | `atr` | True Range 14봉 평균 | 14 |
| | `natr` | `atr / close * 100` (정규화) | - |
| | `bb_z` | `(close - MA20) / std20` | 20 |
| | `vol_z` | 거래량 z-score | 60 |
| **모멘텀 (8)** | `ema_dist_atr` | `(close - EMA20) / atr` | 20 |
| | `adx` | DMI 기반 ADX | 14 |
| | `rsi` | RSI (0~100) | 14 |
| | `rsi_z` | RSI z-score | 60 |
| | `macd`, `macd_signal`, `macd_hist` | MACD (12,26,9) | - |
| | `macd_z` | MACD z-score | 60 |
| **거래량 (2)** | `obv` | On-Balance Volume | 누적 |
| | `obv_z` | OBV z-score | 60 |
| **펀딩 (3)** | `funding_rate` | 현재 펀딩비율 | - |
| | `funding_z` | 펀딩비율 z-score | 120 |
| | `basis` | `(last - index) / index` | - |
| **BTC 레짐 (3)** | `btc_ret_60` | BTC 60봉 수익률 | 60 |
| | `btc_vol_60` | BTC 60봉 변동성 | 60 |
| | `btc_regime` | 변동성 median 기준 0/1 | - |
| **미결제약정 (3)** | `oi_change_1h` | OI 1시간 변화율 | 60 |
| | `oi_change_4h` | OI 4시간 변화율 | 240 |
| | `oi_z` | OI z-score | 120 |
| **롱숏비율 (7)** | `ls_ratio` | 글로벌 롱/숏 비율 | - |
| | `ls_ratio_z` | 롱/숏 비율 z-score | 120 |
| | `ls_extreme` | 극단값 플래그 (>2 or <0.5) | - |
| | `top_ls_ratio` | 탑트레이더 롱/숏 비율 | - |
| | `top_ls_ratio_z` | 탑트레이더 z-score | 120 |
| | `taker_ratio` | 테이커 매수/매도 비율 | - |
| | `taker_ratio_z` | 테이커 비율 z-score | 60 |

---

#### 5.8.3 Triple-Barrier 라벨링 (Stage 3)

**파일**: `services/labeling/triple_barrier.py`, `services/labeling/pipeline.py`

Triple-Barrier 방법은 **가격이 3개 배리어 중 하나에 먼저 도달**하는 것을 기준으로 라벨을 생성합니다.

```
                    TP 배리어 (y=1)
                    ─────────────────────────────────────
                         ↑ k_tp × ATR
        진입가 ──────────●────────────────────────────────
                         ↓ k_sl × ATR
                    SL 배리어 (y=-1)
                    ─────────────────────────────────────

        ← ─────────── h_bars (시간 배리어, y=0) ─────────── →
```

**라벨링 설정** (`LabelSpec`):
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `k_tp` | 1.5 | TP 배리어 = entry + k_tp × ATR |
| `k_sl` | 1.0 | SL 배리어 = entry - k_sl × ATR |
| `h_bars` | 360 | 시간 배리어 (360봉 = 6시간) |
| `risk_mae_atr` | 3.0 | 리스크 MAE 배리어 (강제 청산) |
| `fee_rate` | 0.0004 | 테이커 수수료 (0.04%) |
| `slippage_k` | 0.15 | 슬리피지 계수 |

**라벨 출력** (`labels_long_1m`, `labels_short_1m`):
| 컬럼 | 설명 |
|------|------|
| `y` | 라벨: 1 (TP 히트), 0 (시간 만료), -1 (SL 히트) |
| `ret_net` | 순수익률 (비용 차감 후) |
| `mae` | Maximum Adverse Excursion (최대 역행) |
| `mfe` | Maximum Favorable Excursion (최대 유리) |
| `time_to_event_min` | 이벤트까지 소요 시간 (분) |
| `spec_hash` | 라벨링 설정 해시 (16자) |

**라벨링 실행**:
```bash
# 전체 유니버스 라벨링
python services/labeling/run_labeling.py

# 특정 심볼만
python -c "from services.labeling.pipeline import label_symbol; ..."
```

---

#### 5.8.4 모델 학습 (Stage 4)

**파일**: `services/training/train.py`, `services/training/pipeline.py`

**학습 타겟 (4개)**:
| 타겟 | 원본 컬럼 | Objective | 설명 |
|------|-----------|-----------|------|
| `er_long` | `ret_net` | regression | 기대 수익률 예측 |
| `q05_long` | `ret_net` | quantile (α=0.05) | 5% 분위수 (하방 리스크) |
| `e_mae_long` | `mae` | regression | 예상 MAE (최대 손실) |
| `e_hold_long` | `time_to_event_min` | regression | 예상 보유 시간 |

**LightGBM 하이퍼파라미터**:
```python
params = {
    "objective": "regression" | "quantile",
    "n_estimators": 500,           # Early Stopping으로 조기 종료
    "learning_rate": 0.05,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
# Early Stopping: 50 라운드 patience
callbacks=[lgb.early_stopping(stopping_rounds=50)]
```

**Walk-Forward 검증**:
```
┌────────────────────────────────────────────────────────┐
│     Train Set          │  Val Set   │   Test Set      │
│  (purge_bars 제외)     │            │                 │
├────────────────────────┼────────────┼─────────────────┤
│ 2024-01-01 ~ 2024-10-31│ 2024-11-01 │ 2024-12-01 ~   │
│                        │ ~ 11-30    │  2025-01-01    │
└────────────────────────┴────────────┴─────────────────┘
```

**학습 메트릭**:
| 메트릭 | 설명 |
|--------|------|
| `profit_factor` | 총이익 / 총손실 |
| `max_drawdown` | 최대 낙폭 |
| `expectancy` | 평균 수익률 |
| `tail_loss` | 하위 1% 평균 손실 |
| `turnover` | 거래 횟수 |
| `best_iteration` | Early Stopping 시점 |
| `rmse`, `mae` | 예측 오차 |
| `pinball_loss` | Quantile 예측 손실 (q05만) |

**학습 작업 실행** (Batch Worker):
```sql
-- 학습 작업 큐잉
INSERT INTO training_jobs (job_id, status, config) VALUES (
  gen_random_uuid(),
  'queued',
  '{
    "train_start": "2024-01-01",
    "train_end": "2024-12-31",
    "val_start": "2024-11-01",
    "val_end": "2025-01-01",
    "targets": ["er_long", "q05_long", "e_mae_long", "e_hold_long"],
    "feature_schema_version": 2,
    "label_config": {"k_tp": 1.5, "k_sl": 1.0, "h_bars": 360}
  }'
);
```

```bash
# Batch Worker가 queued 작업 자동 처리
python -m services.batch_worker
```

**학습 파이프라인 흐름** (`run_training_pipeline`):
```
1. LabelingConfig → LabelSpec 생성
2. run_labeling() → labels_long_1m, labels_short_1m 저장
3. features_1m + labels JOIN → 학습 데이터셋 구성
4. Walk-forward splits 생성
5. 타겟별 LightGBM 학습 (Early Stopping)
6. 메트릭 계산 (PF, MDD, Expectancy, Regime별)
7. 모델 MinIO 업로드 (models/{model_id}.pkl)
8. 리포트 JSON 저장 (reports/{model_id}.json)
9. models 테이블에 메타데이터 저장
```

---

#### 5.8.5 모델 등록 및 프로덕션 승격 (Stage 5)

**모델 저장 구조**:
```
MinIO (s3://models/)
├── models/
│   ├── {model_id}.pkl      # LightGBM 모델 (pickle)
│   └── ...
└── reports/
    ├── {model_id}.json     # 학습 리포트
    └── ...
```

**프로덕션 승격**:
```bash
# API를 통한 모델 승격
curl -X POST "http://localhost:8000/api/models/{model_id}/promote"
```

승격되면 `models.is_production = true`로 설정되고, 실시간 워커가 해당 모델을 로드합니다.

---

#### 5.8.6 실시간 추론 (Inference)

**파일**: `services/inference/predictor.py`

```python
# 추론 출력
{
    "er_long": 0.0023,    # 롱 기대수익률
    "er_short": -0.0015,  # 숏 기대수익률
    "q05_long": -0.008,   # 롱 5% 분위수 (하방 리스크)
    "q05_short": -0.012,  # 숏 5% 분위수
    "e_mae_long": 0.005,  # 롱 예상 MAE
    "e_mae_short": 0.007, # 숏 예상 MAE
}
```

**정책 결정** (`services/policy/decide.py`):
```python
EV_long = er_long - costs
EV_short = er_short - costs

# 필터 조건
if EV > ev_min and q05 > q05_min and mae < mae_max:
    signal = "LONG" if EV_long > EV_short else "SHORT"
else:
    signal = "FLAT"
```

---

#### 5.8.7 학습 최소 요구사항

| 항목 | 최소 권장값 | 설명 |
|------|-------------|------|
| **캔들 데이터** | 7일 이상 | h_bars(360) + 피처 윈도우(240) 고려 |
| **피처 데이터** | 7일 이상 | OI/LS 데이터 포함 시 더 필요 |
| **라벨 데이터** | 5일 이상 | Train/Val/Test 분할 고려 |
| **권장 학습 기간** | 30~90일 | 다양한 시장 상황 학습 |

**데이터 확인**:
```sql
-- 심볼별 데이터 현황 확인
SELECT symbol,
       MIN(ts) as start,
       MAX(ts) as end,
       COUNT(*) as rows
FROM candles_1m
GROUP BY symbol;
```

### 5.9 드리프트 모니터링 (`services/monitoring/`)

| 메트릭 | 설명 |
|--------|------|
| PSI | 피처 분포 안정성 (기준 vs 현재) |
| Missing Rate | NaN 비율 |
| Latency | 최신 데이터 지연 시간 |
| Outlier Count | median ± 3*std 벗어난 값 수 |

### 5.10 강화학습 (`services/rl/`) **(v3 추가)**

RL 하이브리드 시스템은 **LightGBM 예측을 상태로 사용**하여 최종 거래 결정을 내립니다.

#### 5.10.1 전체 구조

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RL 하이브리드 파이프라인                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  LightGBM   │ →  │  RL Agent   │ →  │   Policy    │ →  │  Execution  │   │
│  │  예측       │    │  (PPO/A2C)  │    │   Check     │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        ↓                  ↓                                                  │
│  er_long, q05,      State (17D) →                                           │
│  mae, hold          Action (4)                                              │
│                                                                              │
│  [ 기존 모델 예측 ]    [ RL 최종 결정 ]                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 5.10.2 트레이딩 환경 (`environment.py`)

**Gymnasium 기반 커스텀 환경** (`TradingEnvironment`):

**설정** (`TradingConfig`):
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `initial_balance` | 10,000 | 초기 자본 ($) |
| `max_position_size` | 1.0 | 최대 포지션 (자본 대비) |
| `trading_fee` | 0.0004 | 거래 수수료 (0.04%) |
| `slippage` | 0.0001 | 슬리피지 (0.01%) |
| `leverage` | 3 | 레버리지 |
| `lookback_window` | 60 | 과거 참조 기간 (분) |

**상태 공간 (17차원)**:
```python
observation = [
    # 모델 예측 (4개) - LightGBM 출력
    er_long,      # 기대 수익률
    q05_long,     # 5% 분위수 (하방 리스크)
    e_mae_long,   # 예상 MAE
    e_hold_long,  # 예상 보유 시간

    # 시장 지표 (10개)
    ret_1,        # 1분 수익률
    ret_5,        # 5분 수익률
    rsi,          # RSI (정규화: -0.5~0.5)
    macd_z,       # MACD z-score
    bb_z,         # Bollinger Band z-score
    vol_z,        # 거래량 z-score
    atr,          # ATR
    funding_z,    # 펀딩비율 z-score
    btc_regime,   # BTC 변동성 레짐 (0/1)
    adx,          # ADX (정규화)

    # 포지션 상태 (3개)
    position,       # 현재 포지션 (-1~1)
    position_time,  # 보유 시간 (0~1, 최대 360분 기준)
    unrealized_pnl, # 미실현 손익률
]
```

**액션 공간 (4개)**:
| 액션 | 값 | 설명 |
|------|-----|------|
| HOLD | 0 | 현재 포지션 유지 |
| LONG | 1 | 롱 진입 (숏 있으면 청산 후) |
| SHORT | 2 | 숏 진입 (롱 있으면 청산 후) |
| CLOSE | 3 | 현재 포지션 청산 |

**보상 함수**:
```python
reward = realized_pnl - trading_cost - funding_cost

# 거래 비용 = (fee + slippage) × position_size × balance
# 펀딩 비용 = position × funding_rate × balance (8시간마다)
```

---

#### 5.10.3 RL 에이전트 (`agent.py`)

**stable-baselines3 기반 PPO/A2C 래퍼** (`RLAgent`):

**설정** (`RLAgentConfig`):
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `algorithm` | "PPO" | 알고리즘 (PPO/A2C) |
| `learning_rate` | 3e-4 | 학습률 |
| `n_steps` | 2048 | 배치당 스텝 수 |
| `batch_size` | 64 | 미니배치 크기 |
| `n_epochs` | 10 | 에폭 수 |
| `gamma` | 0.99 | 할인율 |
| `verbose` | 1 | 로그 레벨 |

**주요 메서드**:
| 메서드 | 설명 |
|--------|------|
| `train(data, timesteps, eval_data, save_path)` | 에이전트 학습 |
| `predict(obs)` | 단일 관측값에서 액션 예측 |
| `decide(obs)` | 실시간 결정 (액션 + 신뢰도 + 확률분포) |
| `evaluate(env, n_episodes)` | 환경에서 성능 평가 |
| `save(path)` / `load(path)` | 모델 저장/로드 |

**`decide()` 출력 예시**:
```python
{
    "action": 1,                  # LONG
    "action_name": "LONG",
    "confidence": 0.78,           # 신뢰도
    "action_probs": {
        "HOLD": 0.12,
        "LONG": 0.78,
        "SHORT": 0.08,
        "CLOSE": 0.02,
    }
}
```

---

#### 5.10.4 RL 학습 실행

**학습 스크립트** (`scripts/train_rl.py`):
```bash
# 기본 학습 (BTC/ETH, 90일 학습, 100K 스텝)
python scripts/train_rl.py --symbols BTCUSDT,ETHUSDT --train-days 90 --timesteps 100000

# 전체 옵션
python scripts/train_rl.py \
    --symbols BTCUSDT,ETHUSDT,BNBUSDT \
    --train-days 90 \
    --eval-days 30 \
    --timesteps 200000 \
    --algorithm PPO \
    --save-dir ./rl_models
```

**명령줄 옵션**:
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--symbols` | BTCUSDT,ETHUSDT | 학습 심볼 (쉼표 구분) |
| `--train-days` | 90 | 학습 기간 (일) |
| `--eval-days` | 30 | 평가 기간 (일) |
| `--timesteps` | 100000 | 총 학습 스텝 |
| `--algorithm` | PPO | 알고리즘 (PPO/A2C) |
| `--save-dir` | ./rl_models | 모델 저장 경로 |

**학습 데이터 요구사항**:
- `features_1m` + `signals` 테이블 JOIN
- 최소 1,000 rows 이상 (lookback 60분 고려)
- 권장: 30~90일 데이터

**학습 흐름**:
```
1. 심볼별 데이터 로드 (features + signals JOIN)
2. Train/Eval 기간 분리
3. TradingEnvironment 생성
4. PPO/A2C 에이전트 학습
5. 평가 및 메트릭 계산
6. 모델 저장 (파일 + DB)
```

---

#### 5.10.5 RL 메트릭

| 메트릭 | 설명 |
|--------|------|
| `final_balance` | 최종 잔고 |
| `total_pnl` | 총 손익 |
| `total_trades` | 총 거래 횟수 |
| `win_rate` | 승률 |
| `profit_factor` | 이익비율 (총이익/총손실) |
| `sharpe_ratio` | 샤프 비율 |
| `max_drawdown` | 최대 낙폭 |
| `action_distribution` | 액션 분포 (HOLD/LONG/SHORT/CLOSE) |

---

#### 5.10.6 RL 데이터베이스 테이블

**`rl_models`**: 학습된 RL 모델 메타데이터
```sql
CREATE TABLE rl_models (
    model_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    algorithm TEXT NOT NULL,       -- PPO, A2C
    train_start TIMESTAMPTZ,
    train_end TIMESTAMPTZ,
    metrics JSONB,                 -- 학습 메트릭
    model_path TEXT,               -- 파일 경로
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**`rl_decisions`**: 실시간 RL 결정 기록
```sql
CREATE TABLE rl_decisions (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    model_id TEXT,
    action INTEGER,                -- 0-3
    action_name TEXT,              -- HOLD, LONG, SHORT, CLOSE
    confidence REAL,               -- 신뢰도
    action_probs JSONB,            -- 액션별 확률
    state JSONB,                   -- 입력 상태
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**`rl_episodes`**: 학습 에피소드 기록
```sql
CREATE TABLE rl_episodes (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL,
    episode INTEGER,
    total_reward REAL,
    total_trades INTEGER,
    win_rate REAL,
    final_balance REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

#### 5.10.7 RL API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/rl/models` | RL 모델 목록 |
| GET | `/api/rl/models/{id}` | 모델 상세 정보 |
| POST | `/api/rl/models/{id}/promote` | 프로덕션 승격 |
| GET | `/api/rl/decisions?symbol=&limit=` | RL 결정 히스토리 |
| GET | `/api/rl/decisions/stats?hours=24` | 결정 통계 |

---

#### 5.10.8 프론트엔드 컴포넌트

**`RLModelsTable`** (`apps/web/src/components/tables/RLModelsTable.tsx`):
- RL 모델 목록 표시
- 메트릭 (PF, Win Rate, Sharpe, Max DD)
- 프로덕션 승격 버튼

**`RLDecisionsTable`** (`apps/web/src/components/tables/RLDecisionsTable.tsx`):
- 실시간 RL 결정 피드
- 신뢰도 시각화
- 액션별 확률 분포

---

## 6. 데이터 흐름

### 6.1 실시간 트레이딩 루프

```
┌─────────────────────────────────────────────────────────────┐
│  Main Loop (매 5초)                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. FETCH MARKET DATA                                       │
│     ├─ Binance: klines (OHLCV), premium_index               │
│     ├─ Open Interest (미결제약정) [v3]                      │
│     └─ Long/Short Ratio, Taker Ratio [v3]                   │
│                                                             │
│  2. FEATURE ENGINEERING (33개 피처)                         │
│     ├─ 기본: ATR, ADX, BB, vol_z, funding_z, btc_regime     │
│     ├─ 모멘텀: RSI, MACD (signal, hist, z-score) [v3]       │
│     ├─ 거래량: OBV, OBV z-score [v3]                        │
│     └─ 심리: OI 변화율, L/S ratio, Taker ratio [v3]         │
│                                                             │
│  3. INFERENCE                                               │
│     └─ LightGBM 예측: er, q05, mae, hold                    │
│                                                             │
│  4. POLICY DECISION                                         │
│     └─ EV 계산 → LONG/SHORT/FLAT 결정                       │
│                                                             │
│  5. RISK CHECKS                                             │
│     └─ 포트폴리오/데이터/드리프트 체크                       │
│                                                             │
│  6. EXECUTION                                               │
│     ├─ SHADOW: 시뮬레이션 + DB 저장                         │
│     └─ LIVE: 브라켓 주문 실행                               │
│                                                             │
│  7. PUBLISH                                                 │
│     └─ Redis stream으로 시그널 브로드캐스트                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 학습 파이프라인

```
┌──────────────────────────────────────────────────┐
│  Batch Worker                                    │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. training_jobs 테이블 폴링 (status="queued")  │
│  2. 라벨링 실행                                  │
│     └─ 캔들 로드 → Triple-barrier 라벨 생성      │
│  3. 학습 실행                                    │
│     └─ Walk-forward → LightGBM 학습              │
│  4. 모델 등록                                    │
│     └─ MinIO 업로드 → DB 메타데이터 저장         │
│  5. 리포트 생성                                  │
│     └─ 메트릭 JSON 저장                          │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 7. 데이터베이스 스키마

### 시장 데이터
| 테이블 | 설명 | PK |
|--------|------|-----|
| `instruments` | 종목 정보 | symbol |
| `candles_1m` | OHLCV 데이터 | symbol, ts |
| `premium_index` | 마크가격, 펀딩비율 | symbol, ts |
| `funding_rates` | 펀딩 레이트 | symbol, ts |
| `open_interest` | 미결제약정 **(v3 추가)** | symbol, ts |
| `long_short_ratio` | 롱/숏 비율 **(v3 추가)** | symbol, ts |

### 피처 & 라벨
| 테이블 | 설명 | PK |
|--------|------|-----|
| `features_1m` | 엔지니어링된 피처 (JSONB) | symbol, ts |
| `labels_long_1m` | 롱 라벨 | symbol, ts, spec_hash |
| `labels_short_1m` | 숏 라벨 | symbol, ts, spec_hash |

### ML 모델
| 테이블 | 설명 | PK |
|--------|------|-----|
| `models` | 모델 메타데이터 | model_id (UUID) |
| `training_jobs` | 학습 작업 추적 | job_id (UUID) |
| `drift_metrics` | 드리프트 메트릭 | symbol, ts, schema_version |

### 트레이딩
| 테이블 | 설명 | PK |
|--------|------|-----|
| `signals` | 트레이딩 시그널 | symbol, ts |
| `orders` | 주문 | order_id |
| `fills` | 체결 | trade_id |
| `positions` | 포지션 이벤트 | symbol, ts, side |
| `account_snapshots` | 계정 스냅샷 | ts |
| `risk_events` | 리스크 이벤트 | ts, event_type, symbol |

### 강화학습 (v3 추가)
| 테이블 | 설명 | PK |
|--------|------|-----|
| `rl_models` | RL 모델 메타데이터 | model_id |
| `rl_decisions` | RL 에이전트 결정 기록 | id (BIGSERIAL) |
| `rl_episodes` | RL 학습 에피소드 | id |

---

## 8. API 엔드포인트

### 상태 & 모니터링
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 시스템 상태 |
| GET | `/api/universe` | 활성 종목 목록 |
| GET | `/api/account/latest` | 계정 스냅샷 |
| GET | `/api/positions/latest` | 현재 포지션 |
| GET | `/api/data-quality/summary` | 드리프트 메트릭 |
| GET | `/api/risk/state` | 리스크 이벤트 |

### 시그널 & 트레이딩
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/signals/latest` | 최신 시그널 |
| GET | `/api/orders` | 주문 목록 |
| POST | `/api/trading/toggle?mode={mode}` | 모드 전환 |

### 모델 & 학습
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/models` | 모델 목록 |
| POST | `/api/models/{id}/promote` | 프로덕션 승격 |
| GET | `/api/training/jobs` | 학습 작업 목록 |

### 강화학습 (v3 추가)
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/rl/models` | RL 모델 목록 |
| GET | `/api/rl/models/{id}` | RL 모델 상세 |
| POST | `/api/rl/models/{id}/promote` | RL 모델 프로덕션 승격 |
| GET | `/api/rl/decisions` | RL 결정 히스토리 |
| GET | `/api/rl/decisions/stats` | RL 결정 통계 |

### WebSocket
| 경로 | 설명 |
|------|------|
| `/ws/signal_updates` | 실시간 시그널 스트림 |

---

## 9. 환경 설정

### 핵심 설정 (`.env`)

```bash
# 모드
MODE=shadow              # off | shadow | live

# 데이터베이스
DATABASE_URL=postgresql://ta:ta@postgres:5432/ta

# Redis
REDIS_URL=redis://redis:6379

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=models

# Binance
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_TESTNET=false

# 리스크 관리
MAX_USED_MARGIN_PCT=0.35       # 최대 사용 마진 35%
DAILY_LOSS_LIMIT_PCT=0.02      # 일일 손실 한도 2%
MAX_POSITIONS=6                 # 최대 포지션 수
MAX_TOTAL_NOTIONAL_PCT=1.2     # 총 노출도 120%
MAX_DIRECTIONAL_NOTIONAL_PCT=0.8  # 방향성 노출도 80%

# 정책 임계값
EV_MIN=0.0                     # 최소 기대값
Q05_MIN=-0.002                 # 최소 5분위수
MAE_MAX=0.01                   # 최대 MAE

# 드리프트
DRIFT_ALERT_PSI=0.2
DRIFT_BLOCK_PSI=0.4

# 시뮬레이션
TAKER_FEE_RATE=0.0004          # 테이커 수수료 0.04%
SLIPPAGE_K=0.15                # 슬리피지 계수
```

---

## 10. Docker 컴포즈 서비스

```yaml
services:
  postgres:     # PostgreSQL 16 (5433:5432)
  redis:        # Redis 7 (6380:6379)
  minio:        # S3 스토리지 (9000, 9001)
  api:          # FastAPI (8000)
  worker_realtime:  # 실시간 트레이딩 루프
  worker_batch:     # 학습 작업 프로세서
  worker_universe:  # 유니버스 갱신
  web:          # Next.js UI (3000)
```

---

## 11. 아키텍처 패턴

### 1. 모노레포 구조
- `packages/common` 공유 라이브러리
- 서비스 간 느슨한 결합

### 2. 설정 기반
- Pydantic으로 `.env` 로드
- 하드코딩된 값 없음
- Redis로 런타임 모드 전환

### 3. 데이터베이스 중심
- PostgreSQL이 진실의 원천
- 벌크 upsert로 성능 최적화
- 시계열 파티셔닝 (symbol, ts PK)

### 4. 실시간 메시지 버스
- Redis Streams로 API-워커 분리
- WebSocket 클라이언트가 signal_updates 구독

### 5. 리스크 우선 설계
- 드리프트, 누락, 신선도에 대한 서킷 브레이커
- 포트폴리오 제약 (노출도, 마진, 일일 손실)
- 라이브 모드에서 SL/TP 필수
- 브라켓 실패 시 즉시 청산

### 6. 비동기 작업 처리
- 학습 작업 DB 큐잉
- 배치 워커가 폴링 및 처리
- 실시간 루프와 분리

---

## 12. 핵심 함수/클래스 참조

### 설정 & 타입
| 파일 | 클래스/함수 | 용도 |
|------|-------------|------|
| `config.py` | `Settings` | 환경변수 로드 |
| `types.py` | `StatusDTO`, `SignalDTO` | API 응답 모델 |
| `symbol_map.py` | `to_internal()`, `to_rest()` | 심볼 포맷 변환 |

### 데이터 액세스
| 파일 | 클래스/함수 | 용도 |
|------|-------------|------|
| `db.py` | `get_conn()`, `bulk_upsert()` | DB 연산 |
| `bus.py` | `RedisBus.publish()` | Redis Streams |
| `runtime.py` | `get_mode()`, `set_mode()` | 상태 관리 |

### 트레이딩 로직
| 파일 | 함수 | 용도 |
|------|------|------|
| `decide.py` | `decide()` | 시그널 생성 |
| `guard.py` | `check_risk()` | 리스크 체크 |
| `fill.py` | `simulate_trade_path()` | 시뮬레이션 |
| `trader.py` | `place_entry_with_brackets()` | 브라켓 주문 |

### ML 파이프라인
| 파일 | 함수 | 용도 |
|------|------|------|
| `predictor.py` | `Predictor.predict()` | 모델 추론 |
| `triple_barrier.py` | `label_direction_vectorized()` | 라벨 생성 |
| `train.py` | `run_training_job()` | LightGBM 학습 |

### 강화학습 (v3 추가)
| 파일 | 함수/클래스 | 용도 |
|------|-------------|------|
| `environment.py` | `TradingEnvironment` | Gymnasium 트레이딩 환경 |
| `agent.py` | `RLAgent.train()` | RL 에이전트 학습 |
| `agent.py` | `RLAgent.decide()` | 실시간 결정 (신뢰도 포함) |
| `decision_store.py` | `save_decision()` | 결정 DB 저장 |
| `train_rl.py` | `train_rl_agent()` | RL 학습 스크립트 |

---

## 13. 요약

이 시스템은 **안전성을 최우선**으로 하는 프로덕션 레벨의 ML 기반 암호화폐 트레이딩 시스템입니다:

- **견고한 데이터 파이프라인**: 수집 → 피처 엔지니어링 → 저장
- **확장된 시장 데이터**: OI, Long/Short Ratio, Taker Ratio 수집 (v3)
- **ML 기반 의사결정**: LightGBM 예측 + 정책 필터링 + Early Stopping
- **하이브리드 RL**: PPO/A2C 에이전트가 LightGBM 예측 기반 최종 결정 (v3)
- **33개 피처**: 가격, 변동성, 모멘텀, 거래량, 펀딩, BTC 레짐, 시장 심리
- **엄격한 리스크 제어**: 다층 서킷 브레이커 및 가드레일
- **실시간 운영**: 분 해상도 피처로 5초 루프
- **비동기 학습**: 모델 학습 전용 배치 워커
- **완전한 관측성**: 드리프트 감지, 리스크 이벤트, 데이터 품질 메트릭, RL 결정 기록
- **고정 유니버스**: 30개 심볼 (금/은 포함) 안정적 운영
- **유연한 배포**: Docker Compose로 멀티 워커 + UI

---

## 14. 변경 이력

### v3 (2026-02-01) - 시장 심리 데이터 & 기술적 지표 확장 + RL 하이브리드

**강화학습 (RL) 하이브리드 시스템**:
- Gymnasium 기반 트레이딩 환경 (`TradingEnvironment`)
- PPO/A2C 에이전트 (stable-baselines3)
- 17차원 상태 공간: 모델 예측(4) + 시장 지표(10) + 포지션 상태(3)
- 4개 이산 액션: HOLD, LONG, SHORT, CLOSE
- 모든 RL 결정 DB 저장 및 분석
- 새 테이블: `rl_models`, `rl_decisions`, `rl_episodes`
- RL API 엔드포인트: `/api/rl/models`, `/api/rl/decisions`
- 프론트엔드: RLModelsTable, RLDecisionsTable 컴포넌트

**고정 유니버스**:
- 동적 유니버스 업데이트 비활성화
- 30개 심볼 고정 (금/은 포함: XAUUSDT, XAGUSDT, PAXGUSDT)
- `.env`의 UNIVERSE 변수로 관리

**데이터 수집 확장**:
- Open Interest (미결제약정) 수집 추가
- Long/Short Ratio (글로벌, 탑 트레이더) 수집 추가
- Taker Buy/Sell Ratio 수집 추가
- 새 테이블: `open_interest`, `long_short_ratio`

**피처 확장** (14개 → 33개):
- RSI (14봉) 및 RSI Z-score
- MACD (12, 26, 9), Signal, Histogram, Z-score
- NATR (Normalized ATR, %)
- OBV (On-Balance Volume) 및 Z-score
- OI 변화율 (1시간, 4시간) 및 Z-score
- Long/Short 비율 피처 (7개)

**학습 개선**:
- Early Stopping 적용 (50 라운드 patience)
- n_estimators 500으로 증가 (조기 종료로 과적합 방지)
- 메트릭에 best_iteration 추가
- RL 학습 스크립트 (`scripts/train_rl.py`)

**스키마 버전**:
- schema_version=1: 기존 14개 피처
- schema_version=2: 신규 33개 피처

**신규 마이그레이션**:
- `009_rl_tables.sql`: RL 모델 및 결정 테이블

### v2 - 초기 버전
- 기본 트레이딩 시스템 구축
- Triple-Barrier 라벨링
- LightGBM 학습 파이프라인
- 리스크 관리 및 드리프트 모니터링
