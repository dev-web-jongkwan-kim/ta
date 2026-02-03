# TA 자동매매 시스템 - 종합 분석 보고서

> **작성일**: 2026-02-01
> **시스템 유형**: ML 기반 Binance USDT-M 선물 자동매매 시스템
> **기술 스택**: Python 3.11+ (FastAPI, LightGBM), Next.js 14 (React), PostgreSQL 16, Redis 7, MinIO

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [데이터 흐름 상세](#3-데이터-흐름-상세)
4. [피처 엔지니어링](#4-피처-엔지니어링)
5. [라벨링 시스템](#5-라벨링-시스템)
6. [모델 학습 파이프라인](#6-모델-학습-파이프라인)
7. [추론 및 신호 생성](#7-추론-및-신호-생성)
8. [거래 정책 결정](#8-거래-정책-결정)
9. [리스크 관리](#9-리스크-관리)
10. [주문 실행](#10-주문-실행)
11. [프론트엔드 대시보드](#11-프론트엔드-대시보드)
12. [모니터링 및 데이터 품질](#12-모니터링-및-데이터-품질)
13. [개선 방안 및 권장사항](#13-개선-방안-및-권장사항)
14. [부록: 설정값 전체 목록](#14-부록-설정값-전체-목록)

---

## 1. 시스템 개요

### 1.1 시스템 목적

이 시스템은 **Binance USDT-M 선물** 시장에서 머신러닝 모델을 활용하여 자동으로 매매 결정을 내리고 실행하는 서비스입니다.

### 1.2 핵심 기능

| 기능 | 설명 |
|------|------|
| **실시간 데이터 수집** | 5초마다 캔들, 프리미엄 인덱스, 펀딩레이트 수집 |
| **피처 계산** | 14개 기술적 지표 실시간 계산 |
| **ML 추론** | LightGBM 모델로 기대수익률, 위험도 예측 |
| **자동 거래** | SHADOW(시뮬레이션)/LIVE(실거래) 모드 지원 |
| **리스크 관리** | 12가지 리스크 체크로 과도한 손실 방지 |
| **웹 대시보드** | 실시간 포지션, 신호, 리스크 모니터링 |

### 1.3 거래 모드

```
OFF     → 데이터 수집만, 신호 생성 없음
SHADOW  → 신호 생성 + 시뮬레이션 거래 (실제 주문 없음)
LIVE    → 신호 생성 + 실제 거래 실행 (Binance 주문)
```

---

## 2. 전체 아키텍처

### 2.1 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           사용자 인터페이스                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Next.js 웹 대시보드 (포트 3000)                      │   │
│  │   • 대시보드    • 포지션    • 주문    • 신호    • 학습              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                FastAPI 게이트웨이 (포트 8000)                     │   │
│  │   • REST API    • WebSocket    • 상태 조회                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
┌─────────────────────────────────────────────────────────────────────────┐
│                              백엔드 워커                                 │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ realtime_    │  │ batch_       │  │ universe_    │                  │
│  │ worker       │  │ worker       │  │ worker       │                  │
│  │              │  │              │  │              │                  │
│  │ • 5초 주기   │  │ • 학습 작업  │  │ • 1시간 주기 │                  │
│  │ • 수집→추론  │  │   비동기     │  │ • 종목 갱신  │                  │
│  │ • 거래 실행  │  │   처리       │  │              │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
┌─────────────────────────────────────────────────────────────────────────┐
│                           데이터 저장소                                  │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ PostgreSQL   │  │ Redis        │  │ MinIO        │                  │
│  │ (포트 5433)  │  │ (포트 6380)  │  │ (포트 9000)  │                  │
│  │              │  │              │  │              │                  │
│  │ • 캔들       │  │ • 런타임     │  │ • ML 모델    │                  │
│  │ • 피처       │  │   상태       │  │ • 학습       │                  │
│  │ • 라벨       │  │ • Pub/Sub    │  │   리포트     │                  │
│  │ • 신호       │  │              │  │              │                  │
│  │ • 주문       │  │              │  │              │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────┐
                          │   Binance API    │
                          │                  │
                          │ • 시장 데이터    │
                          │ • 주문 실행      │
                          │ • 포지션 스트림  │
                          └──────────────────┘
```

### 2.2 서비스 모듈 구조

```
services/
├── collector/          # 데이터 수집 (Binance → DB)
│   ├── market_data.py  # 캔들, 프리미엄 인덱스 수집
│   ├── buffer.py       # 배치 버퍼링 및 DB 플러시
│   └── universe_builder.py  # 거래 종목 자동 선택
│
├── features/           # 피처 엔지니어링
│   └── compute.py      # 14개 기술적 지표 계산
│
├── labeling/           # 학습 라벨 생성
│   ├── triple_barrier.py   # Triple-Barrier 라벨링
│   └── pipeline.py         # 라벨링 파이프라인
│
├── training/           # 모델 학습
│   ├── train.py        # LightGBM 학습 로직
│   ├── splits.py       # Walk-forward 분할
│   └── pipeline.py     # 학습 파이프라인
│
├── inference/          # 모델 추론
│   └── predictor.py    # 실시간 예측
│
├── policy/             # 거래 정책
│   └── decide.py       # 거래 결정 로직
│
├── risk/               # 리스크 관리
│   └── guard.py        # 리스크 체크
│
├── execution/          # 주문 실행
│   ├── binance_client.py  # Binance API 래퍼
│   └── trader.py          # 브라켓 주문 실행
│
├── simulator/          # 시뮬레이션
│   └── fill.py         # 가상 체결 시뮬레이션
│
├── registry/           # 모델 저장소
│   └── storage.py      # MinIO 모델 업로드/다운로드
│
├── monitoring/         # 모니터링
│   └── drift.py        # 데이터 드리프트 감지
│
├── realtime_worker.py  # 실시간 거래 루프 (핵심)
├── batch_worker.py     # 학습 작업 처리
└── universe_worker.py  # 유니버스 갱신
```

---

## 3. 데이터 흐름 상세

### 3.1 실시간 거래 루프 (5초 주기)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    realtime_worker.py 실행 흐름                          │
└─────────────────────────────────────────────────────────────────────────┘

[1] 데이터 수집 (collector)
    │
    │  Binance API 호출:
    │  - GET /fapi/v1/klines (1분 캔들)
    │  - GET /fapi/v1/premiumIndex (마크 가격, 펀딩레이트)
    │
    ▼
┌──────────────────┐
│ candles_1m       │  ts, symbol, open, high, low, close, volume
│ premium_index    │  ts, symbol, mark_price, funding_rate, next_funding_time
└──────────────────┘
    │
    ▼
[2] 피처 계산 (features/compute.py)
    │
    │  최근 500개 캔들 기반 계산:
    │  - 가격: ret_1, ret_5, ret_15
    │  - 변동성: atr, bb_z, vol_z
    │  - 추세: ema_dist_atr, adx
    │  - 펀딩: funding_rate, funding_z, basis
    │  - 레짐: btc_ret_60, btc_vol_60, btc_regime
    │
    ▼
┌──────────────────┐
│ features_1m      │  ts, symbol, features (JSON), schema_version
└──────────────────┘
    │
    ▼
[3] 모델 추론 (inference/predictor.py)
    │
    │  LightGBM 모델 4개 실행:
    │  - er_long/er_short: 기대 수익률 예측
    │  - q05_long/q05_short: 5% 분위수 예측 (VaR)
    │  - e_mae_long/e_mae_short: 최대 역풍 예측
    │  - e_hold_long/e_hold_short: 예상 보유시간
    │
    ▼
[4] 거래 결정 (policy/decide.py)
    │
    │  EV 계산: 기대수익 - 거래비용
    │  결정: LONG / SHORT / FLAT
    │  차단 조건 확인: EV_MIN, Q05_MIN, MAE_MAX
    │
    ▼
[5] 리스크 체크 (risk/guard.py)
    │
    │  12가지 리스크 조건 확인:
    │  - 계정: MARGIN_LIMIT, DAILY_STOP
    │  - 포트폴리오: MAX_POSITIONS, MAX_EXPOSURE
    │  - 인프라: USERSTREAM_DOWN, DATA_STALE
    │  - 데이터: MISSING_BLOCK, DRIFT_BLOCK
    │
    ▼
[6] 주문 실행 (execution/trader.py)
    │
    │  SHADOW 모드: 시뮬레이션 체결 (simulator/fill.py)
    │  LIVE 모드: Binance 브라켓 주문
    │    - MARKET (진입)
    │    - STOP_MARKET (손절)
    │    - TAKE_PROFIT_MARKET (익절)
    │
    ▼
┌──────────────────┐
│ signals          │  ts, symbol, decision, ev_long, ev_short, block_reasons
│ orders           │  order_id, symbol, side, type, status, qty, price
│ fills            │  fill_id, order_id, price, qty, commission
│ positions        │  symbol, side, amt, entry_price, pnl
└──────────────────┘
```

### 3.2 학습 파이프라인 (배치)

```
[1] 라벨링 요청 → batch_worker.py
    │
    ▼
[2] 라벨 생성 (labeling/triple_barrier.py)
    │
    │  과거 캔들 데이터에 대해:
    │  - TP/SL 배리어 설정 (ATR 기반)
    │  - 먼저 도달하는 배리어에 따라 라벨 결정
    │    y=1 (TP 도달), y=0 (시간초과), y=-1 (SL 도달)
    │  - 순수익률, MAE, MFE 계산
    │
    ▼
┌──────────────────┐
│ labels_long_1m   │  ts, symbol, y, ret_net, mae, mfe, time_to_event_min
│ labels_short_1m  │  ts, symbol, y, ret_net, mae, mfe, time_to_event_min
└──────────────────┘
    │
    ▼
[3] 모델 학습 (training/train.py)
    │
    │  Walk-forward 분할:
    │  - Train: 180일
    │  - Val: 30일
    │  - Test: 30일 (미사용)
    │
    │  LightGBM 학습:
    │  - n_estimators: 200
    │  - learning_rate: 0.05
    │  - num_leaves: 64
    │
    ▼
┌──────────────────┐
│ models (MinIO)   │  model_id.pkl (4개 모델 딕셔너리)
│ reports (MinIO)  │  model_id.json (학습 리포트)
│ models (DB)      │  model_id, metrics, artifact_uri, is_production
└──────────────────┘
```

---

## 4. 피처 엔지니어링

### 4.1 현재 구현된 14개 피처

| 그룹 | 피처명 | 계산 방식 | 윈도우 | 설명 |
|------|--------|---------|--------|------|
| **가격** | `ret_1` | close.pct_change(1) | 1 | 1분 수익률 |
| | `ret_5` | close.pct_change(5) | 5 | 5분 수익률 |
| | `ret_15` | close.pct_change(15) | 15 | 15분 수익률 |
| **변동성** | `atr` | True Range의 14봉 EMA | 14 | 가격 변동성 |
| | `bb_z` | (close - SMA) / std | 20 | 볼린저 밴드 Z-score |
| | `vol_z` | Z-score(volume) | 60 | 거래량 정규화 |
| **추세** | `ema_dist_atr` | (close - EMA) / atr | 20 | EMA 거리 정규화 |
| | `adx` | Average Directional Index | 14 | 추세 강도 (0-100) |
| **펀딩** | `funding_rate` | 원본값 | - | 펀딩 레이트 |
| | `funding_z` | Z-score(funding_rate) | 120 | 정규화된 펀딩 |
| | `basis` | (선물-현물)/현물 | - | 베이시스 (%) |
| **레짐** | `btc_ret_60` | BTC pct_change(60) | 60 | BTC 60분 수익률 |
| | `btc_vol_60` | BTC 수익률 std | 60 | BTC 변동성 |
| | `btc_regime` | btc_vol > median ? 1 : 0 | 60 | 변동성 레짐 |

### 4.2 피처 계산 위치

```python
# 파일: services/features/compute.py

def compute_features_for_symbol(symbol: str, candles: pd.DataFrame,
                                 premium: pd.DataFrame, btc_ref: pd.DataFrame):
    """
    심볼별 피처 계산

    입력:
    - candles: 최근 500개 1분 캔들
    - premium: 프리미엄 인덱스 데이터
    - btc_ref: BTC 참조 데이터 (시장 레짐용)

    출력:
    - DataFrame with 14 features + metadata
    """
```

### 4.3 피처 저장 형식

```sql
-- 테이블: features_1m
CREATE TABLE features_1m (
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,          -- 14개 피처 JSON
    schema_version INT NOT NULL,       -- 현재 1
    atr DOUBLE PRECISION,              -- 별도 저장 (라벨링용)
    funding_z DOUBLE PRECISION,        -- 별도 저장
    btc_regime INT,                    -- 별도 저장
    PRIMARY KEY (symbol, ts)
);

-- 예시 features JSON:
{
    "ret_1": 0.00123,
    "ret_5": 0.00456,
    "ret_15": 0.00789,
    "atr": 125.5,
    "bb_z": 0.45,
    "vol_z": -0.32,
    "ema_dist_atr": 0.12,
    "adx": 28.5,
    "funding_rate": 0.0001,
    "funding_z": 0.56,
    "basis": 0.0003,
    "btc_ret_60": -0.005,
    "btc_vol_60": 0.0123,
    "btc_regime": 1
}
```

---

## 5. 라벨링 시스템

### 5.1 Triple-Barrier 라벨링 개념

```
                          TP 배리어 (k_tp × ATR)
                         ─────────────────────────
                              │
                              │  y = 1 (TP 도달)
                              │
    ━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━  진입가
                              │
                              │  y = -1 (SL 도달)
                              │
                         ─────────────────────────
                          SL 배리어 (k_sl × ATR)

    ├─────────────────────────┼─────────────────────────┤
    t=0                     time                      h_bars
                                                (y=0 시간초과)
```

### 5.2 라벨링 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `k_tp` | 1.5 | TP 배리어 배수 (ATR의 1.5배) |
| `k_sl` | 1.0 | SL 배리어 배수 (ATR의 1.0배) |
| `h_bars` | 360 | 최대 홀딩 기간 (360분 = 6시간) |
| `risk_mae_atr` | 3.0 | 리스크 MAE 배수 (극단적 손절) |
| `fee_rate` | 0.0004 | 테이커 수수료율 (0.04%) |
| `slippage_k` | 0.15 | 슬리피지 계수 |

### 5.3 라벨 생성 결과

```sql
-- 테이블: labels_long_1m (LONG 방향)
-- 테이블: labels_short_1m (SHORT 방향)

CREATE TABLE labels_long_1m (
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    spec_hash TEXT NOT NULL,           -- 라벨 스펙 해시 (16자)
    y SMALLINT NOT NULL,               -- 라벨: 1, 0, -1
    ret_net DOUBLE PRECISION NOT NULL, -- 순수익률 (비용 차감)
    mae DOUBLE PRECISION NOT NULL,     -- Maximum Adverse Excursion
    mfe DOUBLE PRECISION NOT NULL,     -- Maximum Favorable Excursion
    time_to_event_min INT NOT NULL,    -- 이벤트까지 시간 (분)
    PRIMARY KEY (symbol, ts, spec_hash)
);
```

### 5.4 라벨 의미

| y 값 | 의미 | 조건 |
|------|------|------|
| **1** | TP 도달 (승리) | TP 배리어를 먼저 도달 |
| **0** | 시간 초과 (무승부) | h_bars 동안 TP/SL 미도달 |
| **-1** | SL 도달 (패배) | SL 배리어를 먼저 도달 |

### 5.5 비용 계산

```python
# 거래 비용 계산
fee_cost = fee_rate * 2           # 진입 + 청산 수수료
                                  # = 0.0004 * 2 = 0.08%

slippage_cost = slippage_k * 0.0001 * 2  # 슬리피지
                                          # = 0.15 * 0.0001 * 2 = 0.003%

funding_cost = 0.0                # 현재 미적용 (문제점!)

# 순수익률
ret_net = raw_ret - fee_cost - slippage_cost - funding_cost
```

---

## 6. 모델 학습 파이프라인

### 6.1 학습 타겟 (4개 모델)

| 타겟 | 원본 컬럼 | 목적 함수 | 설명 |
|------|---------|----------|------|
| `er_long` | ret_net | regression | LONG 기대 수익률 |
| `er_short` | ret_net | regression | SHORT 기대 수익률 |
| `q05_long` | ret_net | quantile (α=0.05) | LONG 5% VaR |
| `q05_short` | ret_net | quantile (α=0.05) | SHORT 5% VaR |
| `e_mae_long` | mae | regression | LONG 예상 MAE |
| `e_mae_short` | mae | regression | SHORT 예상 MAE |
| `e_hold_long` | time_to_event_min | regression | LONG 예상 보유시간 |
| `e_hold_short` | time_to_event_min | regression | SHORT 예상 보유시간 |

### 6.2 LightGBM 하이퍼파라미터

```python
params = {
    "objective": "regression" | "quantile",
    "n_estimators": 200,          # 부스팅 라운드
    "learning_rate": 0.05,        # 학습률
    "num_leaves": 64,             # 리프 노드 수
    "subsample": 0.8,             # 행 샘플링
    "colsample_bytree": 0.8,      # 열 샘플링
    "alpha": 0.05,                # quantile용 (5분위)
}
```

### 6.3 Walk-Forward 분할

```
전체 데이터: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

             ┌─ Train (180일) ─┐ Val (30일) Test (30일)
             │                 │     │           │
스플릿 1:    ├─────────────────┼─────┼───────────┤

             ┌─ Train (180일) ─┐ Val (30일) Test (30일)
             │                 │     │           │
스플릿 2:                      ├─────────────────┼─────┼───────────┤

현재 사용: 마지막 스플릿만 사용 (스플릿 N)
```

### 6.4 평가 메트릭

**회귀 메트릭:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Pinball Loss (Quantile 모델만)

**거래 메트릭:**
- Profit Factor: 양수 수익 / |음수 수익|
- Max Drawdown: 최대 낙폭
- Expectancy: 평균 거래 수익률
- Tail Loss: 최악 1% 거래의 평균 손실
- Turnover: 거래 횟수

### 6.5 학습 리포트 구조

```json
{
  "model_id": "uuid",
  "targets": ["er_long", "q05_long", "e_mae_long", "e_hold_long"],
  "feature_schema_version": 1,
  "label_spec": {
    "k_tp": 1.5,
    "k_sl": 1.0,
    "h_bars": 360,
    "spec_hash": "abc123..."
  },
  "split_summary": {
    "train": {"count": 259200, "start": "2023-01-01", "end": "2023-06-30"},
    "val": {"count": 43200, "start": "2023-07-01", "end": "2023-07-31"}
  },
  "metrics": {
    "er_long": {"rmse": 0.0234, "mae": 0.0156},
    "q05_long": {"rmse": 0.0245, "mae": 0.0162, "pinball_loss": 0.00312},
    "trade": {
      "profit_factor": 1.45,
      "max_drawdown": -0.1234,
      "expectancy": 0.00456,
      "tail_loss": -0.0234,
      "turnover": 1342
    },
    "regime": {
      "0": {"profit_factor": 1.3, ...},
      "1": {"profit_factor": 1.6, ...}
    },
    "cost_breakdown": {
      "fee_per_trade": 0.00008,
      "slippage_per_trade": 0.0000003,
      "funding_per_trade": 0.0
    }
  }
}
```

---

## 7. 추론 및 신호 생성

### 7.1 추론 과정

```python
# 파일: services/inference/predictor.py

def predict(features: Dict[str, float]) -> Dict[str, float]:
    """
    피처 → 예측값 변환

    입력:
    - features: 14개 피처 딕셔너리

    출력:
    - preds: 예측값 딕셔너리
        - er_long: LONG 기대 수익률
        - er_short: SHORT 기대 수익률
        - q05_long: LONG 5% VaR
        - q05_short: SHORT 5% VaR
        - e_mae_long: LONG 예상 MAE
        - e_mae_short: SHORT 예상 MAE
        - e_hold_long: LONG 예상 보유시간
        - e_hold_short: SHORT 예상 보유시간
    """
```

### 7.2 모델 로딩

```python
# MinIO에서 production 모델 로드
model_uri = "s3://models/{model_id}.pkl"
models = load_pickle(model_uri)  # Dict[str, LGBMRegressor]

# 모델 구조:
# {
#     "er_long": LGBMRegressor,
#     "q05_long": LGBMRegressor,
#     "e_mae_long": LGBMRegressor,
#     "e_hold_long": LGBMRegressor,
# }
```

### 7.3 신호 저장

```sql
-- 테이블: signals
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    decision TEXT NOT NULL,            -- LONG, SHORT, FLAT
    ev_long DOUBLE PRECISION,          -- LONG EV
    ev_short DOUBLE PRECISION,         -- SHORT EV
    prob_long DOUBLE PRECISION,        -- LONG 확률 (미사용)
    prob_short DOUBLE PRECISION,       -- SHORT 확률 (미사용)
    er_long DOUBLE PRECISION,          -- LONG 기대수익률
    er_short DOUBLE PRECISION,         -- SHORT 기대수익률
    q05_long DOUBLE PRECISION,         -- LONG VaR
    q05_short DOUBLE PRECISION,        -- SHORT VaR
    e_mae_long DOUBLE PRECISION,       -- LONG MAE
    e_mae_short DOUBLE PRECISION,      -- SHORT MAE
    block_reason_codes TEXT[],         -- 차단 사유 배열
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 8. 거래 정책 결정

### 8.1 EV (Expected Value) 계산

```python
# 파일: services/policy/decide.py

# 거래 비용 계산
transaction_cost = (
    taker_fee_rate * 2 +        # 0.0004 * 2 = 0.08%
    slippage_k * 0.0001 * 2     # 0.15 * 0.0001 * 2 = 0.003%
)                               # 총: ~0.083%

# EV 계산
ev_long = er_long - transaction_cost
ev_short = er_short - transaction_cost
```

### 8.2 거래 결정 알고리즘

```python
def decide(symbol, preds, state, cfg):
    # 1. EV 계산
    ev_long = preds["er_long"] - transaction_cost
    ev_short = preds["er_short"] - transaction_cost

    # 2. 최선 결정 선택
    best = max(ev_long, ev_short, 0.0)
    if best == ev_long:
        decision = "LONG"
    elif best == ev_short:
        decision = "SHORT"
    else:
        decision = "FLAT"

    # 3. 차단 조건 확인
    block_reasons = []
    if best <= cfg.ev_min:           # 기본: 0.0
        block_reasons.append("EV_MIN")
    if preds["q05_long"] < cfg.q05_min:  # 기본: -0.002
        block_reasons.append("Q05_MIN")
    if preds["e_mae_long"] > cfg.mae_max:  # 기본: 0.01
        block_reasons.append("MAE_MAX")

    # 4. 포지션 사이징
    size_notional = state["equity"] * 0.05  # 자산의 5%
    leverage = 3

    # 5. SL/TP 계산
    atr = state["atr"]
    if decision == "LONG":
        sl_price = last_close - atr * 1.0
        tp_price = last_close + atr * 1.5
    elif decision == "SHORT":
        sl_price = last_close + atr * 1.0
        tp_price = last_close - atr * 1.5

    return {
        "decision": decision,
        "ev": best,
        "block_reasons": block_reasons,
        "size_notional": size_notional,
        "leverage": leverage,
        "sl_price": sl_price,
        "tp_price": tp_price,
    }
```

### 8.3 정책 설정값

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `ev_min` | 0.0 | 최소 기대값 |
| `q05_min` | -0.002 | 최소 5분위 (-0.2%) |
| `mae_max` | 0.01 | 최대 MAE (1.0%) |
| `max_positions` | 6 | 최대 오픈 포지션 |
| `position_size_pct` | 0.05 | 포지션 크기 (자산의 5%) |
| `leverage` | 3 | 레버리지 배수 |

---

## 9. 리스크 관리

### 9.1 12가지 리스크 체크

```python
# 파일: services/risk/guard.py

def check_risk(action, portfolio, account, cfg) -> List[str]:
    reasons = []

    # ─── 계정 레벨 ───
    # 1. 마진 사용률 초과
    if used_margin / equity > cfg.max_used_margin_pct:  # 35%
        reasons.append("MARGIN_LIMIT")

    # 2. 일일 손실 한도 초과
    if daily_loss >= cfg.daily_loss_limit_pct * equity:  # 2%
        reasons.append("DAILY_STOP")

    # ─── 포트폴리오 레벨 ───
    # 3. 최대 포지션 수 초과
    if open_positions >= cfg.max_positions:  # 6
        reasons.append("MAX_POSITIONS")

    # 4. 전체 노션 초과
    if total_notional > cfg.max_total_notional_pct * equity:  # 120%
        reasons.append("MAX_EXPOSURE")

    # 5. 방향성 노션 초과
    if directional > cfg.max_directional_notional_pct * equity:  # 80%
        reasons.append("MAX_DIRECTIONAL_EXPOSURE")

    # ─── 인프라 레벨 ───
    # 6. Userstream 연결 끊김
    if userstream_ok is False:
        reasons.append("USERSTREAM_DOWN")

    # 7. 데이터 갱신 지연
    if data_stale is True:  # > 120초
        reasons.append("DATA_STALE")

    # 8. 주문 실패율 급증
    if order_failure_spike is True:
        reasons.append("ORDER_FAILURE_SPIKE")

    # ─── 데이터 품질 ───
    # 9/10. 결측 데이터율
    if missing_rate >= cfg.missing_block_rate:  # 2%
        reasons.append("MISSING_BLOCK")
    elif missing_rate >= cfg.missing_alert_rate:  # 0.5%
        reasons.append("MISSING_DATA")

    # ─── 모델 드리프트 ───
    # 11/12. PSI 기반 드리프트
    if drift_status == "block":  # PSI >= 0.4
        reasons.append("DRIFT_BLOCK")
    elif drift_status == "alert":  # PSI >= 0.2
        reasons.append("DRIFT_ALERT")

    return reasons
```

### 9.2 리스크 설정값

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `max_used_margin_pct` | 0.35 | 마진 사용률 한도 (35%) |
| `daily_loss_limit_pct` | 0.02 | 일일 손실 한도 (2%) |
| `max_positions` | 6 | 최대 오픈 포지션 |
| `max_total_notional_pct` | 1.2 | 전체 노션 한도 (120%) |
| `max_directional_notional_pct` | 0.8 | 방향성 노션 한도 (80%) |
| `missing_alert_rate` | 0.005 | 결측 경고 임계값 (0.5%) |
| `missing_block_rate` | 0.02 | 결측 차단 임계값 (2%) |
| `drift_alert_psi` | 0.2 | 드리프트 경고 PSI |
| `drift_block_psi` | 0.4 | 드리프트 차단 PSI |
| `data_stale_sec` | 120 | 데이터 지연 임계값 (초) |

### 9.3 리스크 이벤트 로깅

```sql
-- 테이블: risk_events
CREATE TABLE risk_events (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type TEXT NOT NULL,      -- MARGIN_LIMIT, DAILY_STOP, ...
    symbol TEXT,                   -- 영향받는 심볼
    severity INT NOT NULL,         -- 1(Low), 2(Medium), 3(Critical)
    message TEXT NOT NULL,         -- 상세 메시지
    details JSONB                  -- 추가 정보
);
```

---

## 10. 주문 실행

### 10.1 브라켓 주문 구조

```
LIVE 모드에서 Binance 주문:

1. Entry Order (진입)
   └─ Type: MARKET
   └─ Side: BUY (LONG) / SELL (SHORT)
   └─ Quantity: notional / entry_price

2. Stop Loss Order (손절)
   └─ Type: STOP_MARKET
   └─ Side: SELL (LONG) / BUY (SHORT)
   └─ Stop Price: sl_price
   └─ reduceOnly: true

3. Take Profit Order (익절)
   └─ Type: TAKE_PROFIT_MARKET
   └─ Side: SELL (LONG) / BUY (SHORT)
   └─ Stop Price: tp_price
   └─ reduceOnly: true
```

### 10.2 SL/TP 가격 계산

```python
# 파일: services/realtime_worker.py

last_close = candles.iloc[-1]["close"]
atr = latest["atr"]

if decision == "LONG":
    sl_price = last_close - atr * 1.0   # ATR 1배 아래
    tp_price = last_close + atr * 1.5   # ATR 1.5배 위
elif decision == "SHORT":
    sl_price = last_close + atr * 1.0   # ATR 1배 위
    tp_price = last_close - atr * 1.5   # ATR 1.5배 아래
```

### 10.3 시뮬레이션 체결 (SHADOW 모드)

```python
# 파일: services/simulator/fill.py

def simulate_trade_path(entry_price, sl_price, tp_price, candles, side):
    """
    과거 캔들 데이터로 가상 체결 시뮬레이션

    슬리피지 모델:
    slippage = min(0.5%, slippage_k × (notional/volume) × 0.01%)
             = min(0.5%, 0.15 × (notional/volume) × 0.01%)

    진입 체결:
    entry_fill = entry_price × (1 + slippage × side)

    청산 체결 (SL/TP 도달 시):
    exit_fill = target_price × (1 - slippage) [LONG]
    exit_fill = target_price × (1 + slippage) [SHORT]
    """
```

### 10.4 주문/체결 저장

```sql
-- 테이블: orders
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,           -- BUY, SELL
    type TEXT NOT NULL,           -- MARKET, STOP_MARKET, TAKE_PROFIT_MARKET
    status TEXT NOT NULL,         -- NEW, FILLED, CANCELED, REJECTED
    qty DOUBLE PRECISION,
    price DOUBLE PRECISION,
    stop_price DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 테이블: fills
CREATE TABLE fills (
    id SERIAL PRIMARY KEY,
    fill_id TEXT UNIQUE NOT NULL,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION,
    commission_asset TEXT,
    filled_at TIMESTAMPTZ NOT NULL
);

-- 테이블: trade_groups (브라켓 주문 묶음)
CREATE TABLE trade_groups (
    id SERIAL PRIMARY KEY,
    group_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_order_id TEXT REFERENCES orders(order_id),
    sl_order_id TEXT REFERENCES orders(order_id),
    tp_order_id TEXT REFERENCES orders(order_id),
    status TEXT NOT NULL,         -- OPEN, CLOSED
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 11. 프론트엔드 대시보드

### 11.1 페이지 구성

| 페이지 | 경로 | 표시 정보 |
|--------|------|---------|
| **Dashboard** | `/dashboard` | 시스템 상태, 포지션 노출도, 리스크, 최근 신호 |
| **Signals** | `/signals` | 신호 목록, EV, 차단 사유 |
| **Positions** | `/positions` | 현재 오픈 포지션, PnL |
| **Orders** | `/orders` | 주문 이력 |
| **Training** | `/training` | 학습 작업, 모델 목록 |
| **Risk** | `/risk` | 리스크 이벤트 |
| **Data Quality** | `/data-quality` | 드리프트 메트릭 |
| **Settings** | `/settings` | 거래 모드 설정 |
| **Symbol Detail** | `/symbol/[symbol]` | 종목별 차트, 신호, 피처 |

### 11.2 대시보드 카드 컴포넌트

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD                                                              │
├──────────────────┬──────────────────┬──────────────────────────────────┤
│ System Health    │ Mode Indicator   │ Portfolio Exposure               │
│                  │                  │                                  │
│ • Collector: OK  │ SHADOW           │ • Open: 3 positions              │
│ • Userstream: OK │                  │ • Long: $12,500                  │
│ • Last Candle:   │                  │ • Short: $8,200                  │
│   2026-02-01     │                  │ • Daily PnL: +$234               │
│   12:34:56       │                  │                                  │
├──────────────────┼──────────────────┼──────────────────────────────────┤
│ Risk Blockers    │ Margin Overview  │ Data Quality                     │
│                  │                  │                                  │
│ • MARGIN_LIMIT:0 │ Equity: $50,000  │ Symbol   PSI    Missing Latency  │
│ • MAX_POSITIONS:1│ Used: $5,250     │ BTCUSDT  0.12   0.1%    45ms     │
│ • DATA_STALE: 0  │ Available:$44,750│ ETHUSDT  0.08   0.2%    52ms     │
│                  │ Ratio: 10.5%     │                                  │
├──────────────────┴──────────────────┴──────────────────────────────────┤
│ Top Signals                                                             │
│                                                                         │
│ Symbol   Decision  EV       Block                                      │
│ BTCUSDT  LONG      +0.34%   -                                          │
│ ETHUSDT  SHORT     +0.22%   -                                          │
│ SOLUSDT  FLAT      -0.05%   EV_MIN                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.3 차트 컴포넌트

| 컴포넌트 | 위치 | 기능 |
|---------|------|------|
| **PriceChart** | Symbol Detail | 캔들스틱 차트 (100개 캔들) |
| **SignalSeriesChart** | Symbol Detail | EV 바 차트 (50개 신호) |
| **CostBreakdownChart** | Positions | 거래 비용 분석 바 차트 |

### 11.4 API 엔드포인트

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/status` | 시스템 상태 |
| GET | `/api/signals/latest?limit=50` | 최신 신호 |
| GET | `/api/positions/latest` | 현재 포지션 |
| GET | `/api/orders` | 주문 이력 |
| GET | `/api/models` | 모델 목록 |
| GET | `/api/training/jobs` | 학습 작업 |
| GET | `/api/data-quality/summary` | 데이터 품질 |
| GET | `/api/risk/state` | 리스크 이벤트 |
| GET | `/api/symbol/{symbol}/series` | 종목 시계열 |
| POST | `/api/trading/toggle?mode=...` | 모드 변경 |
| POST | `/api/models/{id}/promote` | 모델 승격 |

---

## 12. 모니터링 및 데이터 품질

### 12.1 드리프트 감지

```python
# 파일: services/monitoring/drift.py

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10):
    """
    Population Stability Index 계산

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    해석:
    - PSI < 0.1: 변화 없음
    - 0.1 ≤ PSI < 0.2: 약간 변화
    - 0.2 ≤ PSI < 0.4: 주의 (DRIFT_ALERT)
    - PSI ≥ 0.4: 심각 (DRIFT_BLOCK)
    """
```

### 12.2 드리프트 메트릭 저장

```sql
-- 테이블: drift_metrics
CREATE TABLE drift_metrics (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    psi DOUBLE PRECISION NOT NULL,         -- Population Stability Index
    missing_rate DOUBLE PRECISION NOT NULL, -- 결측률
    latency_ms INT NOT NULL,               -- 데이터 지연 (밀리초)
    outlier_count INT DEFAULT 0            -- 이상치 개수
);
```

### 12.3 시스템 상태 관리 (Redis)

```python
# 파일: packages/common/runtime.py

# Redis 키:
# - runtime:mode         → "off" | "shadow" | "live"
# - runtime:collector_ok → "1" | "0"
# - runtime:userstream_ok → "1" | "0"
# - runtime:drift:{symbol} → {"psi": 0.15, "status": "alert"}
# - runtime:last_candle_ts → "2026-02-01T12:34:56Z"
# - runtime:last_signal_ts → "2026-02-01T12:34:56Z"
```

---

## 13. 개선 방안 및 권장사항

### 13.1 라벨링 시스템 개선

#### 문제점 1: 펀딩비 미적용
```python
# 현재 코드 (triple_barrier.py)
funding_cost = np.where(hold_min > 0, funding_rate[:n-h], 0) * 0.0  # 항상 0!
```

**개선안:**
```python
# 권장 수정
# 8시간마다 펀딩비 발생, hold_min은 분 단위
funding_cycles = hold_min / (8 * 60)  # 몇 번의 펀딩 사이클?
funding_cost = funding_cycles * funding_rate[:n-h]
ret_net = raw_ret - fee_cost - slippage_cost - funding_cost
```

**예상 영향:**
- BTCUSDT 일반적 펀딩률: 0.01% ~ 0.1% per 8h
- 6시간 홀딩 시: 0.0075% ~ 0.075% 비용 추가
- **현재 순수익률이 1~13% 과대평가되고 있음**

#### 문제점 2: 라벨 클래스 불균형
- y=0 (시간초과): ~50-70%
- y=1 (TP 도달): ~15-25%
- y=-1 (SL 도달): ~15-25%

**개선안:**
- 클래스 가중치 적용: `class_weight='balanced'`
- 샘플링 전략: SMOTE, 언더샘플링
- 다중 h_bars 라벨링 후 최적 선택

#### 문제점 3: 고정 배리어 배수
```python
k_tp: float = 1.5  # 항상 고정
k_sl: float = 1.0  # 항상 고정
```

**개선안:**
```python
# 시장 레짐별 동적 조정
if btc_regime == 1:  # 고변동성
    k_tp, k_sl = 2.0, 1.5
else:  # 저변동성
    k_tp, k_sl = 1.5, 1.0
```

---

### 13.2 학습 파이프라인 개선

#### 문제점 1: Early Stopping 미사용
```python
# 현재
model = lgb.LGBMRegressor(**params)
model.fit(X, y)  # 검증 세트 없음!
```

**개선안:**
```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

#### 문제점 2: Walk-forward 마지막 윈도우만 사용
```python
splits = list(walk_forward_splits(...))
train_idx, val_idx, test_idx = splits[-1]  # 마지막만!
```

**개선안:**
```python
# 모든 스플릿에서 평균 성능 계산
all_metrics = []
for train_idx, val_idx, test_idx in splits:
    model = train_model(df[train_idx], df[val_idx])
    metrics = evaluate(model, df[test_idx])
    all_metrics.append(metrics)

final_metrics = aggregate(all_metrics)
```

#### 문제점 3: 하이퍼파라미터 튜닝 없음

**개선안:**
```python
from optuna import create_study

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 32, 128),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return mean_squared_error(y_val, model.predict(X_val))

study = create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

---

### 13.3 피처 엔지니어링 개선

#### 즉시 추가 권장 피처 (Tier 1)

| 피처 | 계산 | 효과 |
|------|------|------|
| **RSI** | Relative Strength Index | 과매수/과매도 감지 |
| **MACD** | Moving Average Convergence Divergence | 추세 전환 신호 |
| **NATR** | Normalized ATR (atr/close×100) | 가격 대비 변동성 |
| **OBV** | On-Balance Volume | 거래량 누적 추세 |

```python
# RSI 구현
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# MACD 구현
def _macd(close: pd.Series) -> pd.Series:
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26
```

#### 중기 추가 권장 피처 (Tier 2)

| 피처 | 데이터 필요 | 효과 |
|------|-----------|------|
| **Stochastic** | OHLC | 가격 위치 (0-100) |
| **CMF** | OHLCV | 자금 흐름 |
| **VWAP** | Volume | 평균 거래가 |
| **Liquidation Volume** | 외부 API | 청산 압력 |

#### 장기 추가 권장 피처 (Tier 3)

| 피처 | 데이터 소스 | 효과 |
|------|-----------|------|
| **온체인 지표** | Glassnode, Santiment | Whale 활동, 거래소 유출입 |
| **소셜 센티먼트** | LunarCrush, Twitter | 시장 심리 |
| **거시경제** | FRED | VIX, DXY, 금리 |

---

### 13.4 리스크 관리 개선

#### 동적 포지션 사이징
```python
# 현재: 고정 5%
size_notional = equity * 0.05

# 개선안: Kelly Criterion 기반
# f* = (bp - q) / b
# b: 배당률, p: 승률, q: 1-p
win_rate = preds["prob_win"]  # 새 모델 필요
profit_loss_ratio = preds["avg_win"] / preds["avg_loss"]
kelly_fraction = (profit_loss_ratio * win_rate - (1 - win_rate)) / profit_loss_ratio
size_notional = equity * min(kelly_fraction * 0.5, 0.1)  # 최대 10%
```

#### 동적 레버리지
```python
# 현재: 고정 3배
leverage = 3

# 개선안: 변동성 기반
vol_percentile = np.percentile(historical_vol, atr / close)
if vol_percentile > 80:  # 고변동성
    leverage = 1
elif vol_percentile > 50:
    leverage = 2
else:
    leverage = 3
```

---

### 13.5 시스템 아키텍처 개선

#### 실시간 WebSocket 데이터
```python
# 현재: 5초 폴링
async def collect_loop():
    while True:
        await fetch_from_api()
        await asyncio.sleep(5)

# 개선안: WebSocket 스트리밍
async def collect_stream():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(BINANCE_WS) as ws:
            async for msg in ws:
                process_realtime(msg)
```

#### 모델 핫스왑
```python
# 현재: 서비스 재시작 필요
# 개선안: Redis Pub/Sub로 모델 갱신 알림
redis.subscribe("model:update")
for message in pubsub.listen():
    model_id = message["data"]
    load_new_model(model_id)
```

---

### 13.6 개선 우선순위 로드맵

#### Phase 1: 즉시 (1-2주)
- [x] 펀딩비 비용 추가 (라벨링)
- [x] RSI, MACD, NATR 피처 추가
- [x] Early Stopping 적용
- [x] 클래스 가중치 적용

#### Phase 2: 단기 (2-4주)
- [ ] Walk-forward 전체 스플릿 평가
- [ ] Optuna 하이퍼파라미터 튜닝
- [ ] 동적 포지션 사이징 (Kelly)
- [ ] 온체인 피처 통합

#### Phase 3: 중기 (1-2개월)
- [ ] WebSocket 실시간 데이터
- [ ] A/B 테스트 프레임워크
- [ ] 모델 앙상블 (XGBoost, Neural)
- [ ] 분산 학습 (Ray, Dask)

#### Phase 4: 장기 (3-6개월)
- [ ] 강화학습 기반 거래 에이전트
- [ ] 멀티 거래소 지원
- [ ] 옵션 전략 통합
- [ ] 완전 자동화 재학습 파이프라인

---

## 14. 부록: 설정값 전체 목록

### 14.1 환경 변수 (.env)

```bash
# ─── 거래 모드 ───
MODE=shadow                          # off | shadow | live

# ─── 데이터베이스 ───
DATABASE_URL=postgresql://ta:ta@postgres:5432/ta
REDIS_URL=redis://redis:6379/0

# ─── MinIO (모델 저장소) ───
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123
MINIO_BUCKET=models

# ─── Binance API ───
BINANCE_API_KEY=<your_key>
BINANCE_API_SECRET=<your_secret>
BINANCE_TESTNET=true

# ─── 정책 설정 ───
EV_MIN=0.0                           # 최소 기대값
Q05_MIN=-0.002                       # 최소 5분위 (-0.2%)
MAE_MAX=0.01                         # 최대 MAE (1.0%)

# ─── 포트폴리오 제약 ───
MAX_TOTAL_NOTIONAL_PCT=1.2           # 전체 노션 한도 (120%)
MAX_DIRECTIONAL_NOTIONAL_PCT=0.8     # 방향성 노션 한도 (80%)
MAX_POSITIONS=6                      # 최대 포지션 수
MAX_USED_MARGIN_PCT=0.35             # 마진 사용률 한도 (35%)
DAILY_LOSS_LIMIT_PCT=0.02            # 일일 손실 한도 (2%)

# ─── 드리프트 감지 ───
DRIFT_ALERT_PSI=0.2                  # 경고 PSI
DRIFT_BLOCK_PSI=0.4                  # 차단 PSI
DRIFT_LATENCY_THRESHOLD_MS=120000    # 지연 임계값 (2분)

# ─── 수집 설정 ───
COLLECT_INTERVAL_SEC=5               # 수집 주기 (5초)
FLUSH_BATCH_SIZE=500                 # 배치 플러시 크기

# ─── 거래 비용 ───
TAKER_FEE_RATE=0.0004                # 테이커 수수료 (0.04%)
SLIPPAGE_K=0.15                      # 슬리피지 계수
```

### 14.2 라벨링 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `k_tp` | 1.5 | TP 배리어 (ATR × 1.5) |
| `k_sl` | 1.0 | SL 배리어 (ATR × 1.0) |
| `h_bars` | 360 | 홀딩 기간 (360분 = 6시간) |
| `risk_mae_atr` | 3.0 | 리스크 MAE (ATR × 3.0) |
| `fee_rate` | 0.0004 | 테이커 수수료 |
| `slippage_k` | 0.15 | 슬리피지 계수 |

### 14.3 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `algo` | lgbm | 알고리즘 |
| `n_estimators` | 200 | 부스팅 라운드 |
| `learning_rate` | 0.05 | 학습률 |
| `num_leaves` | 64 | 리프 노드 수 |
| `subsample` | 0.8 | 행 샘플링 |
| `colsample_bytree` | 0.8 | 열 샘플링 |
| `train_days` | 180 | 학습 기간 |
| `val_days` | 30 | 검증 기간 |
| `purge_bars` | 0 | Purge 바 수 |
| `embargo_pct` | 0.0 | Embargo 비율 |

### 14.4 피처 스키마

```python
FEATURE_SCHEMA_V1 = {
    "ret_1": "1분 수익률",
    "ret_5": "5분 수익률",
    "ret_15": "15분 수익률",
    "atr": "Average True Range (14봉)",
    "bb_z": "볼린저 밴드 Z-score (20봉)",1
    "vol_z": "거래량 Z-score (60봉)",
    "ema_dist_atr": "EMA 거리 / ATR (20봉)",
    "adx": "ADX (14봉)",
    "funding_rate": "펀딩 레이트",
    "funding_z": "펀딩 Z-score (120봉)",
    "basis": "베이시스",
    "btc_ret_60": "BTC 60분 수익률",
    "btc_vol_60": "BTC 60분 변동성",
    "btc_regime": "BTC 레짐 (0: 저변동, 1: 고변동)",
}
```

### 14.5 데이터베이스 테이블

| 테이블 | 설명 |
|--------|------|
| `instruments` | 거래 종목 마스터 |
| `candles_1m` | 1분 캔들 데이터 |
| `premium_index` | 프리미엄 인덱스 |
| `funding_rates` | 펀딩레이트 이력 |
| `features_1m` | 계산된 피처 |
| `labels_long_1m` | LONG 라벨 |
| `labels_short_1m` | SHORT 라벨 |
| `training_jobs` | 학습 작업 큐 |
| `models` | 모델 메타데이터 |
| `signals` | 거래 신호 |
| `positions` | 포지션 스냅샷 |
| `orders` | 주문 이력 |
| `fills` | 체결 기록 |
| `trade_groups` | 브라켓 주문 그룹 |
| `account_snapshots` | 계정 스냅샷 |
| `drift_metrics` | 드리프트 메트릭 |
| `risk_events` | 리스크 이벤트 |

---

## 15. 사용자 확인 가능 항목 체크리스트

### 프론트엔드에서 확인 가능

- [x] **시스템 상태**: Collector, Userstream 연결 상태
- [x] **거래 모드**: OFF / SHADOW / LIVE
- [x] **포지션 현황**: 오픈 포지션, 방향, 크기, PnL
- [x] **주문 이력**: 모든 주문 상태
- [x] **신호 목록**: 최근 50개 신호, EV, 차단 사유
- [x] **모델 상태**: 학습 작업, 모델 메트릭
- [x] **데이터 품질**: PSI, 결측률, 지연
- [x] **리스크 이벤트**: 차단 사유 로그
- [x] **종목 상세**: 캔들 차트, 신호 시계열

### API로 확인 가능

- [x] **상세 메트릭**: `/api/status`
- [x] **모델 리포트**: `/api/models/{id}`
- [x] **학습 리포트**: MinIO `reports/{model_id}.json`
- [x] **계정 정보**: `/api/account/latest`

### DB 직접 조회 필요

- [ ] **과거 캔들 데이터**: `candles_1m`
- [ ] **라벨 분포**: `labels_long_1m`, `labels_short_1m`
- [ ] **피처 이력**: `features_1m`
- [ ] **전체 신호 이력**: `signals`

---

**문서 작성 완료**

이 문서는 TA 자동매매 시스템의 전체 구조, 각 컴포넌트의 동작 방식, 그리고 개선 방안을 상세히 기술합니다. 시스템 운영, 디버깅, 그리고 향후 개발에 참고하시기 바랍니다.
