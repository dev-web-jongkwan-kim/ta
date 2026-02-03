좋아. 아래는 `ta.md`를 **바이낸스 USDT-M 선물 기준**으로 “운영급(Production-grade) v2”로 **재작성한 명세서**야.
네가 말한 “코딩만 하면 되는 수준”을 목표로, 내가 앞서 지적한 **수정 필요 10개를 전부 반영**해서:

* **스코프(v1.0)에서 반드시 되는 것** / **v2 확장**을 명확히 분리
* 각 서비스별 **폴더 트리, 책임, 핵심 클래스/함수 시그니처(스텁)**
* **DB 스키마(DDL)**, **API DTO**, **프론트 페이지/컴포넌트**, **잡 오케스트레이션**
* 선물 실거래에서 핵심인 **STOP_MARKET SL / TAKE_PROFIT_MARKET TP / reduceOnly / naked 금지 / kill-switch**
* “학습-백테스트-실거래” 괴리 줄이기 위한 **체결 시뮬레이터 규칙 고정**
* “전략이 아니라 학습”을 유지하면서도, 실전 자동매매가 되도록 **정책/리스크 엔진을 하드 가드**로 설계

까지 포함했어.

---

# Binance Futures Supervised Trading System Spec v2 (Codex-ready)

## 0) 목표/원칙

### 0.1 목표

* 상위 50개 심볼(USDT-M Perp)에 대해:

    * 실시간 데이터 수집(1m/5m), Mark/Funding/OI(가능하면)
    * 피처 생성(온라인/오프라인 동일 코드)
    * 이벤트 라벨링(Triple-Barrier + 비용/펀딩 반영)
    * 지도학습 모델(수익/하방리스크/MAE/보유시간 등 멀티타깃)
    * 실시간 추론 → EV(Long/Short) 계산
    * 정책(최소 룰) + 리스크(하드 가드) 통과 시 자동 주문
    * 프론트에서 “지금 뭐가 어떻게 되고 있는지”를 **로그가 아니라 지표/차트/상태**로 확인

### 0.2 운영 원칙(선물 생존)

* **Isolated + One-way** 기본
* 레버리지: 기본 3~5x (레짐/유동성 따라 상한)
* **SL은 반드시 STOP_MARKET (reduceOnly)**
* TP는 TAKE_PROFIT_MARKET(기본) + 필요 시 분할청산은 LIMIT 옵션
* “보호주문 설치 실패(naked)”는 즉시 강제 청산
* user stream 끊기면 거래 OFF (kill-switch)

---

# 1) v2에서 반영된 “10개 수정 포인트” (명세에 녹여둠)

1. 데이터 범위 v1 축소: orderbook/trades는 v2 이후(또는 top10만)
2. DB 적재 배치화(메시지당 commit 금지)
3. 심볼 표준화(내부 `BTCUSDT` 단일키 + 어댑터 변환)
4. 라벨 검증 체계(purged/embargo + time-alignment unit test)
5. 모델 타깃 확장(pwin 단독 금지 → E[return] + 하방리스크 + MAE + hold)
6. 평가지표 교체(accuracy 중심 금지 → PF/MDD/tail loss/expectancy)
7. 주문 타입 수정(SL STOP_MARKET 고정, TP MARKET류 기본)
8. 포트폴리오 리스크 하드가드(동시 포지션/총 노출/상관 제한)
9. Shadow 모드(실주문 전 실시간 가상체결/의사결정 검증)
10. Drift/데이터품질 UI(결측/지연/피처 분포변화 최상단 노출)

---

# 2) 시스템 구성(모노레포)

## 2.1 폴더 트리

```text
repo/
  apps/
    api/                      # FastAPI: REST + WS Gateway, Auth, Admin
    web/                      # Next.js + TS + Tailwind + shadcn/ui
  services/
    collector/                # market data ingest (REST+WS)
    userstream/               # listenKey 유지 + ORDER/ACCOUNT events
    normalizer/               # SYMBOL canonicalization + data normalization
    features/                 # feature compute (online/offline same)
    labeling/                 # vectorized triple-barrier labeling
    training/                 # train + purged CV + walk-forward + reports
    registry/                 # model registry + artifact store
    inference/                # real-time inference (prod model)
    policy/                   # decision from model outputs
    risk/                     # hard constraints (account + portfolio)
    execution/                # live trading (orders + state machines)
    simulator/                # execution+fill simulator for backtest & shadow
    backtest/                 # portfolio backtest using simulator
    monitoring/               # metrics + health checks + drift compute
  packages/
    common/
      config.py
      types.py                # Pydantic models / dataclasses
      symbol_map.py
      time.py
      math.py
      db.py
      bus.py                  # Redis Streams or PubSub
  infra/
    docker-compose.yml
    migrations/               # Alembic
  scripts/
    init_universe.py
    bootstrap_history.py
  .env.example
  README.md
```

---

# 3) 인프라(로컬에서 바로 실행)

## 3.1 docker-compose 구성

* Postgres(+Timescale 권장) / Redis / MinIO / API / workers / web
* **중요**: 실시간 워커와 배치 워커 분리(학습/라벨링이 API 프로세스 막지 않게)

### 워커 종류

* `worker_realtime`: collector + userstream + features(online) + inference + policy + risk + execution + ws push
* `worker_batch`: labeling + training + backtest + drift(offline) + report

---

# 4) 심볼 표준화(무조건 구현)

## 4.1 내부 표준 심볼

* 내부 표준키: `"BTCUSDT"` (대문자, 슬래시 없음)

## 4.2 어댑터 변환 규칙

* WS stream key: `"btcusdt"` (소문자)
* REST symbol param: `"BTCUSDT"`
* (ccxt 사용 시) market symbol: `"BTC/USDT:USDT"` 등 거래소별 포맷 → 내부키로 매핑

## 4.3 구현(필수 파일)

* `packages/common/symbol_map.py`

    * `to_internal(symbol: str, source: Literal["rest","ws","ccxt"]) -> str`
    * `to_ws(internal: str) -> str`
    * `to_rest(internal: str) -> str`
    * `validate_internal(internal: str) -> None`

---

# 5) 데이터 수집(collector/userstream) — v1 범위 고정

## 5.1 v1 데이터(필수)

* 1m Kline OHLCV (USDT-M futures)
* Mark price + next funding time + funding rate snapshot
* Funding rate history(주기적으로)
* Account snapshot / Position snapshot(주기 + userstream)
* (가능하면) Open Interest(있으면 사용, 없으면 v2)

> trades/orderbook는 v1에서 제외하거나 “top10 심볼 + 1초 1회 snapshot”로 제한.

## 5.2 적재 방식(배치 flush 강제)

* collector는 수집한 레코드를 Queue에 넣고,
* writer가 `flush_interval_ms=1000~5000`로 묶어서 bulk insert/upsert

---

# 6) DB 스키마(v2) — JSONB 단일 피처 + 핵심 컬럼 일부(운영/쿼리용)

> 피처는 “스키마 버전 관리”가 핵심이라, v1은 JSONB로 단순화하고
> 운영상 자주 보는 값(atr, funding_z 등)만 컬럼으로 뽑아도 됨. (아래는 혼합형)

## 6.1 필수 테이블(DDL 핵심)

```sql
CREATE TABLE instruments (
  symbol TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'active',
  liquidity_tier TEXT NOT NULL DEFAULT 'A',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE candles_1m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE premium_index (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  mark_price DOUBLE PRECISION NOT NULL,
  index_price DOUBLE PRECISION,
  last_price DOUBLE PRECISION,
  last_funding_rate DOUBLE PRECISION,
  next_funding_time TIMESTAMPTZ,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE funding_rates (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  funding_rate DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (symbol, ts)
);

-- features: jsonb + schema_version + 일부 핵심 컬럼(옵션)
CREATE TABLE features_1m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  schema_version INT NOT NULL,
  features JSONB NOT NULL,
  atr DOUBLE PRECISION,
  funding_z DOUBLE PRECISION,
  btc_regime INT,
  PRIMARY KEY (symbol, ts)
);

-- labels: long/short 각각 (triple barrier)
CREATE TABLE labels_long_1m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  spec_hash TEXT NOT NULL,
  y SMALLINT NOT NULL,                 -- +1 TP, -1 SL/RISK, 0 TIME
  ret_net DOUBLE PRECISION NOT NULL,
  mae DOUBLE PRECISION NOT NULL,
  mfe DOUBLE PRECISION NOT NULL,
  time_to_event_min INT NOT NULL,
  PRIMARY KEY (symbol, ts, spec_hash)
);

CREATE TABLE labels_short_1m (... same ...);

CREATE TABLE training_jobs (
  job_id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  progress DOUBLE PRECISION NOT NULL DEFAULT 0,
  config JSONB NOT NULL,
  report_uri TEXT,
  error TEXT
);

CREATE TABLE models (
  model_id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  algo TEXT NOT NULL,
  feature_schema_version INT NOT NULL,
  label_spec_hash TEXT NOT NULL,
  train_start TIMESTAMPTZ NOT NULL,
  train_end TIMESTAMPTZ NOT NULL,
  metrics JSONB NOT NULL,
  artifact_uri TEXT NOT NULL,
  is_production BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE signals (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  model_id UUID NOT NULL REFERENCES models(model_id),
  ev_long DOUBLE PRECISION NOT NULL,
  ev_short DOUBLE PRECISION NOT NULL,
  er_long DOUBLE PRECISION,            -- expected return
  er_short DOUBLE PRECISION,
  q05_long DOUBLE PRECISION,           -- downside quantile proxy
  q05_short DOUBLE PRECISION,
  e_mae_long DOUBLE PRECISION,
  e_mae_short DOUBLE PRECISION,
  e_hold_long_min INT,
  e_hold_short_min INT,
  decision TEXT NOT NULL,              -- LONG/SHORT/FLAT
  size_notional DOUBLE PRECISION,
  leverage INT,
  sl_price DOUBLE PRECISION,
  tp_price DOUBLE PRECISION,
  block_reason_codes JSONB,            -- why blocked by risk/policy
  explain JSONB,                       -- top features / shap
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE account_snapshots (
  ts TIMESTAMPTZ PRIMARY KEY,
  equity DOUBLE PRECISION NOT NULL,
  wallet_balance DOUBLE PRECISION NOT NULL,
  unrealized_pnl DOUBLE PRECISION NOT NULL,
  available_margin DOUBLE PRECISION NOT NULL,
  used_margin DOUBLE PRECISION NOT NULL,
  margin_ratio DOUBLE PRECISION NOT NULL
);

CREATE TABLE positions (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  side TEXT NOT NULL,                   -- LONG/SHORT
  amt DOUBLE PRECISION NOT NULL,
  entry_price DOUBLE PRECISION,
  mark_price DOUBLE PRECISION,
  leverage INT,
  margin_type TEXT,
  liquidation_price DOUBLE PRECISION,
  notional DOUBLE PRECISION,
  unrealized_pnl DOUBLE PRECISION,
  PRIMARY KEY (symbol, ts, side)
);

CREATE TABLE orders (
  order_id BIGINT PRIMARY KEY,
  client_order_id TEXT,
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  side TEXT NOT NULL,
  type TEXT NOT NULL,
  status TEXT NOT NULL,
  reduce_only BOOLEAN NOT NULL,
  price DOUBLE PRECISION,
  stop_price DOUBLE PRECISION,
  qty DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE fills (
  trade_id BIGINT PRIMARY KEY,
  order_id BIGINT NOT NULL REFERENCES orders(order_id),
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  price DOUBLE PRECISION NOT NULL,
  qty DOUBLE PRECISION NOT NULL,
  fee DOUBLE PRECISION NOT NULL,
  fee_asset TEXT,
  realized_pnl DOUBLE PRECISION,
  ts TIMESTAMPTZ NOT NULL
);

CREATE TABLE risk_events (
  ts TIMESTAMPTZ NOT NULL,
  event_type TEXT NOT NULL,
  symbol TEXT,
  severity INT NOT NULL,
  message TEXT NOT NULL,
  details JSONB,
  PRIMARY KEY (ts, event_type, COALESCE(symbol,''))
);

CREATE TABLE drift_metrics (
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  schema_version INT NOT NULL,
  psi DOUBLE PRECISION,
  missing_rate DOUBLE PRECISION,
  latency_ms DOUBLE PRECISION,
  PRIMARY KEY (symbol, ts, schema_version)
);
```

---

# 7) 피처 생성(features) — 온라인/오프라인 동일 코드

## 7.1 피처 스키마 v1 (JSONB keys)

필수 키:

* `ret_1`, `ret_5`, `ret_15` (1m 기준)
* `atr` (mark 기반)
* `ema_dist_atr` (price - ema)/atr
* `adx`
* `bb_z`
* `vol_z`
* `funding_rate`, `funding_z`
* `basis` = (last - index)/index (가능하면)
* `btc_ret_60`, `btc_vol_60`, `btc_regime` (시장 레짐)

## 7.2 구현 규칙

* “미래 데이터” 절대 사용 금지: 오프라인도 `<= ts`까지만
* 동일 함수로 offline batch / online streaming 둘 다 계산 가능해야 함

### 파일/시그니처

`services/features/compute.py`

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class FeatureRow:
    symbol: str
    ts: str
    schema_version: int
    features: Dict[str, Any]
    atr: Optional[float]
    funding_z: Optional[float]
    btc_regime: Optional[int]

def compute_features_for_symbol(symbol: str, candles: "pd.DataFrame", premium: "pd.DataFrame", btc_candles: "pd.DataFrame") -> "pd.DataFrame":
    """Return features indexed by ts with JSONB dict + key summary columns (atr, funding_z, btc_regime)."""
```

---

# 8) 라벨링(labeling) — “속도/검증/누수 방지” 포함

## 8.1 라벨 정의(Triple-Barrier + 비용/펀딩 반영)

* entry: `mark_price(t)`
* TP/SL: `k_tp * ATR(t)`, `k_sl * ATR(t)`
* TIME: `H` bars
* RISK proxy(v1): `mae > risk_mae_atr * ATR(t)`이면 SL로 합침
* ret_net: 수수료 + 슬리피지 + 펀딩(보유시간 동안) 반영한 순수익률

## 8.2 **벡터화/가속 필수**

* 50코인×1년×1m는 “루프 per ts” 방식이면 죽는다.
* v1 구현 원칙:

    * 넘파이 기반으로 `high/low` window scanning을 가속
    * 또는 numba 사용
    * “모든 분 라벨링” 대신 **이벤트 샘플링 옵션** 제공(예: ATR 변화/볼륨 스파이크 때만)

### 파일/시그니처

`services/labeling/triple_barrier.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LabelSpec:
    k_tp: float
    k_sl: float
    h_bars: int
    risk_mae_atr: float
    fee_rate: float
    slippage_k: float

    def hash(self) -> str: ...

def label_direction_vectorized(
    symbol: str,
    ts_index: "np.ndarray",
    high: "np.ndarray",
    low: "np.ndarray",
    close: "np.ndarray",
    mark: "np.ndarray",
    atr: "np.ndarray",
    funding_rate: "np.ndarray",
    next_funding_ts: "np.ndarray",
    spec: LabelSpec,
    direction: int,  # +1 long, -1 short
) -> "pd.DataFrame":
    """Return y, ret_net, mae, mfe, time_to_event_min for each ts."""
```

## 8.3 **타임 얼라인먼트 유닛 테스트(필수)**

`services/labeling/tests/test_time_alignment.py`

* 임의의 ts에서 피처가 미래를 참조하지 않는지
* 라벨이 future window를 사용하되 학습 split에서 purge가 적용되는지 검증

---

# 9) 검증(Training) — Purged/Embargo + Walk-forward “명세로 고정”

## 9.1 split 규칙

* 기본: Walk-forward

    * train 180일 / val 30일 / test 30일 rolling
* purge/embargo:

    * 각 split 경계 주변 `h_bars` 만큼 purge
    * embargo(예: 5% 기간) 옵션

## 9.2 모델 타깃(수정 5번 반영)

pwin 단독 금지. 최소 3개 타깃:

* `er` : expected return (회귀, ret_net 기반)
* `q05` : downside quantile proxy (라벨 데이터에서 하위 5% return 근사 / quantile regression)
* `e_mae` : expected MAE (회귀)
* (옵션) `e_hold_min` : 기대 보유시간(회귀)
* (옵션) `p_tp_first` : TP hit probability (분류)

> 구현 난이도 고려한 v1 권장:
>
> * 모델 3개(ER, MAE, HOLD) 회귀 + TP확률 분류 1개
> * 또는 멀티타깃을 별도 모델로 나눔(운영 안정)

### 파일/시그니처

`services/training/train.py`

```python
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class TrainConfig:
    label_spec_hash: str
    feature_schema_version: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    targets: Tuple[str, ...]     # ("er_long","q05_long","e_mae_long","hold_long", ...)
    algo: str = "lgbm"
    purge_bars: int = 0
    embargo_pct: float = 0.0

def run_training_job(cfg: TrainConfig) -> Dict[str, Any]:
    """Train models, compute metrics, run walk-forward backtest, write artifacts + registry."""
```

## 9.3 평가 지표(수정 6번 반영)

모델 지표(보조):

* RMSE/MAE, calibration(분류), PSI drift

**트레이딩 지표(핵심)**:

* Profit Factor, Max Drawdown, Expectancy, Tail loss(하위 1% 손실), Turnover, 비용 분해(수수료/슬리피지/펀딩)

---

# 10) 체결 시뮬레이터(simulator) — 백테스트/섀도우에서 “동일 규칙” 사용(수정 6/9번의 핵심)

## 10.1 Fill 규칙 고정(명세)

* 진입: 다음 캔들의 open 기준(누수 방지) + 슬리피지 함수 적용
* SL/TP 트리거: **Mark price 기준** (바이낸스 설정과 일치시키기)
* SL 주문: STOP_MARKET (즉시 체결 가정 + 슬리피지)
* TP 주문: TAKE_PROFIT_MARKET (즉시 체결 가정 + 슬리피지)
* 수수료: taker 고정(보수적으로)
* 펀딩: 실제 펀딩 timestamp 경과 시 notional * funding_rate 반영

### 파일/시그니처

`services/simulator/fill.py`

```python
from dataclasses import dataclass

@dataclass
class FillModel:
    fee_rate: float
    slippage_k: float

    def slippage(self, notional: float, volume: float) -> float: ...

def simulate_trade_path(
    candles: "pd.DataFrame",
    premium: "pd.DataFrame",
    entry_ts: str,
    side: int,
    notional: float,
    sl_price: float,
    tp_price: float,
    fill: FillModel
) -> Dict[str, float]:
    """Return realized_pnl, fees, slippage, funding, exit_ts, exit_reason."""
```

---

# 11) 포트폴리오 백테스트(backtest) — 단일 포지션 금지(수정 4/8 반영)

## 11.1 포지션 구조

* `positions: dict[symbol] -> PositionState`
* 계정 레벨:

    * 총 노출, 사용 마진, 심볼별 max notional, 동시 포지션 수 제한

### 파일/시그니처

`services/backtest/portfolio.py`

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class PositionState:
    symbol: str
    side: int
    notional: float
    entry_price: float
    sl_price: float
    tp_price: float
    open_ts: str

@dataclass
class PortfolioState:
    equity: float
    used_margin: float
    positions: Dict[str, PositionState]
    daily_pnl: float
    consecutive_losses: int

def run_walkforward_backtest(...) -> Dict[str, float]:
    """Return PF, MDD, expectancy, turnover, costs breakdown."""
```

---

# 12) 실시간 추론(inference) → EV 계산(policy 입력)

## 12.1 모델 출력(권장)

각 심볼/시점에서:

* `er_long/short`
* `q05_long/short` (하방 리스크)
* `e_mae_long/short`
* `e_hold_long/short`

## 12.2 EV 산식(수정 5/6 반영)

* `EV = er - cost_est - risk_penalty`
* `risk_penalty`는 q05/MAE 기반 하드/소프트 혼합
* `cost_est`는 수수료+슬리피지+펀딩(hold 기반)

---

# 13) 정책(policy) — “최소 룰”이지만 명세로 고정

## 13.1 Decision

* `score_long = EV_long`
* `score_short = EV_short`
* `decision = argmax(score_long, score_short, 0)`
* 정책 게이트:

    * `EV_best > EV_MIN`
    * `q05_best > Q05_MIN` (하방 너무 크면 차단)
    * `e_mae_best < MAE_MAX`
    * 레짐 필터: BTC high-vol 시 size down 또는 차단

### 파일/시그니처

`services/policy/decide.py`

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PolicyConfig:
    ev_min: float
    q05_min: float
    mae_max: float
    max_positions: int

def decide(symbol: str, preds: Dict[str, float], state: Dict[str, Any], cfg: PolicyConfig) -> Dict[str, Any]:
    """Return decision + desired notional/leverage + sl/tp + reason codes."""
```

---

# 14) 리스크(risk) — 하드 가드레일(수정 8/10 반영)

## 14.1 계정 레벨 하드가드

* `used_margin/equity <= MAX_USED_MARGIN_PCT`
* daily loss limit 도달 시:

    * `TRADING_ENABLED=false` + `close all positions` + risk_event 기록
* userstream down / data latency high / order failure spike:

    * 즉시 kill-switch

## 14.2 포트폴리오 레벨 하드가드(필수)

* `max_open_positions`
* `max_total_notional` (equity 대비)
* `max_same_direction_exposure` (롱/숏 한쪽 쏠림 제한)
* 상관 제한(간이 버전):

    * “BTC 베타 근사”로 알트 노출 제한
    * 또는 심볼을 tier로 묶어 tier별 상한

### 파일/시그니처

`services/risk/guard.py`

```python
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class RiskConfig:
    max_used_margin_pct: float
    daily_loss_limit_pct: float
    max_positions: int
    max_total_notional_pct: float
    max_directional_notional_pct: float

def check_risk(action: Dict[str, Any], portfolio: Dict[str, Any], account: Dict[str, Any], cfg: RiskConfig) -> List[str]:
    """Return list of block_reason_codes; empty means allowed."""
```

---

# 15) 실행(execution) — 주문 타입 고정(수정 7), naked 금지

## 15.1 주문 플로우(필수)

1. Entry: MARKET (taker)
2. Entry fill 확인
3. SL 설치: STOP_MARKET, reduceOnly=true
4. TP 설치: TAKE_PROFIT_MARKET, reduceOnly=true
5. **보호주문 설치 실패** → 즉시 시장가 청산 + risk_event

## 15.2 포지션 동기화

* userstream의 ORDER_TRADE_UPDATE, ACCOUNT_UPDATE가 끊길 수 있으니
* 주기적으로 REST snapshot으로 reconciliation(예: 30~60초)

### 파일/시그니처

`services/execution/trader.py`

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OrderRequest:
    symbol: str
    side: str           # BUY/SELL
    notional: float
    leverage: int
    sl_price: float
    tp_price: float

def place_entry_with_brackets(req: OrderRequest) -> Dict[str, Any]:
    """Place entry market + attach SL/TP. Must be atomic in behavior; if fail, flatten."""
```

---

# 16) Shadow 모드(수정 9) — 실거래 전 필수 운영 단계

## 16.1 모드 정의

* `MODE=shadow`:

    * 실시간으로 features → inference → policy → risk는 동일 실행
    * execution은 “시뮬레이터에만 반영” (가상체결)
    * UI에 “Shadow PnL / Shadow Positions”로 표시
* `MODE=live`: 실제 주문

## 16.2 전환 조건(DoD)

* Shadow 2주 이상:

    * 데이터 결측/지연 < 임계
    * 주문 실패율(가상) < 임계
    * MDD/손실폭이 리스크 한도 내
    * 드리프트 경고 빈도 관리 가능

---

# 17) Drift/데이터 품질(수정 10) — UI 최상단에 노출

## 17.1 Drift metrics

* PSI(피처 분포 변화)
* missing_rate
* latency_ms(캔들 지연, feature 지연, inference 지연)
* outlier_rate(급격한 스파이크)

### 파일/시그니처

`services/monitoring/drift.py`

```python
def compute_psi(current: "pd.Series", reference: "pd.Series") -> float: ...
def compute_missing_rate(df: "pd.DataFrame") -> float: ...
```

---

# 18) 잡 오케스트레이션(학습/라벨링/리포트) — API 프로세스와 분리(수정 10 관련 운영)

## 18.1 큐/잡

* Celery/RQ/Arq 중 1개 고정(권장: Celery + Redis)
* training_jobs 테이블로 상태/진행률 UI 제공

### 작업 종류

* `job:bootstrap_history`
* `job:label_recent`
* `job:train_model`
* `job:walkforward_backtest`
* `job:compute_drift`
* `job:promote_model`

---

# 19) API (FastAPI) — 프론트가 “로그가 아니라 상태”를 보는 계약

## 19.1 REST endpoints (최소 필수)

* `GET /api/status`

    * latest timestamps, ws connected, queue lag, error rates
* `GET /api/universe`
* `GET /api/account/latest`
* `GET /api/positions/latest`
* `GET /api/orders?from&to`
* `GET /api/signals/latest?sort=ev&limit=50`
* `GET /api/symbol/{symbol}/series?from&to`

    * candles(1m), mark/last, signals, position lines
* `GET /api/training/jobs`
* `GET /api/models`
* `GET /api/models/{id}`
* `POST /api/models/{id}/promote`
* `GET /api/data-quality/summary`
* `GET /api/risk/state`
* `POST /api/trading/toggle` (shadow/live/off)
* `POST /api/settings` (policy/risk/label preset)

## 19.2 WebSocket channels

* `status_updates`
* `account_updates`
* `position_updates`
* `order_updates`
* `signal_updates`
* `risk_events`
* `training_updates`

## 19.3 DTO (Pydantic)

`packages/common/types.py`

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class StatusDTO(BaseModel):
    collector_ok: bool
    userstream_ok: bool
    latest_candle_ts: Optional[str]
    latest_feature_ts: Optional[str]
    latest_signal_ts: Optional[str]
    queue_lag: Dict[str, float]
    latency_ms: Dict[str, float]

class SignalDTO(BaseModel):
    symbol: str
    ts: str
    ev_long: float
    ev_short: float
    er_long: Optional[float]
    er_short: Optional[float]
    q05_long: Optional[float]
    q05_short: Optional[float]
    decision: str
    size_notional: Optional[float]
    leverage: Optional[int]
    sl_price: Optional[float]
    tp_price: Optional[float]
    block_reasons: List[str] = []
    explain: Dict[str, Any] = {}
```

---

# 20) 프론트(Next.js) — “세련/깔끔 + 상태 중심”

## 20.1 페이지(필수)

1. `/dashboard`

* System Health(데이터 지연/결측/WS 연결/큐 적체)
* Margin Overview(Equity/Used/Avail/Margin ratio)
* Drift Alerts(PSI/결측/latency)
* Top Signals(50심볼 테이블: EV/decision/차단사유)

2. `/signals`

* 필터( EV 상위 / 펀딩 극단 / 변동성 상위 / 차단만 보기 )
* Symbol drawer: “왜 진입/왜 차단?”

3. `/symbol/[symbol]`

* 가격차트(Last/Mark 토글) + 청산가/SL/TP 라인
* EV/ER/q05/MAE/hold 타임시리즈
* Explain panel(top features)
* Live state(현재 포지션/주문/체결)

4. `/positions`

* 포지션 리스트 + 위험 Top5 (청산거리/마진사용)
* 비용(수수료/슬리피지/펀딩) 누적

5. `/orders`

* 주문/체결 타임라인 + 실패율 + 슬리피지 실제 vs 예상

6. `/training`

* job 진행률(퍼센트), 결과 리포트(워크포워드 PF/MDD)
* 모델 비교, promote 버튼(승격/롤백)

7. `/risk`

* 가드레일 설정값 + 현재 차단 원인 + risk event 타임라인

8. `/data-quality`

* 심볼별 결측/지연 히트맵 + 이상치

9. `/settings`

* 모드(shadow/live/off), 유니버스, 라벨/정책/리스크 프리셋

## 20.2 컴포넌트 트리(코딩 단위)

```text
components/
  cards/
    SystemHealthCard.tsx
    MarginCard.tsx
    DriftCard.tsx
    TopSignalsTable.tsx
    RiskBlockersCard.tsx
  charts/
    PriceChart.tsx
    SignalSeriesChart.tsx
    CostBreakdownChart.tsx
  panels/
    ExplainPanel.tsx
    LiveOpsPanel.tsx
  tables/
    PositionsTable.tsx
    OrdersTable.tsx
    TrainingJobsTable.tsx
    ModelsTable.tsx
```

---

# 21) 설정 프리셋(초기값, 안전 우선)

## 21.1 Label preset

* `k_tp=1.5`, `k_sl=1.0`, `h_bars=360`(1m 기준 6시간)
* `risk_mae_atr=3.0`
* `fee_rate=0.0004`(taker 보수 가정, 실제는 설정에서 변경)
* `slippage_k=0.15`

## 21.2 Policy preset

* `EV_MIN = 0`
* `Q05_MIN = -0.8 * ATR` (너무 큰 하방꼬리 차단)
* `MAE_MAX = 1.2 * ATR`
* `MAX_POSITIONS = 6`

## 21.3 Risk preset

* `MAX_USED_MARGIN_PCT=0.35`
* `MAX_TOTAL_NOTIONAL_PCT=1.2` (equity 대비 총 노출 상한)
* `MAX_DIRECTIONAL_NOTIONAL_PCT=0.8`
* `DAILY_LOSS_LIMIT_PCT=0.02`

---

# 22) 런타임 루프(실시간) — 구현 순서(고정)

1. 최신 캔들/mark/funding ingest (queue)
2. writer batch flush → DB
3. feature compute(온라인) → DB + bus publish
4. inference(prod model) → signals 저장
5. policy decide(차단사유 코드 생성)
6. risk check(하드 가드) → 허용/차단
7. (shadow) simulator 반영 / (live) execution 주문
8. userstream 이벤트 반영(orders/fills/positions/account)
9. api ws push로 UI 업데이트

---

# 23) DoD(“제대로 돌아간다” 기준) — 기능 + 운영

## 23.1 기능 DoD

* 데이터 끊김 없이 24h 수집/저장(결측률 < 0.1%)
* 피처/라벨 생성 재현 가능(schema/spec hash로 고정)
* 학습 job → 모델 레지스트리 등록 → 프로덕션 승격까지 UI로 수행
* 실시간 신호(50심볼) 갱신 + 상세 drill-down
* 주문/체결/포지션 상태가 실시간으로 UI 반영

## 23.2 운영 DoD

* Shadow 2주 운영에서:

    * MDD/일손실이 가드레일 내
    * drift/결측/지연 알람이 UI에 즉시 반영
    * “차단 사유”가 명확히 노출(왜 진입 안 했는지)
* Live 모드에서:

    * naked 포지션 0회(보호주문 실패 시 즉시 청산)
    * userstream down 시 kill-switch 작동 확인
    * SL 미체결 방지(STOP_MARKET 강제)

---

# 24) Codex 전달용 “구현 지시문” (짧게 복붙용)

```text
Rewrite ta.md into production-grade v2 for Binance USDT-M futures.
Must implement: batching DB writes, symbol canonicalization (internal BTCUSDT), vectorized labeling, purged/embargo validation, multi-target models (expected return + downside quantile proxy + MAE + hold), portfolio-level backtest (multi positions), execution order types fixed (STOP_MARKET SL reduceOnly, TAKE_PROFIT_MARKET TP reduceOnly), portfolio risk hard guards, shadow mode, drift & data quality UI.
Provide FastAPI REST+WS APIs, Next.js dashboard with state-first UI (not logs).
Use workers separation (realtime vs batch) with a job queue and progress tracking.
Simulator fill rules must match live assumptions (next candle entry, mark-price triggers, funding timestamps).
```

---

## 다음 액션(너는 질문 답 안 해도 됨, 바로 진행 가능)

* Codex에 이 문서를 그대로 던지고,
* 구현은 **Phase 1: Shadow**부터 시작해:

    1. 수집/저장/피처/신호/UI(대시보드)
    2. 라벨/학습/레지스트리/백테스트
    3. 시뮬레이터로 Shadow 운용
    4. Live 전환(킬스위치/보호주문/리스크 이벤트 검증)

원하면, 다음 메시지에서 **"각 서비스별 실제 파일 단위 스텁 코드(클래스/함수 몸체는 TODO, 하지만 import/타입/호출 순서/에러 처리 뼈대 포함)"**까지 만들어서, Codex가 *거의 그대로 생성/완성*하게 더 내려줄게.

---

# 25) 강화학습(RL) 하이브리드 시스템

## 25.1 RL 아키텍처 개요

기존 지도학습(LightGBM) 모델의 예측값을 RL 에이전트의 **상태 입력**으로 사용하는 하이브리드 구조:

```
시장 데이터 → 피처 생성 → LightGBM 예측 → RL 에이전트 → 최종 결정
                              ↓
                     (er_long, q05_long, e_mae_long, e_hold_long)
```

**장점**:
- 기존 모델의 예측력 + RL의 시퀀셜 의사결정 능력 결합
- 포지션 관리, 타이밍, 리스크 조절을 RL이 학습
- 시장 상황에 따른 적응적 전략

## 25.2 환경 설계 (Gymnasium)

### 상태 공간 (17차원)

```python
observation_space = spaces.Box(low=-inf, high=inf, shape=(17,), dtype=float32)
```

| 카테고리 | 피처 | 설명 |
|---------|------|------|
| 모델 예측 (4) | er_long | 기대 수익률 |
| | q05_long | 하방 리스크 (5% 분위수) |
| | e_mae_long | 기대 MAE |
| | e_hold_long | 기대 보유시간 |
| 시장 지표 (10) | ret_1, ret_5 | 최근 수익률 |
| | rsi, macd_z, bb_z | 기술적 지표 |
| | vol_z, atr, adx | 변동성/추세 강도 |
| | funding_z, btc_regime | 펀딩/레짐 |
| 포지션 상태 (3) | position | 현재 포지션 (-1~1) |
| | position_time | 보유 시간 (정규화) |
| | unrealized_pnl | 미실현 손익률 |

### 액션 공간 (4 이산)

```python
action_space = spaces.Discrete(4)
# 0: HOLD - 현재 상태 유지
# 1: LONG - 롱 포지션 진입 (숏 보유 시 청산 후 롱)
# 2: SHORT - 숏 포지션 진입 (롱 보유 시 청산 후 숏)
# 3: CLOSE - 포지션 청산
```

### 보상 함수

```python
reward = realized_pnl - trading_cost - funding_cost
```

- 실현 손익 기반 (미실현 손익은 보상에 미포함)
- 거래 비용 (수수료 + 슬리피지) 반영
- 펀딩 비용 (8시간마다) 반영

## 25.3 RL 에이전트 구현

### 알고리즘

- **PPO** (Proximal Policy Optimization) - 기본 권장
- **A2C** (Advantage Actor-Critic) - 대안

### 학습 설정

```python
@dataclass
class RLAgentConfig:
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01  # 탐색 장려
```

### 학습 프로세스

```bash
python scripts/train_rl.py \
  --symbols BTCUSDT,ETHUSDT \
  --train-days 90 \
  --eval-days 30 \
  --timesteps 100000 \
  --algorithm PPO
```

## 25.4 DB 스키마 (RL 전용)

```sql
-- RL 모델 테이블
CREATE TABLE rl_models (
    model_id VARCHAR(32) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    algorithm VARCHAR(20) NOT NULL DEFAULT 'PPO',
    train_start TIMESTAMPTZ NOT NULL,
    train_end TIMESTAMPTZ NOT NULL,
    metrics JSONB,  -- PF, win_rate, sharpe, max_dd 등
    model_path VARCHAR(255),
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RL 결정 기록 (모든 결정 저장)
CREATE TABLE rl_decisions (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_id VARCHAR(32) REFERENCES rl_models(model_id),
    action INTEGER NOT NULL,  -- 0-3
    action_name VARCHAR(10) NOT NULL,
    confidence FLOAT,  -- 선택된 액션의 확률
    action_probs JSONB,  -- 모든 액션 확률
    value_estimate FLOAT,  -- 가치 함수 추정
    model_predictions JSONB,  -- 입력된 LightGBM 예측값
    position_before FLOAT,
    position_after FLOAT,
    executed BOOLEAN DEFAULT FALSE,
    pnl_result FLOAT,  -- 실제 결과 (후에 업데이트)
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 25.5 API 엔드포인트

```
GET  /api/rl/models                    # RL 모델 목록
GET  /api/rl/models/{model_id}         # 모델 상세
POST /api/rl/models/{model_id}/promote # 프로덕션 승격
GET  /api/rl/decisions                 # 결정 히스토리
GET  /api/rl/decisions/stats           # 결정 통계
```

## 25.6 프론트엔드 컴포넌트

```
components/tables/
  RLModelsTable.tsx       # RL 모델 목록/관리
  RLDecisionsTable.tsx    # 실시간 결정 피드
```

**RLModelsTable 기능**:
- 모델별 성능 메트릭 (PF, Win Rate, Sharpe, Max DD)
- 액션 분포 시각화
- 프로덕션 승격 버튼

**RLDecisionsTable 기능**:
- 실시간 결정 스트림
- 액션별 통계 (24시간)
- 신뢰도 바 시각화
- 액션 확률 분포 차트

## 25.7 RL 운영 가이드

### 학습 주기

1. **초기 학습**: 90일 데이터로 기본 모델 학습
2. **재학습**: 주 1회 또는 성능 저하 시
3. **A/B 테스트**: Shadow 모드에서 신규 모델 평가 후 승격

### 평가 지표

- **Profit Factor**: 총 이익 / 총 손실
- **Win Rate**: 승리 거래 비율
- **Sharpe Ratio**: 위험 조정 수익률
- **Max Drawdown**: 최대 낙폭

### 모델 승격 기준

1. Shadow 모드에서 최소 1주 운영
2. PF > 1.2 (기존 모델 대비 개선)
3. Max DD < 10%
4. 액션 분포 합리성 (HOLD 과다 아님)

---

# 26) Universe 관리 (고정 방식)

## 26.1 고정 유니버스

동적 유니버스 업데이트 비활성화. `.env`의 `UNIVERSE` 변수로 30개 심볼 고정:

```
UNIVERSE=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,LTCUSDT,BCHUSDT,SUIUSDT,AAVEUSDT,1000PEPEUSDT,FILUSDT,AXSUSDT,ENAUSDT,ZKUSDT,XAUUSDT,XAGUSDT,HYPEUSDT,ZECUSDT,BNXUSDT,ALPHAUSDT,ZORAUSDT,TRUMPUSDT,PUMPUSDT,NEARUSDT,PAXGUSDT
```

## 26.2 금/은 포함

- XAUUSDT (금)
- XAGUSDT (은)
- PAXGUSDT (금 담보 토큰)

## 26.3 유니버스 동기화

`worker_universe` 서비스 비활성화. `universe_worker.py`는 `.env`의 UNIVERSE를 읽어 instruments 테이블에 동기화만 수행 (1시간 주기).
