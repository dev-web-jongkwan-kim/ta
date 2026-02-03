아래는 **Codex에 그대로 복붙**하면 되는 “완전 지시문(Implementation Directive v1.0)”이야.
목표는 **두 문서(ta.md + v2 spec)를 기반으로, Binance USDT-M 선물 자동매매 시스템을 Shadow→Live까지 운영급으로 완성**하는 것.
(이 지시문은 “무엇을”, “어떻게”, “어떤 품질 기준으로”, “어떤 산출물을” 만들지까지 전부 못 박아 둠)

---

# ✅ CODEX IMPLEMENTATION DIRECTIVE (Binance USDT-M Futures ML Trading System) v1.0

## 0) Inputs (반드시 읽고 기준으로 삼아라)

You are given two documents:

* **Doc A**: `ta.md` (baseline architecture + detailed code-level examples)
* **Doc B**: `v2 spec` (production-grade design and required fixes)

### Priority rule

* **Doc B overrides Doc A** wherever they conflict.
* Use **Doc A only** to fill in missing code-level details that Doc B doesn’t specify.

### Non-negotiable goal

Build a system that works **end-to-end**:

* data → features → inference → policy → risk → (shadow simulator) → dashboard
  and then can switch to **live trading** with hard safety constraints.

---

## 1) Hard Requirements (MUST IMPLEMENT EXACTLY)

These are mandatory; do not skip; do not approximate.

### 1.1 Scope limits (v1 must be achievable)

* v1 market data scope must be **OHLCV(1m and/or 5m) + mark price + funding + account/positions + optional Open Interest**.
* **No full orderbook** and **no tick trades** for all 50 symbols in v1.

    * If orderbook/trades exist in Doc A, disable or gate behind feature flag and apply only to top-N (default N=0 in v1).

### 1.2 DB write strategy (no per-message commits)

* DO NOT commit per websocket message.
* Implement buffered ingestion:

    * memory buffer → flush every 1–5 seconds OR by batch size threshold
    * bulk insert/upsert using efficient method (COPY/execute_values)
* Ensure idempotency: unique constraint on `(symbol, ts)` with upsert semantics.

### 1.3 Symbol canonicalization (must be single internal format)

* Internal symbol key must be strictly `"BTCUSDT"` format.
* Implement adapter conversions:

    * REST uses `"BTCUSDT"`
    * WS streams use `"btcusdt"` (lowercase)
    * CCXT (if used) maps to/from internal key
* Any stored DB rows MUST use internal canonical symbol only.

### 1.4 Time-series leakage prevention

* All training/validation must use **time-series safe splits**:

    * walk-forward and/or purged K-fold with embargo
* If labels use future windows (triple barrier), implement **purge window >= horizon**
* Add unit tests:

    * time alignment: features at time t never use data after t
    * split correctness: no overlap leakage across splits

### 1.5 Modeling targets (p(win)-only forbidden)

Implement supervised learning with at least:

* `ER`: expected return (net of costs or aligned to simulator)
* `Downside risk proxy`: `q05` (5th percentile), or CVaR proxy, or quantile regression
* `E[MAE]`: expected maximum adverse excursion
* `E[HoldTime]`: expected holding time (minutes) (for funding/cost estimation)
  Optional: `P(TP_first)` classification can be included but cannot be the only target.

### 1.6 Evaluation metrics (trading-first)

Primary metrics MUST include:

* Profit Factor (PF)
* Max Drawdown (MDD)
* Expectancy (avg trade EV realized)
* Tail loss (e.g., 1% worst trade losses, or CVaR)
* Turnover
* Cost breakdown: fees/slippage/funding
  Accuracy/F1/AUC are secondary.

### 1.7 Execution safety: order types fixed

For Binance USDT-M futures:

* Entry: MARKET (taker)
* Stop-loss: **STOP_MARKET** with `reduceOnly=true`
* Take-profit: default **TAKE_PROFIT_MARKET** with `reduceOnly=true`
* **Naked positions are forbidden**:

    * if SL/TP placement fails after entry fill, immediately flatten the position with market order.
* Must support Isolated margin + One-way mode baseline.
* Must implement kill-switch logic.

### 1.8 Portfolio risk hard constraints (not per-symbol only)

Implement these hard guards:

* max open positions (e.g., 6 default)
* max total notional exposure (as % of equity)
* max directional exposure (long vs short notional)
* max used margin ratio
* daily loss stop: if breached, flatten all + disable trading for rest of day
* circuit breakers: disable trading when

    * data latency spikes
    * missing candles exceed threshold
    * userstream disconnected
    * order failure rate spikes
* All blocks must return **reason codes** (for UI).

### 1.9 Shadow mode is required before live mode

Implement run modes:

* `OFF` (no orders, no shadow)
* `SHADOW` (full pipeline; execution routed only to simulator; store shadow trades)
* `LIVE` (real orders to Binance)
  Default mode must be `SHADOW`.

### 1.10 Drift & data quality dashboards top-level

Implement drift/data quality computation and UI:

* PSI on key features (or a reasonable drift proxy)
* missing rates (candles/features/signals)
* pipeline latencies (candle lag, feature lag, inference lag)
* outlier counts
  This must appear on `/dashboard` and `/data-quality`.
  All blocked actions show reason codes.

---

## 2) Technical Stack (fixed)

* Backend: Python 3.11+, FastAPI for API gateway, Pydantic DTOs
* Storage: Postgres (TimescaleDB recommended but not required), Redis, MinIO/S3 for model artifacts
* Frontend: Next.js + TypeScript + Tailwind + shadcn/ui
* Messaging: Redis Streams or PubSub (choose one and standardize)
* ML: LightGBM (preferred) or XGBoost; must support quantile regression or equivalent for downside proxy.

---

## 3) Repository Deliverables (must produce)

Deliver a runnable monorepo:

* `docker-compose.yml` that boots DB/Redis/MinIO/API/workers/web
* `apps/api` FastAPI server
* `apps/web` Next.js web UI
* `services/*` workers for realtime and batch
* `migrations` for DB schema
* `README.md` with exact run commands + env setup + mode switch + Binance testnet instructions
* Unit tests for time alignment & split leakage prevention
* End-to-end shadow mode works with live market data (no real orders)

---

## 4) Binance Integration (required endpoints & streams)

Use Binance USDT-M endpoints:

### 4.1 Market data

* Mark/Funding snapshot: `GET /fapi/v1/premiumIndex` (markPrice, lastFundingRate, nextFundingTime)
* Funding history: `GET /fapi/v1/fundingRate`
* Klines: `GET /fapi/v1/klines` and/or WS `@kline_1m`

### 4.2 Account/positions

* Position risk: `GET /fapi/v2/positionRisk` (liquidationPrice etc.)
* Leverage brackets: `GET /fapi/v1/leverageBracket`
* Set leverage: `POST /fapi/v1/leverage`
* User data stream listenKey:

    * `POST /fapi/v1/listenKey`, keepalive `PUT /fapi/v1/listenKey`
* Process user events:

    * `ORDER_TRADE_UPDATE`
    * `ACCOUNT_UPDATE`
      Implement reconnect + keepalive + fallback REST reconciliation.

---

## 5) DB Schema (must match; can extend)

Implement tables from Doc B (or harmonize with Doc A) but must include at least:

* instruments
* candles_1m (or 5m)
* premium_index
* funding_rates
* features (schema_versioned; JSONB ok)
* labels_long / labels_short with `spec_hash`
* training_jobs
* models registry with `is_production`
* signals with EV/ER/q05/MAE/Hold/decision/block_reasons/explain
* account_snapshots
* positions
* orders
* fills
* risk_events
* drift_metrics

Constraints:

* Unique `(symbol, ts)` where appropriate
* Use `DOUBLE PRECISION` for numeric performance unless precision is critical.

---

## 6) Real-time Pipeline (must implement)

On each closed candle (1m or 5m):

1. ingest candle + premiumIndex snapshot (mark/funding)
2. flush batch to DB
3. compute features (online) for all symbols with consistent time alignment
4. run inference using production model
5. compute EV_long/EV_short (costs + funding + risk penalty)
6. policy decision returns desired action and params + reason codes
7. risk check blocks/permits; logs risk_event if blocked
8. if SHADOW: simulate entry+brackets with simulator
9. if LIVE: place market entry + brackets; if bracket fails => flatten
10. write signals + actions + order/position updates; push WS updates to UI

---

## 7) Simulator (must implement; used in shadow & backtest)

Implement a fill simulator whose rules are consistent with live assumptions:

* entry occurs next candle open (avoid lookahead)
* SL/TP triggers based on mark price
* apply slippage model: function of notional and candle volume (simple is OK but consistent)
* apply taker fees
* funding applied when crossing funding timestamps using funding_rates/premium_index nextFundingTime
* output realized pnl and cost breakdown
  Simulator must support multi-symbol portfolio states.

---

## 8) Labeling (must implement; performant)

Implement triple-barrier labels:

* entry at mark price
* TP/SL barriers based on ATR(t) * k
* time barrier H bars
* risk proxy: MAE threshold (or separate risk barrier) that maps to SL class
  Compute:
* y ∈ {+1 TP, -1 SL/RISK, 0 TIME}
* ret_net (with costs & funding)
* MAE/MFE
* time_to_event
  Generate long & short labels separately.
  Must be vectorized or accelerated (no naive O(N*H) loops for full dataset).

---

## 9) Training & Validation (must implement)

* Generate training datasets from features + labels
* Perform walk-forward or purged CV with embargo
* Train models:

    * ER regression
    * downside proxy (q05) regression (quantile regression preferred)
    * MAE regression
    * hold time regression
* Save artifacts to MinIO and register in DB
* Compute full backtest (portfolio) using simulator
* Produce report JSON for UI: metrics, curves, regime splits, cost breakdown
* Implement `promote model` endpoint: sets is_production true and others false.

---

## 10) Policy + Risk (must implement)

Policy:

* chooses LONG/SHORT/FLAT using EV and gates (EV_min, q05_min, mae_max, regime filter)
* outputs notional/leverage/SL/TP targets

Risk:

* hard blocks actions and returns reason codes:

    * MARGIN_LIMIT, DAILY_STOP, MAX_POSITIONS, MAX_EXPOSURE, MAX_DIRECTIONAL_EXPOSURE,
      USERSTREAM_DOWN, DATA_STALE, ORDER_FAILURE_SPIKE, DRIFT_ALERT, etc.
* logs risk_events
* enforces kill-switch if critical events triggered
* ensures bracket placement; forbids naked.

---

## 11) Frontend UX Requirements (must implement)

Build a clean UI (no log dumps as primary view).

### Pages

* `/dashboard`:

    * system health cards (latency/missing/ws connectivity)
    * margin overview (equity/used/avail/margin ratio)
    * drift/data quality summary (PSI/missing/latency/outliers)
    * top signals table (50 symbols with EV/decision/block reasons)
* `/signals`:

    * sortable/filterable signals table, symbol drawer details
* `/symbol/[symbol]`:

    * price chart (last/mark toggle) with SL/TP/liquidation lines
    * EV/ER/q05/MAE/hold time series charts
    * explain panel (top features / shap or feature importance)
    * live ops panel (current position, open orders)
* `/positions`, `/orders`, `/training`, `/risk`, `/data-quality`, `/settings`

UI must show:

* for every blocked action: reason codes + human-readable explanation
* shadow vs live mode indicator and toggle in settings (guarded)

---

## 12) Testing & Quality Gates (must pass)

### Unit tests

* time alignment test (features use only ≤ t data)
* split leakage test (no overlap across splits within purge/embargo)
* symbol mapping test
* simulator deterministic test on fixed candles

### Runtime self-checks

* if latest candle ts lags > threshold => disable trading and emit risk_event
* if userstream disconnected => disable trading and emit risk_event
* if bracket placement fails => immediate flatten and emit risk_event

---

## 13) Implementation Plan (must follow; deliver incrementally)

### Phase 1: Shadow E2E (must fully work)

* ingestion + DB batching + features + inference dummy model + policy/risk + simulator
* dashboard updates via WS
* no live orders

### Phase 2: Training + Registry + Backtest

* labeling + training + evaluation + model registry + promote
* training UI shows metrics and backtest

### Phase 3: Live trading (guarded)

* Binance execution + bracket safety + kill-switch
* mode toggle from SHADOW to LIVE requires explicit setting and shows warnings

---

## 14) Output Format Requirements

* Provide code + docker compose + migrations + README
* Provide clear instructions for:

    * environment variables
    * how to start all services
    * how to bootstrap historical data
    * how to run labeling/training jobs
    * how to run shadow mode
    * how to switch to live mode (with safety warnings)
* Do not leave unresolved TODOs in critical path (shadow E2E must run).

---

## 15) Final Acceptance Criteria

System is accepted only if:

1. Shadow mode runs end-to-end for at least 24 hours without crashing:

    * stable ingestion
    * signals generated for all active symbols
    * simulated trades recorded
    * dashboard updates in real time
2. Training job produces a registered model and backtest report.
3. Live mode safety verified:

    * bracket order placement enforced
    * naked position prevention verified
    * kill-switch triggers on userstream disconnect/data stale
4. UI clearly shows:

    * drift/data-quality
    * reasons for blocking
    * mode (OFF/SHADOW/LIVE)
    * portfolio exposure and margin state

END.

```

---

## 추가로(중요): “문서 2개 + 지시문” 전달 팁
Codex가 흔들리지 않게, **파일 첨부/붙여넣기 순서**를 이렇게 해:
1) `ta.md` (baseline)
2) 내가 준 v2 spec(짧은 명세)
3) **위 지시문**(최종 directive)

그리고 “Doc B overrides Doc A”를 지시문에 이미 박아뒀기 때문에 충돌 정리도 Codex가 하게 돼.

---

원하면, 이 지시문을 더 강력하게 만들기 위해 **“API 응답 스키마(필드 레벨) / Binance 이벤트 샘플 JSON / 마이그레이션 파일명 규칙 / 서비스별 엔트리포인트 명령어”**까지 넣어줄 수도 있어.  
하지만 지금 지시문도 이미 Codex가 "설계 해석"을 거의 못 하게 꽉 묶어둔 상태야.

---

## 16) Reinforcement Learning (RL) Hybrid System (MUST IMPLEMENT)

### 16.1 RL Architecture

Implement a hybrid system where RL agent uses supervised model predictions as state input:

```
Market Data → Features → LightGBM Predictions → RL Agent → Final Decision
```

### 16.2 Environment (Gymnasium)

Implement `TradingEnvironment` in `services/rl/environment.py`:

**State Space (17 dimensions)**:
- Model predictions (4): er_long, q05_long, e_mae_long, e_hold_long
- Market indicators (10): ret_1, ret_5, rsi, macd_z, bb_z, vol_z, atr, funding_z, btc_regime, adx
- Position state (3): position (-1 to 1), position_time (normalized), unrealized_pnl (normalized)

**Action Space (Discrete 4)**:
- 0: HOLD
- 1: LONG
- 2: SHORT
- 3: CLOSE

**Reward Function**:
```python
reward = realized_pnl - trading_cost - funding_cost
```

### 16.3 RL Agent

Implement `RLAgent` in `services/rl/agent.py`:
- Support PPO and A2C algorithms (stable-baselines3)
- `decide()` method for real-time decisions
- Record all decisions with confidence scores and action probabilities

### 16.4 RL Training Script

Implement `scripts/train_rl.py`:
- Symbol-by-symbol training
- Walk-forward evaluation
- Model metrics: PF, Win Rate, Sharpe, Max Drawdown
- Save to rl_models table

### 16.5 RL Database Tables

```sql
CREATE TABLE rl_models (
    model_id VARCHAR(32) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    algorithm VARCHAR(20) NOT NULL,
    train_start TIMESTAMPTZ NOT NULL,
    train_end TIMESTAMPTZ NOT NULL,
    metrics JSONB,
    model_path VARCHAR(255),
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE rl_decisions (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_id VARCHAR(32),
    action INTEGER NOT NULL,
    action_name VARCHAR(10) NOT NULL,
    confidence FLOAT,
    action_probs JSONB,
    value_estimate FLOAT,
    model_predictions JSONB,
    position_before FLOAT,
    position_after FLOAT,
    executed BOOLEAN DEFAULT FALSE,
    pnl_result FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 16.6 RL API Endpoints

Add to FastAPI:
- `GET /api/rl/models` - List RL models
- `GET /api/rl/models/{id}` - Model details
- `POST /api/rl/models/{id}/promote` - Promote to production
- `GET /api/rl/decisions` - Decision history with filters
- `GET /api/rl/decisions/stats` - Decision statistics

### 16.7 RL Frontend Components

Add to `/training` page:
- `RLModelsTable.tsx`: Display RL models, metrics, promote button
- `RLDecisionsTable.tsx`: Real-time decision feed, action probabilities, confidence visualization

### 16.8 RL Acceptance Criteria

1. RL training produces a model that can be promoted to production
2. All RL decisions are recorded to `rl_decisions` table
3. Frontend displays RL models and decisions in real-time
4. RL agent can be used in shadow mode alongside existing policy

---

## 17) Fixed Universe Configuration

### 17.1 Disable Dynamic Universe

- Comment out `worker_universe` in docker-compose.yml
- Use fixed UNIVERSE from .env (30 symbols including gold/silver)

### 17.2 Universe List

```
BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,
LTCUSDT,BCHUSDT,SUIUSDT,AAVEUSDT,1000PEPEUSDT,FILUSDT,AXSUSDT,ENAUSDT,ZKUSDT,XAUUSDT,
XAGUSDT,HYPEUSDT,ZECUSDT,BNXUSDT,ALPHAUSDT,ZORAUSDT,TRUMPUSDT,PUMPUSDT,NEARUSDT,PAXGUSDT
```

### 17.3 Gold/Silver Inclusion

- XAUUSDT (Gold)
- XAGUSDT (Silver)
- PAXGUSDT (Gold-backed token)
