# TA: Binance USDT-M Supervised Trading System (v2)

This repo delivers a runnable monorepo with:

- FastAPI API/WS gateway (`apps/api`)
- Next.js web UI (`apps/web`)
- Realtime worker (ingestion → features → inference → policy → risk → shadow simulator)
- Batch worker (labeling/training jobs)
- Postgres + Redis + MinIO via Docker Compose

Default run mode is **SHADOW**.

## Quickstart

```bash
cp .env.example .env

docker compose up -d postgres redis minio

# Apply schema
psql postgresql://ta:ta@localhost:5433/ta -f infra/migrations/001_init.sql
psql postgresql://ta:ta@localhost:5433/ta -f infra/migrations/002_trade_group.sql
psql postgresql://ta:ta@localhost:5433/ta -f infra/migrations/003_training_report.sql
psql postgresql://ta:ta@localhost:5433/ta -f infra/migrations/004_drift_metrics.sql
psql postgresql://ta:ta@localhost:5433/ta -f infra/migrations/005_market_sentiment.sql
psql postgresql://ta:ta@localhost:5433/ta -f infra/migrations/009_rl_tables.sql

## Python environment notes

Before installing dependencies, make sure PostgreSQL headers/binaries (`pg_config`) and PyO3 expectable flags are visible. On macOS run:

```bash
brew install libpq
echo 'export PATH="$(brew --prefix libpq)/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

Then create the venv with a supported interpreter (Python 3.11 or 3.13 via Homebrew) and install requirements:

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests services/training/tests services/labeling/tests
```

The `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` setting lets `pydantic-core` build against Python 3.13 while still using the stable ABI.

# Insert universe
python scripts/init_universe.py

# Optional: bootstrap recent history
python scripts/bootstrap_history.py

# Build top-50 universe (USDT-M perpetual, trading, by quoteVolume)
python -m services.collector.universe_builder

# Start API + workers + web
docker compose up -d api worker_realtime worker_batch worker_universe web
```

UI: http://localhost:3000
API: http://localhost:8000
MinIO: http://localhost:9001 (user `minio`, pass `minio123`)

## Modes

Runtime mode is stored in Redis and defaults to `shadow`.

```bash
# Set mode via API
curl -X POST "http://localhost:8000/api/trading/toggle?mode=shadow"
curl -X POST "http://localhost:8000/api/trading/toggle?mode=live"
curl -X POST "http://localhost:8000/api/trading/toggle?mode=off"
```

## Binance Testnet

Set in `.env`:

```
BINANCE_TESTNET=true
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
```

Live trading requires API keys and `mode=live`. Shadow mode does not place real orders.

## Labeling + Training

### 기본 학습

```bash
python services/labeling/run_labeling.py

# Queue a training job (batch worker label -> train -> register -> report)
psql postgresql://ta:ta@localhost:5433/ta -c "\
INSERT INTO training_jobs (job_id, status, config) VALUES (
  gen_random_uuid(),
  'queued',
  '{
    \"train_start\":\"2024-01-01\",
    \"train_end\":\"2024-12-31\",
    \"val_start\":\"2024-11-01\",
    \"val_end\":\"2025-01-01\",
    \"targets\":[\"er_long\",\"q05_long\",\"e_mae_long\",\"e_hold_long\"],
    \"feature_schema_version\":1,
    \"label_config\":{\"k_tp\":1.5,\"k_sl\":1.0,\"h_bars\":360,\"risk_mae_atr\":3.0},
    \"purge_bars\":0,
    \"embargo_pct\":0.0
  }'
);"
```

### 개선된 학습 파이프라인 (v3)

멀티 타임프레임 + Phase 1/2 개선사항을 적용한 학습:

```bash
# 멀티 타임프레임 마이그레이션 (최초 1회)
psql postgresql://ta:ta@localhost:5433/ta -f infra/migrations/011_multi_timeframe.sql

# Phase 1 + Phase 2 개선된 학습 실행
python scripts/train_phase1_phase2.py

# 옵션
python scripts/train_phase1_phase2.py --optuna-trials 50  # Optuna 시행 횟수
python scripts/train_phase1_phase2.py --skip-tb-opt       # Triple Barrier 최적화 스킵
python scripts/train_phase1_phase2.py --skip-meta         # Meta-labeling 스킵
```

#### 멀티 타임프레임 피처

3개 타임프레임(1m, 15m, 1h)의 피처를 결합하여 학습:

- **1분봉**: 실시간 시장 상태 (ADX, RSI, ATR, 펀딩율 등)
- **15분봉**: 중기 추세 및 변동성
- **1시간봉**: 장기 추세 및 레짐

새로운 피처:
- `atr_percentile`: 변동성 백분위
- `trend_strength`: ADX와 RSI 기반 추세 강도
- `hour_of_day`, `day_of_week`: 시간대 피처
- `is_asia_session`, `is_us_session`: 세션 구분
- `btc_lead_ret`, `eth_btc_spread`: 심볼 간 상관관계

#### Phase 1 개선사항

1. **Optuna 하이퍼파라미터 튜닝**: LightGBM 파라미터 자동 최적화
2. **피처 선택**: Feature importance 기반 저중요도 피처 제거
3. **클래스 불균형 처리**: scale_pos_weight 조정

#### Phase 2 개선사항

1. **Meta-labeling**: 1차 모델 예측에 대한 2차 필터 모델
2. **Triple Barrier 최적화**: k_tp, k_sl, h_bars 파라미터 최적화
3. **CatBoost 앙상블**: LightGBM(0.6) + CatBoost(0.4) 앙상블

#### 성능 개선 결과

| 필터 | Profit Factor | Expectancy | 거래 수 |
|------|---------------|------------|---------|
| Baseline | 0.71 | -0.09% | 129,600 |
| er > 0 | 2.02 | +0.20% | 41,118 |
| er > 0.001 | **2.64** | **+0.30%** | 25,412 |

### 백테스트

```bash
# 모델 예측 기반 필터 백테스트
python scripts/backtest_with_model.py

# 필터 조합 테스트 (threshold, session, hour)
python scripts/backtest_filters.py
```

Completed training jobs now populate both `report_uri` (S3 path `s3://models/reports/{model_id}.json`) and `report_json` (stored in Postgres). The JSON payload contains PF, MDD, Expectancy, Tail loss, Turnover, regime splits, and cost breakdown so the UI can compare models and expose the training report.

Batch worker picks up queued jobs.

## Trade Group Integrity

Run the orphan-detection helper periodically (or after system restarts) so missing orders/fills are surfaced as risk events:

```bash
python services/simulator/orphan_check.py
```

The script logs `ORPHAN_FILL` and `INCOMPLETE_TRADE_GROUP` risk events and prints samples for inspection.

## Drift & Data Quality

Run the drift job frequently (e.g., cron/minutely) so PSI, missing, latency, and outlier metrics stay current and risk-gating status flows into the UI:

```bash
python services/monitoring/drift_job.py
```

This job augments `drift_metrics` with `outlier_count`, writes `DRIFT_ALERT`/`DRIFT_BLOCK`/`MISSING_*`/`LATENCY_ALERT` risk events, and updates the runtime drift status that `check_risk` reads (blocking decisions with `DRIFT_ALERT`, `DRIFT_BLOCK`, `MISSING_DATA`, `MISSING_BLOCK`). Adjust the thresholds via the new `.env` settings (`DRIFT_ALERT_PSI`, `DRIFT_BLOCK_PSI`, `DRIFT_LATENCY_THRESHOLD_MS`, `DRIFT_MISSING_ALERT_RATE`, `DRIFT_MISSING_BLOCK_RATE`).

## Live Guardrails

Live mode enforces the following guardrails before any real order executes:

- **Mode gating**: Only `mode=live` allows bracket placement; SHADOW stays simulated.
- **Brackets only**: Every entry goes through `services/execution/trader.py`, which places a MARKET entry plus STOP_MARKET/TAKE_PROFIT_MARKET reduce-only brackets and flattens the position if brackets fail.
- **Portfolio constraints**: `check_risk` uses exposures/daily PnL from `positions`/`fills` (`packages/common/portfolio`) to block if open positions, total or directional notional, used margin %, or daily loss exceed configured caps.
- **Circuit breakers**: Drift alerts/blocks, missing data, latency alert, userstream disconnect, and order failure spikes all add reason codes (`DRIFT_ALERT`, `MISSING_DATA`, `LATENCY_ALERT`, `USERSTREAM_DOWN`, `ORDER_FAILURE_SPIKE`). The UI can read these codes from `risk_events`.
- **Reason codes**: Every block emits a `risk_event` (e.g., `MAX_EXPOSURE`, `DAILY_STOP`, `DRIFT_BLOCK`, `MISSING_BLOCK`), so operators can see why trading pausing and adjust thresholds via `.env`.

Set thresholds such as `MAX_TOTAL_NOTIONAL_PCT`, `MAX_DIRECTIONAL_NOTIONAL_PCT`, `MAX_POSITIONS`, `DAILY_LOSS_LIMIT_PCT`, and the drift/missing limits in `.env` to tune aggressiveness. Once all guards clear, the live broker uses the bracket orders for safety.

## Tests

```bash
pytest services/labeling/tests services/training/tests tests
```

## Key Endpoints

- `GET /api/status`
- `GET /api/universe`
- `GET /api/signals/latest?limit=50`
- `GET /api/data-quality/summary`
- `POST /api/models/{id}/promote`
 
`GET /api/status` now also returns exposure metrics (open positions, total/directional notional, daily PnL/loss) so the UI can display real-time portfolio risk alongside the reason codes.

## Safety

Live execution enforces:

- MARKET entry
- STOP_MARKET SL (reduceOnly)
- TAKE_PROFIT_MARKET TP (reduceOnly)
- If bracket placement fails, the position is flattened immediately

Kill-switch triggers on data stale or userstream failure.

## Reinforcement Learning (RL) Agent

시스템은 하이브리드 RL 에이전트를 지원합니다. LightGBM 모델의 예측값을 상태로 사용하여 RL 에이전트가 최종 거래 결정을 내립니다.

### RL 아키텍처

- **환경**: Gymnasium 기반 `TradingEnvironment` (`services/rl/environment.py`)
- **상태 공간**: 17차원 (모델 예측 4 + 시장 지표 10 + 포지션 상태 3)
- **액션 공간**: 4개 이산 액션 (Hold, Long, Short, Close)
- **알고리즘**: PPO/A2C (stable-baselines3)

### RL 학습

```bash
source .venv/bin/activate
python scripts/train_rl.py --symbols BTCUSDT,ETHUSDT --train-days 90 --timesteps 100000
```

옵션:
- `--symbols`: 학습할 심볼 (쉼표 구분)
- `--train-days`: 학습 기간 (일)
- `--eval-days`: 평가 기간 (일, 기본 30)
- `--timesteps`: 총 학습 스텝 (기본 100000)
- `--algorithm`: PPO 또는 A2C

### RL 테이블

```sql
-- RL 모델 저장
SELECT * FROM rl_models;

-- RL 결정 기록
SELECT * FROM rl_decisions ORDER BY ts DESC LIMIT 100;
```

### RL API 엔드포인트

- `GET /api/rl/models` - RL 모델 목록
- `GET /api/rl/models/{id}` - 모델 상세
- `POST /api/rl/models/{id}/promote` - 프로덕션 승격
- `GET /api/rl/decisions?symbol=BTCUSDT&limit=100` - 결정 히스토리
- `GET /api/rl/decisions/stats?hours=24` - 결정 통계

### RL 프론트엔드

Training 페이지 (`/training`)에서 다음을 확인할 수 있습니다:
- RL 모델 목록 및 성능 메트릭 (PF, Win Rate, Sharpe, Max DD)
- 액션 분포 (Hold/Long/Short/Close)
- 실시간 RL 결정 피드 (신뢰도, 액션 확률)
- 모델 프로덕션 승격 기능

## Universe (고정)

현재 30개 심볼로 고정되어 있습니다 (금/은 포함):
```
BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,
LTCUSDT,BCHUSDT,SUIUSDT,AAVEUSDT,1000PEPEUSDT,FILUSDT,AXSUSDT,ENAUSDT,ZKUSDT,XAUUSDT,
XAGUSDT,HYPEUSDT,ZECUSDT,BNXUSDT,ALPHAUSDT,ZORAUSDT,TRUMPUSDT,PUMPUSDT,NEARUSDT,PAXGUSDT
```

`.env`의 `UNIVERSE` 변수에서 설정됩니다.
