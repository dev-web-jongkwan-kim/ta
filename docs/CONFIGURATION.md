# Configuration

## Environment Variables (.env)

### Core
| Variable | Value | Description |
|----------|-------|-------------|
| `MODE` | `shadow` | 거래 모드 (shadow/live/off) |
| `APP_ENV` | `local` | 환경 |

### Database
| Variable | Value | Description |
|----------|-------|-------------|
| `DATABASE_URL` | `postgresql://ta:ta@localhost:5433/ta` | PostgreSQL 연결 |
| `REDIS_URL` | `redis://localhost:6380/0` | Redis 연결 |

### MinIO (모델 저장소)
| Variable | Value |
|----------|-------|
| `MINIO_ENDPOINT` | `localhost:9000` |
| `MINIO_ACCESS_KEY` | `minio` |
| `MINIO_SECRET_KEY` | `minio123` |
| `MINIO_BUCKET` | `models` |

### Binance
| Variable | Value | Description |
|----------|-------|-------------|
| `BINANCE_API_KEY` | (설정됨) | API Key |
| `BINANCE_API_SECRET` | (설정됨) | API Secret |
| `BINANCE_TESTNET` | `false` | 실제 네트워크 사용 |

### Universe (22개 심볼)
```
BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,
DOTUSDT,LINKUSDT,LTCUSDT,BCHUSDT,SUIUSDT,AAVEUSDT,1000PEPEUSDT,
FILUSDT,AXSUSDT,ENAUSDT,ZKUSDT,ZECUSDT,TRUMPUSDT,NEARUSDT
```

**제외된 심볼**: XAUUSDT, XAGUSDT, HYPEUSDT, ZORAUSDT, PUMPUSDT, PAXGUSDT, ALPHAUSDT, BNXUSDT

### Data Collection
| Variable | Value | Description |
|----------|-------|-------------|
| `COLLECT_INTERVAL_SEC` | `5` | 데이터 수집 간격 |
| `FLUSH_INTERVAL_SEC` | `2` | 버퍼 플러시 간격 |
| `FLUSH_BATCH_SIZE` | `500` | 배치 크기 |

### Risk Management
| Variable | Value | Description |
|----------|-------|-------------|
| `MAX_USED_MARGIN_PCT` | `0.35` | 최대 마진 사용률 35% |
| `DAILY_LOSS_LIMIT_PCT` | `0.02` | 일일 손실 한도 2% |
| `MAX_POSITIONS` | `6` | 최대 동시 포지션 |
| `MAX_DIRECTIONAL_POSITIONS` | `4` | 같은 방향 최대 포지션 (신규) |
| `MAX_TOTAL_NOTIONAL_PCT` | `1.2` | 최대 총 명목가치 |
| `MAX_DIRECTIONAL_NOTIONAL_PCT` | `0.8` | 최대 방향성 명목가치 |

### Policy (진입 필터)
| Variable | Value | Description |
|----------|-------|-------------|
| `EV_MIN` | `0.001` | 최소 Expected Return (0.1%) |
| `Q05_MIN` | `-0.02` | 최소 5th percentile (-2%) |
| `MAE_MAX` | `0.05` | 최대 MAE (5%) |

### Trading (복리 시스템)
| Variable | Value | Description |
|----------|-------|-------------|
| `INITIAL_CAPITAL` | `300.0` | 초기 자본 $300 |
| `POSITION_SIZE` | `30.0` | (미사용, 복리로 대체) |
| `LEVERAGE` | `20` | 레버리지 20배 |

**복리 계산**: 포지션 사이즈 = 현재 자본 × 10% × 레버리지(20)

### Simulation
| Variable | Value | Description |
|----------|-------|-------------|
| `TAKER_FEE_RATE` | `0.0004` | Taker 수수료 0.04% |
| `SLIPPAGE_K` | `0.15` | 슬리피지 계수 |

---

## Signal Filters (services/policy/decide.py)

### 기본 필터 (EV/Q05/MAE)
| 필터 | 조건 | 설명 |
|------|------|------|
| EV_MIN | EV < 0.001 | 최소 기대수익 0.1% 미만 시 차단 |
| Q05_MIN | q05 < -0.02 | 5% 손실 확률 > 2% 시 차단 |
| MAE_MAX | e_mae > 0.05 | 최대 손실 예상 > 5% 시 차단 |

### 시장 상황 필터 (신규)

#### 1. Market Direction Filter (극단적 시장 방향 차단)
| 조건 | 액션 | 사유 코드 |
|------|------|----------|
| btc_ret_60 > 1.5% & decision=SHORT | 차단 | EXTREME_PUMP |
| btc_ret_60 < -1.5% & decision=LONG | 차단 | EXTREME_DUMP |

#### 2. Volatility Filter (고변동성 필터)
| 조건 | 액션 | 사유 코드 |
|------|------|----------|
| atr_percentile > 90% & EV < 0.2% | 차단 | HIGH_VOL_LOW_EV |

#### 3. Consecutive Loss Filter (연속 손실 필터)
| 조건 | 액션 | 사유 코드 |
|------|------|----------|
| 연속 손실 >= 3회 & EV < 0.3% | 차단 | RECOVERY_MODE |

### 필터 상수 (decide.py)
```python
EXTREME_PUMP_THRESHOLD = 0.015   # 1시간 1.5% 상승
EXTREME_DUMP_THRESHOLD = -0.015  # 1시간 1.5% 하락
HIGH_VOLATILITY_THRESHOLD = 90   # ATR 상위 10%
HIGH_VOL_EV_MIN = 0.002          # 고변동성 EV 기준 0.2%
MAX_CONSECUTIVE_LOSSES = 3       # 연패 임계값
RECOVERY_EV_MIN = 0.003          # 연패 후 EV 기준 0.3%
```

---

## Hardcoded Values

### Triple Barrier (services/labeling/)

| 파일 | 변수 | 값 | 설명 |
|------|------|-----|------|
| `pipeline.py:62-64` | `k_tp` | `1.5` | TP = ATR × 1.5 |
| `pipeline.py:62-64` | `k_sl` | `1.0` | SL = ATR × 1.0 |
| `pipeline.py:62-64` | `h_bars` | `360` | 최대 보유 360분 |
| `pipeline.py:31` | 1m h_bars | `360` | 1분봉 6시간 |
| `pipeline.py:32` | 15m h_bars | `24` | 15분봉 6시간 |
| `pipeline.py:33` | 1h h_bars | `6` | 1시간봉 6시간 |

### Realtime Worker (services/realtime_worker_ws.py)

| 라인 | 변수 | 값 | 설명 |
|------|------|-----|------|
| `50-54` | DATA_FRESHNESS_MAX_GAP | 1m=5, 15m=30, 1h=90 | 데이터 최대 갭 (분) |
| `55` | REQUIRED_HISTORY_HOURS | `60` | 필요 히스토리 60시간 |
| `683-684` | k_sl, k_tp | `1.0`, `1.5` | SL/TP 계수 |
| `675` | ATR fallback | `atr_1m * 10` | 15m ATR 없을 때 근사치 |

### Features (services/features/compute.py)

| 변수 | 값 | 설명 |
|------|-----|------|
| ADX window | `14` | ADX 계산 기간 |
| RSI window | `14` | RSI 계산 기간 |
| ATR window | `14` | ATR 계산 기간 |
| MA periods | `5, 20` | 이동평균 기간 |
| Z-score window | `60` | Z-score 계산 기간 |
| Correlation window | `60` | 상관관계 기간 |

### Position Manager (services/engine/position_manager.py)

| 라인 | 변수 | 값 | 설명 |
|------|------|-----|------|
| `76` | `_hit_cooldown_sec` | `5` | SL/TP 중복 방지 쿨다운 |

### RL Environment (services/rl/environment.py)

| 변수 | 값 | 설명 |
|------|-----|------|
| `max_position_size` | `1.0` | 최대 포지션 크기 |
| 보유시간 정규화 | `360` | 최대 360분 기준 |

### Fill Model (services/simulator/fill.py)

| 변수 | 값 | 설명 |
|------|-----|------|
| max slippage | `0.005` | 최대 슬리피지 0.5% |

---

## Current Model

| 항목 | 값 |
|------|-----|
| Model ID | `e0925468-bda8-4711-86d8-9b1866b7fc70` |
| Algorithm | LightGBM |
| Feature Schema | v5 (Multi-TF: 1m+15m+1h = 105 features) |
| Label Spec Hash | `97ed8e99c6d9f2ea` |
| Train Period | 2024-01-01 ~ 2026-01-30 |
| Symbols | 21개 |
| Production | Yes |

---

## Docker Compose Ports

| Service | Internal | External |
|---------|----------|----------|
| postgres | 5432 | 5433 |
| redis | 6379 | 6380 |
| minio | 9000/9001 | 9000/9001 |
| api | 7101 | 7101 |
| web | 7000 | 7100 |
