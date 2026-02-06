# System Flow

## 1. 실시간 데이터 플로우

```
Binance WebSocket
       │
       ├── kline_1m ──────────────────┐
       ├── kline_15m ─────────────────┤
       ├── kline_1h ──────────────────┤
       ├── markPrice@1s ──────────────┼── StreamRouter
       └── bookTicker ────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                     │
            KlineHandler                          MarkPriceHandler
                   │                                     │
                   ▼                                     ▼
            IngestBuffer                         SL/TP 체크
                   │                                     │
                   ▼                                     │
              PostgreSQL                                 │
         (candles_1m/15m/1h)                            │
                   │                                     │
                   └──────────── 1분봉 close ────────────┤
                                      │                  │
                                      ▼                  │
                               run_inference()           │
                                      │                  │
                                      ▼                  ▼
                              _execute_shadow_trade()   close_position()
                                      │                  │
                                      ▼                  ▼
                               PositionManager     positions (FINAL)
                                      │
                                      ▼
                              positions (entry)
```

## 2. 추론 플로우 (run_inference)

```
1분봉 Close 감지
       │
       ▼
데이터 Freshness 체크 ──── 실패 ──→ Skip
       │
       │ 성공
       ▼
캔들 로딩 (1m, 15m, 1h)
       │
       ▼
피처 계산 (compute_features_for_symbol)
       │
       ├── 1m 피처 (44개)
       ├── 15m 피처 (44개)
       └── 1h 피처 (44개)
       │
       ▼
피처 저장 (features_1m)
       │
       ▼
Multi-TF 피처 병합 (132개)
       │
       ├── f_1m_* (44개)
       ├── f_15m_* (44개)
       └── f_1h_* (44개)
       │
       ▼
Predictor.predict()
       │
       ├── er_long, er_short
       ├── q05_long, q05_short
       ├── e_mae_long, e_mae_short
       └── e_hold_long, e_hold_short
       │
       ▼
PolicyConfig.decide()
       │
       ├── EV_MIN (0.002) 체크
       ├── Q05_MIN (-0.02) 체크
       └── MAE_MAX (0.05) 체크
       │
       ▼
Decision: LONG / SHORT / HOLD
       │
       ▼
SL/TP 계산
       │
       ├── 15m ATR 로딩
       ├── SL = price ∓ ATR × 1.0
       └── TP = price ± ATR × 1.5
       │
       ▼
Signal 저장 (signals 테이블)
       │
       ▼
_execute_shadow_trade()
```

## 3. Shadow 거래 플로우

### 진입 (Entry)

```
_execute_shadow_trade()
       │
       ├── session_manager.is_trading 체크
       ├── mode == "shadow" 체크
       ├── decision in (LONG, SHORT) 체크
       └── block_reasons 체크
       │
       │ 모두 통과
       ▼
_position_entry_lock 획득
       │
       ├── position_manager.get_position() ─── 있음 ──→ Skip
       ├── has_open_position_in_db() ────────── 있음 ──→ Skip
       └── count_open_positions_in_db() ─── >= MAX ──→ Skip
       │
       │ 모두 통과
       ▼
체결가 결정
       │
       ├── LONG: book_ticker.ask
       └── SHORT: book_ticker.bid
       │
       ▼
position_manager.update_position_from_cache()
       │
       ├── 메모리 캐시 업데이트
       └── DB 저장 (event_type='entry')
       │
       ▼
Redis 이벤트 발행 (trading_events)
```

### 청산 (Exit)

```
markPrice@1s 수신
       │
       ▼
MarkPriceHandler._check_sl_tp()
       │
       ▼
position_manager.check_sl_tp(mark_price)
       │
       ├── SL 체크: LONG → mark <= sl_price
       │            SHORT → mark >= sl_price
       │
       └── TP 체크: LONG → mark >= tp_price
                    SHORT → mark <= tp_price
       │
       │ Hit!
       ▼
체결가 결정
       │
       ├── LONG 청산: book_ticker.bid
       └── SHORT 청산: book_ticker.ask
       │
       ▼
position_manager.close_position()
       │
       ├── PnL 계산
       ├── DB 저장 (event_type='FINAL')
       └── 메모리 캐시 제거
       │
       ▼
session_manager.record_trade()
       │
       ▼
Redis 이벤트 발행 (trading_events)
```

## 4. 학습 플로우

```
training_jobs 테이블에 작업 등록 (status='queued')
       │
       ▼
batch_worker 감지
       │
       ▼
라벨링 (Triple Barrier)
       │
       ├── 캔들 로딩
       ├── ATR 계산 (15m ATR 사용)
       ├── TP/SL/Time 배리어 설정
       └── 라벨 생성 (y=1: TP hit, y=0: SL/Time hit)
       │
       ▼
labels_long_1m, labels_short_1m 저장
       │
       ▼
학습 데이터 로딩
       │
       ├── features_1m
       ├── features_15m
       ├── features_1h
       └── labels
       │
       ▼
LightGBM 학습
       │
       ├── 8개 타겟 각각 학습
       └── MultiOutputRegressor
       │
       ▼
모델 평가
       │
       ├── Profit Factor
       ├── Expectancy
       └── Win Rate
       │
       ▼
MinIO 저장 (models 버킷)
       │
       ▼
models 테이블 등록
```

## 5. 포지션 상태 관리

### DB 스키마 (positions 테이블)

```sql
PRIMARY KEY (trade_group_id, symbol, ts, side)

event_type:
  - 'entry': 포지션 진입
  - 'FINAL': 포지션 청산
```

### 열린 포지션 쿼리

```sql
SELECT * FROM positions p
WHERE p.event_type = 'entry'
AND NOT EXISTS (
    SELECT 1 FROM positions p2
    WHERE p2.trade_group_id = p.trade_group_id
    AND p2.event_type = 'FINAL'
);
```

### 완료된 거래 쿼리

```sql
SELECT e.*, f.*
FROM positions e
JOIN positions f ON e.trade_group_id = f.trade_group_id
WHERE e.event_type = 'entry'
AND f.event_type = 'FINAL';
```
