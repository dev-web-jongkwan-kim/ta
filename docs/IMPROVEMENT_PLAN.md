# 개선 계획서: 30종목 저사양 PC 지원 + 성능 향상

## 현재 상황

- **목표**: 30개 종목 학습 및 실시간 매매
- **제약**: 저사양 PC (메모리/CPU 부족)
- **현재 성능**: PF=2.64, Expectancy=+0.30% (3종목 기준)

---

## Part 1: 코드 레벨 병목 분석

### 1.1 메모리 병목

| 위치 | 파일 | 문제 | 영향 |
|------|------|------|------|
| 전체 심볼 로드 | `train.py:122-179` | 모든 심볼 데이터를 메모리에 concat | **2.5GB+** (50종목 기준) |
| 멀티 타임프레임 | `train.py:182-315` | 1m+15m+1h 3배 데이터 로드 | 심볼당 3배 메모리 |
| JSON 파싱 | `train.py:70-85` | row-by-row 파싱 + 중간 복사본 생성 | 500-1000ms/100K rows |
| DataFrame 복사 | `train_improved.py:390-690` | meta-labeling 시 val_df.copy() 다수 | 400-500MB 중복 |

**핵심 문제**: `pd.concat(all_dfs)` - 모든 심볼 데이터를 한번에 메모리에 적재

```python
# 현재 코드 (문제)
for symbol in symbols_list:
    df = load_symbol(symbol)
    all_dfs.append(df)  # 메모리 누적
result = pd.concat(all_dfs)  # 전체 복사
```

### 1.2 CPU 병목

| 위치 | 파일 | 문제 | 영향 |
|------|------|------|------|
| Optuna 튜닝 | `train_improved.py:253-310` | 50 trials × 4 targets = 200회 학습 | **100+ 시간** |
| Feature Selection | `train_improved.py:337-373` | 매 타겟마다 LightGBM 학습 | 30-120초/회 |
| Rolling 계산 | `compute.py:23-172` | 20+ rolling/ewm 연산 | 50-200ms/심볼 |
| Apply 연산 | `compute.py:298, 372` | row-by-row 함수 호출 | 50-100ms/심볼 |

**핵심 문제**: Optuna 50 trials × 4 targets × 30초 = **100시간 이상** 소요

### 1.3 I/O 병목

| 위치 | 문제 | 영향 |
|------|------|------|
| 심볼당 3 쿼리 | 1m + 15m + 1h 별도 쿼리 | 30종목 × 3 = 90회 DB 접근 |
| fetch_all() | 전체 결과 메모리 로드 | 대용량 결과셋 OOM 위험 |
| 순차 처리 | 심볼 병렬 처리 없음 | I/O 대기 시간 낭비 |

---

## Part 2: ML 트레이딩 개선 포인트

### 2.1 Feature Engineering

| 개선 | 설명 | 기대 효과 | 난이도 | 메모리 |
|------|------|----------|--------|--------|
| **Rolling Correlation** | BTC-ETH-타겟 상관관계 | 높음 | 쉬움 | 낮음 |
| **HMM 레짐 감지** | 변동성 상태 분류 (2-3상태) | 높음 | 중간 | 낮음 |
| **Lead-Lag 분석** | 심볼간 선행/후행 관계 | 중간 | 중간 | 낮음 |
| **Fear & Greed Index** | 시장 심리 지표 | 중간 | 쉬움 | 매우 낮음 |

```python
# Rolling Correlation 구현 예시
def _rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    return s1.rolling(window).corr(s2)

df["btc_corr_60"] = _rolling_corr(df["ret_1"], btc_ret, window=60)
df["corr_regime_break"] = (df["btc_corr_60"] < 0.3).astype(int)
```

### 2.2 Model Architecture

| 개선 | 설명 | 기대 효과 | 난이도 | 메모리 |
|------|------|----------|--------|--------|
| **XGBoost 추가** | LightGBM 대안 비교 | 중간 | 쉬움 | 유사 |
| **Stacking 앙상블** | LightGBM+XGBoost 메타러너 | 높음 | 중간 | 2-3배 |
| **증분 학습** | init_model로 이어서 학습 | 중간 | 쉬움 | **낮음** |

### 2.3 Labeling 전략

| 개선 | 설명 | 기대 효과 | 난이도 |
|------|------|----------|--------|
| **Dynamic TP/SL** | ATR percentile 기반 동적 배리어 | 높음 | 쉬움 |
| **Meta-labeling 강화** | 2차 필터 모델 정확도 향상 | 높음 | 중간 |
| **Multi-horizon** | 15m/1h/4h 예측 일치 시 진입 | 중간 | 중간 |

```python
# Dynamic Barrier 구현 예시
def get_dynamic_barriers(atr_percentile: float, base_k_tp: float = 1.5):
    if atr_percentile > 80:  # 고변동
        k_tp, k_sl = base_k_tp * 1.5, base_k_tp * 1.0
    elif atr_percentile < 20:  # 저변동
        k_tp, k_sl = base_k_tp * 0.8, base_k_tp * 0.6
    else:
        k_tp, k_sl = base_k_tp, base_k_tp * 0.8
    return k_tp, k_sl
```

### 2.4 Risk Management

| 개선 | 설명 | 기대 효과 | 난이도 |
|------|------|----------|--------|
| **Half Kelly** | 최적 포지션 크기의 50% | 높음 | 쉬움 |
| **ATR 기반 사이징** | 변동성 비례 포지션 | 높음 | 쉬움 |
| **상관관계 필터** | 고상관 종목 동시 진입 방지 | 높음 | 쉬움 |

---

## Part 3: 저사양 PC 최적화 전략

### 3.1 메모리 최적화

#### A. float64 → float32 변환 (메모리 50% 절감)

```python
# 현재
feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0)

# 개선
feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0).astype(np.float32)
```

| 데이터 타입 | 바이트 | 1M rows × 100 features |
|------------|--------|------------------------|
| float64 | 8 | 800 MB |
| float32 | 4 | **400 MB** |

#### B. 심볼별 증분 학습 (메모리 고정)

```python
def train_by_symbol_incremental(symbols: List[str]) -> lgb.Booster:
    model = None
    for symbol in symbols:
        df = load_single_symbol(symbol)  # 한 종목만 로드

        train_data = lgb.Dataset(
            df[feature_cols].astype(np.float32),
            label=df['ret_net'].astype(np.float32),
            free_raw_data=True  # 학습 후 원본 데이터 해제
        )

        model = lgb.train(
            params, train_data,
            num_boost_round=30,
            init_model=model,  # 이전 모델에서 이어서 학습
            keep_training_booster=True
        )

        del df, train_data
        gc.collect()

    return model
```

**메모리 사용**: 전체 데이터 대신 **1개 심볼분만 유지** (30종목 → 1/30 메모리)

#### C. Server-side Cursor (스트리밍)

```python
from contextlib import contextmanager

@contextmanager
def streaming_cursor(query: str, params: tuple, chunk_size: int = 5000):
    conn = get_conn()
    cursor = conn.cursor(name='streaming')  # Named cursor = server-side
    cursor.itersize = chunk_size
    try:
        cursor.execute(query, params)
        yield cursor
    finally:
        cursor.close()
        conn.close()

# 사용
with streaming_cursor(query, params) as cursor:
    while True:
        rows = cursor.fetchmany(5000)
        if not rows:
            break
        process_chunk(rows)
```

### 3.2 CPU 최적화

#### A. Optuna 최적화 (100시간 → 1시간)

```python
# 현재: 50 trials × 4 targets = 200회 학습
# 개선: 대표 타겟 1개로 파라미터 찾고 나머지 적용

def optimize_once_apply_all(X_train, y_train, X_val, y_val, n_trials=30):
    # er_long에 대해서만 Optuna 실행
    best_params = optimize_lgbm_params(X_train, y_train['ret_net'],
                                       X_val, y_val['ret_net'], n_trials=n_trials)

    # 동일 파라미터로 모든 타겟 학습 (튜닝 없이)
    models = {}
    for target in ['er_long', 'q05_long', 'e_mae_long', 'e_hold_long']:
        models[target] = lgb.LGBMRegressor(**best_params)
        models[target].fit(X_train, y_train[target_map[target]])

    return models
```

**시간 절감**: 200회 → 30회 (약 **85% 감소**)

#### B. LightGBM 저메모리 파라미터

```python
lgbm_low_memory_params = {
    'n_estimators': 300,      # 500 → 300
    'num_leaves': 31,         # 64 → 31
    'max_depth': 6,           # 제한 추가
    'max_bin': 63,            # 255 → 63 (메모리 감소)
    'min_data_in_leaf': 100,  # 20 → 100 (작은 트리)
    'feature_fraction': 0.6,  # 피처 샘플링
    'bagging_fraction': 0.6,  # 데이터 샘플링
}
```

#### C. Feature Selection 캐싱

```python
# 피처 중요도는 한번만 계산하고 재사용
@lru_cache(maxsize=1)
def get_selected_features(train_hash: str) -> List[str]:
    # 첫 호출 시에만 실행
    model = lgb.LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })
    return importances.nlargest(50, 'importance')['feature'].tolist()
```

### 3.3 I/O 최적화

#### A. 쿼리 통합

```python
# 현재: 심볼당 3회 쿼리 (1m, 15m, 1h)
# 개선: 1회 JOIN 쿼리

query = """
SELECT
    f1.symbol, f1.ts, f1.features as f_1m,
    f15.features as f_15m,
    f1h.features as f_1h,
    l.ret_net, l.y
FROM features_1m f1
JOIN labels_long_1m l ON f1.symbol = l.symbol AND f1.ts = l.ts
LEFT JOIN features_15m f15 ON f1.symbol = f15.symbol
    AND f15.ts = date_trunc('minute', f1.ts - interval '1 minute' * (extract(minute from f1.ts)::int % 15))
LEFT JOIN features_1h f1h ON f1.symbol = f1h.symbol
    AND f1h.ts = date_trunc('hour', f1.ts)
WHERE f1.symbol = %s AND f1.ts BETWEEN %s AND %s
"""
```

**효과**: 30종목 × 3쿼리 = 90회 → 30종목 × 1쿼리 = **30회**

---

## Part 4: 30종목 지원 수정 방향

### 4.1 아키텍처 변경

```
[현재]
load_all_symbols() → concat() → train()
     ↓ 메모리 폭발

[개선]
for symbol in symbols:
    load_single(symbol) → train_incremental() → gc.collect()
     ↓ 메모리 고정 (1종목분)
```

### 4.2 구현 단계

#### Phase 1: 메모리 최적화 (필수)

1. **float32 변환**: `_normalize_features()` 수정
2. **심볼별 증분 학습**: `train_by_symbol_incremental()` 구현
3. **Streaming cursor**: DB 쿼리 수정

#### Phase 2: CPU 최적화 (필수)

4. **Optuna 단일화**: 대표 타겟만 튜닝
5. **저메모리 LightGBM 파라미터** 적용
6. **Feature Selection 캐싱**

#### Phase 3: ML 개선 (선택)

7. **Rolling Correlation 피처** 추가
8. **Dynamic TP/SL** 구현
9. **Half Kelly 포지션 사이징**

### 4.3 예상 리소스 사용량

| 항목 | 현재 (3종목) | 개선 후 (30종목) |
|------|-------------|-----------------|
| 메모리 | 3-5 GB | **4-6 GB** |
| 학습 시간 | 2시간 | **3-4시간** |
| DB 쿼리 | 9회 | **30회** |

---

## Part 5: 우선순위 로드맵

### Week 1: 필수 메모리/CPU 최적화

- [x] float32 변환
- [x] ~~심볼별 증분 학습 구현~~ (배치 학습이 성능 우수, 폐기)
- [x] Optuna 타겟별 개별 튜닝 (50 trials × 4 targets)
- [x] 최적 LightGBM 파라미터 (num_leaves=64-256, n_estimators=500-1000)

### Week 2: I/O 및 추가 최적화

- [x] Feature Selection (누적 중요도 99.9%)
- [x] 멀티 타임프레임 (1m, 15m, 1h)
- [x] CatBoost 앙상블

### Week 3: ML 개선

- [x] Rolling Correlation 피처 (packages/common/risk.py)
- [x] Dynamic TP/SL (packages/common/risk.py)
- [x] Half Kelly 사이징 (packages/common/risk.py)
- [x] Meta-labeling (2차 필터 모델)

### Week 4: 검증 및 배포

- [x] 3종목 학습 테스트 완료
- [ ] 30종목 학습 테스트
- [ ] 실시간 추론 성능 테스트
- [ ] 실거래 테스트

---

## Part 6: 최종 결과

### 배치 학습 vs 증분 학습 비교 (2026-02-03)

| 학습 방식 | er>0.001 PF | er>0.001 Expectancy | 결론 |
|----------|-------------|---------------------|------|
| **배치 학습** | **2.64** | **+0.30%** | **최고 성능** |
| 증분 학습 | 0.71 | -0.07% | 성능 저하 |

**결론**: 증분 학습(심볼별 init_model)은 메모리 절약에는 좋지만 성능이 크게 저하됨.
배치 학습(전체 데이터 한번에)을 권장.

### 최종 권장 설정

```python
# scripts/train_30_symbols.py 사용
cfg = ImprovedTrainConfig(
    use_multi_tf=True,           # 멀티 타임프레임
    use_optuna=True,             # Optuna 튜닝
    optuna_trials=50,            # 50 trials × 4 targets
    use_feature_selection=True,  # 피처 선택
    feature_importance_threshold=0.001,  # 99.9% 누적 중요도
    use_meta_labeling=True,      # Meta-labeling
    use_catboost_ensemble=True,  # CatBoost 앙상블
)
```

### 최고 결과

- **PF**: 2.64
- **Expectancy**: +0.30%
- **필터**: er>0.001
- **거래 수**: 25,993건 (검증 기간)

---

## 참고 자료

- [LightGBM Incremental Learning](https://medium.com/data-science-collective/incremental-learning-in-lightgbm-and-xgboost-9641c2e68d4b)
- [Memory-Efficient Data Pipelines](https://mljourney.com/how-to-write-memory-efficient-data-pipelines-in-python/)
- [PostgreSQL Keyset Pagination](https://www.citusdata.com/blog/2016/03/30/five-ways-to-paginate/)
- [Kelly Criterion 2026](https://medium.com/@tmapendembe_28659/risk-management-using-kelly-criterion-2eddcf52f50b)
