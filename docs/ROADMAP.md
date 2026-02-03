# 트레이딩 시스템 로드맵

## 목표

30개 종목에 대해 1-2년간의 데이터를 기반으로 학습하여, 실시간 웹소켓 스트림을 구독하면서 매수/매도를 반복하여 수익 극대화

## 현재 시스템 상태

| 구성 요소 | 상태 | 설명 |
|-----------|------|------|
| 데이터 수집 | ✅ 완성 | 30종목 1분봉, 펀딩율, 오픈인터레스트 등 |
| 피처 계산 | ✅ 완성 | 기술적 지표 33개 |
| 라벨링 | ✅ 완성 | Triple Barrier 방식 |
| LightGBM 학습 | ✅ 완성 | Optuna 튜닝, CatBoost 앙상블 |
| 강화학습 (RL) | ⚠️ 부분완성 | 환경/에이전트 있음, 실시간 연동 필요 |
| 추론 (예측) | ✅ 완성 | 실시간 신호 생성 |
| 거래 실행 | ✅ 완성 | Binance API 연동됨 |
| 테스트넷 | ✅ 준비됨 | 설정만 바꾸면 사용 가능 |

---

## 1단계: 프론트엔드 정리 (1-2일)

### 해야 할 일
- 더미 데이터 제거
- 용어 툴팁 추가

### 수정 필요한 파일

| 파일 | 문제 |
|------|------|
| `apps/web/src/app/data-quality/page.tsx` | 하드코딩된 품질 지표 |
| `apps/web/src/app/risk/page.tsx` | 하드코딩된 리스크 블로커 |
| `apps/web/src/components/panels/ExplainPanel.tsx` | 더미 설명값 |
| `apps/web/src/components/panels/LiveOpsPanel.tsx` | 더미 포지션 데이터 |

### 체크리스트
- [ ] 더미 데이터 제거
- [ ] 용어 툴팁 추가

---

## 2단계: 30종목 LightGBM 학습 (3-4시간)

### 실행 명령어

```bash
# 30종목 전체 학습 (권장 설정)
python scripts/train_30_symbols.py

# 메모리 부족 시
python scripts/train_30_symbols.py --skip-optuna --skip-catboost

# 특정 종목만
python scripts/train_30_symbols.py --symbols BTCUSDT,ETHUSDT,DOTUSDT
```

### 결과 목표

| 지표 | 목표값 | 설명 |
|------|--------|------|
| **PF (Profit Factor)** | > 1.5 | 총이익 / 총손실 |
| **Expectancy** | > +0.1% | 거래당 평균 수익률 |

### 용어 설명

| 용어 | 설명 |
|------|------|
| **PF (Profit Factor)** | 총 이익 ÷ 총 손실. 1보다 크면 수익, 2 이상이면 우수 |
| **Expectancy** | 거래 1회당 평균 기대 수익률 |
| **Optuna** | 하이퍼파라미터 자동 튜닝 라이브러리 |
| **CatBoost** | LightGBM과 비슷한 또 다른 ML 알고리즘 |
| **앙상블** | 여러 모델의 예측을 합쳐서 더 좋은 결과를 내는 기법 |

### 체크리스트
- [ ] `python scripts/train_30_symbols.py` 실행
- [ ] PF > 1.5 확인
- [ ] Expectancy > +0.1% 확인

---

## 3단계: 강화학습 (RL) - 선택사항

### 강화학습이란?

```
[일반 ML - LightGBM]
"이 시점에 롱 포지션을 잡으면 수익률이 +0.5%일 것이다"
→ 예측만 함, 행동 결정은 규칙 기반

[강화학습 - RL]
"현재 상황에서 롱/숏/홀드/청산 중 뭘 하면 장기적으로 수익이 최대화될까?"
→ 스스로 행동을 학습
```

### 현재 RL 구현 상태

| 구성요소 | 파일 | 상태 |
|----------|------|------|
| 환경 (Environment) | `services/rl/environment.py` | ✅ 완성 |
| 에이전트 (Agent) | `services/rl/agent.py` | ✅ 완성 |
| 실시간 연동 | `services/realtime_worker.py` | ❌ 미연동 |

### RL 용어 설명

| 용어 | 설명 |
|------|------|
| **State (상태)** | 현재 시장 정보 (가격, 지표, 포지션 등 17개 값) |
| **Action (행동)** | Hold(대기), Long(매수), Short(공매도), Close(청산) |
| **Reward (보상)** | 수익이면 +, 손실이면 - |
| **PPO** | 학습 알고리즘 이름 (Proximal Policy Optimization, 안정적이고 범용적) |
| **Episode** | 한 번의 거래 시뮬레이션 (시작~끝) |
| **stable-baselines3** | RL 알고리즘 라이브러리 (PPO, A2C 등 제공) |

### RL이 필요한가?

**LightGBM만으로도 거래 가능합니다!**

```python
# 간단한 규칙 기반 전략
if LightGBM_예측값 > 0.001:
    롱 진입
elif LightGBM_예측값 < -0.001:
    숏 진입
```

RL은 **추가 최적화** 용도:
- 포지션 사이징 최적화
- 진입/청산 타이밍 미세조정
- 시장 상황에 따른 동적 전략

**권장**: 먼저 LightGBM으로 테스트넷 돌려보고, 결과가 만족스러우면 RL은 나중에

### 체크리스트
- [ ] LightGBM 결과가 좋으면 스킵 가능
- [ ] 필요시 `python scripts/train_rl.py` 실행

---

## 4단계: Shadow 모드 테스트 (1주일)

### Shadow 모드란?

```
실제 거래는 안 하고, "만약 거래했다면" 결과를 시뮬레이션
→ 돈 잃을 위험 없이 전략 검증
```

### 설정 확인

```bash
# .env 파일
TRADING_MODE=SHADOW  # 기본값
```

### 확인 사항

| 항목 | 확인 방법 |
|------|-----------|
| 신호 발생 빈도 | 대시보드 Signals 페이지 |
| Shadow PnL | 대시보드 메인 페이지 |
| 예상 수익률 | Positions 페이지 |
| 시스템 안정성 | 에러 로그 확인 |

### 체크리스트
- [ ] Shadow 모드로 1주일 운영
- [ ] 신호 발생 확인
- [ ] Shadow PnL 확인
- [ ] 시스템 안정성 확인 (24시간 무중단)

---

## 5단계: Binance Testnet (1주일)

### Testnet이란?

```
바이낸스가 제공하는 "가짜 돈" 거래소
→ 실제 API와 동일하게 동작하지만 진짜 돈이 아님
→ 실전과 동일한 환경에서 테스트 가능
```

### 설정 방법

#### 1. Testnet API 키 발급

1. https://testnet.binancefuture.com 접속
2. GitHub 계정으로 로그인
3. API Management에서 API Key 생성

#### 2. 환경 변수 수정

```bash
# .env 파일 수정
BINANCE_TESTNET=true
BINANCE_API_KEY=테스트넷_API_키
BINANCE_SECRET_KEY=테스트넷_시크릿_키
TRADING_MODE=LIVE  # SHADOW → LIVE로 변경
```

#### 3. 실행

```bash
docker compose up -d
```

### 확인 사항 (1주일간)

| 항목 | 설명 |
|------|------|
| 주문 체결 | 실제 주문이 Testnet에서 체결되는지 |
| PnL 검증 | 예상 PnL과 실제 PnL 차이 |
| 슬리피지 | 예상가와 체결가 차이 |
| 시스템 안정성 | 에러 없이 24시간 동작 |

### 용어 설명

| 용어 | 설명 |
|------|------|
| **슬리피지 (Slippage)** | 주문 시 예상 가격과 실제 체결 가격의 차이 |
| **체결 (Fill)** | 주문이 실제로 거래소에서 처리되는 것 |
| **API Key** | 거래소 접근용 인증 키 |
| **Secret Key** | API Key와 함께 사용하는 비밀 키 (절대 노출 금지) |

### 체크리스트
- [ ] Testnet API 키 발급
- [ ] .env 설정 변경
- [ ] 실제 주문 체결 확인
- [ ] 1주일간 PnL 검증

---

## 6단계: 실거래

### 전환 조건

Testnet에서 다음 조건 충족 시:
- [ ] PF > 1.3 유지
- [ ] 시스템 24시간 안정 동작
- [ ] 예상 PnL과 실제 PnL 오차 < 10%

### 설정 변경

```bash
# .env 파일 수정
BINANCE_TESTNET=false  # 실제 거래소로 변경
BINANCE_API_KEY=실제_API_키
BINANCE_SECRET_KEY=실제_시크릿_키
TRADING_MODE=LIVE
```

### 주의사항

| 항목 | 권장 |
|------|------|
| 시작 자본 | 전체 자본의 5-10%로 시작 |
| 모니터링 | 처음 24시간은 상시 모니터링 |
| 손실 한도 | 일일 손실 한도 설정 (예: -3%) |
| 긴급 중지 | 비상 시 즉시 중지 방법 숙지 |

### 리스크 관리 설정

```bash
# .env 파일의 리스크 설정
MAX_POSITION_SIZE=0.05      # 포지션당 최대 5%
MAX_DAILY_LOSS=0.03         # 일일 최대 손실 3%
MAX_POSITIONS=5             # 동시 포지션 최대 5개
LEVERAGE=3                  # 레버리지 3배
```

### 체크리스트
- [ ] 실제 API 키로 변경
- [ ] 소액으로 시작
- [ ] 24시간 모니터링
- [ ] 비상 중지 방법 확인

---

## 용어 사전

### 거래 관련

| 용어 | 설명 |
|------|------|
| **롱 (Long)** | 가격 상승에 베팅. 매수 후 매도 |
| **숏 (Short)** | 가격 하락에 베팅. 빌려서 매도 후 매수 |
| **레버리지** | 빌린 돈으로 거래. 3배 = 100만원으로 300만원치 거래 |
| **청산 (Liquidation)** | 손실이 커서 강제로 포지션 종료됨 |
| **펀딩율 (Funding Rate)** | 롱/숏 균형 유지용 수수료. 8시간마다 발생 |
| **오픈 인터레스트 (OI)** | 현재 열려있는 총 포지션 규모 |

### 학습 관련

| 용어 | 설명 |
|------|------|
| **LightGBM** | 빠르고 효율적인 ML 알고리즘 (Gradient Boosting) |
| **Triple Barrier** | 라벨링 방법. TP(익절)/SL(손절)/시간제한 3개 경계 |
| **피처 (Feature)** | 모델 입력값 (이동평균, RSI 등 기술적 지표) |
| **라벨 (Label)** | 모델이 맞춰야 할 정답 (수익률, TP 도달 여부) |
| **하이퍼파라미터** | 모델 설정값 (트리 개수, 학습률 등) |
| **RMSE** | 예측 오차 측정 지표 (낮을수록 좋음) |

### 시스템 관련

| 용어 | 설명 |
|------|------|
| **웹소켓 (WebSocket)** | 실시간 양방향 통신. 가격 변동 즉시 수신 |
| **REST API** | 요청-응답 방식 통신. 필요할 때 데이터 요청 |
| **MinIO** | S3 호환 저장소. 학습된 모델 파일 저장 |
| **Redis** | 빠른 메모리 DB. 캐시, 실시간 데이터 저장 |
| **PostgreSQL** | 메인 데이터베이스. 캔들, 피처, 거래 기록 저장 |

---

## 예상 일정

| 단계 | 기간 | 누적 |
|------|------|------|
| 1단계: 프론트엔드 정리 | 1-2일 | 2일 |
| 2단계: 30종목 학습 | 3-4시간 | 2-3일 |
| 3단계: RL 학습 (선택) | 1-2일 | 4-5일 |
| 4단계: Shadow 테스트 | 7일 | 12일 |
| 5단계: Testnet | 7일 | 19일 |
| 6단계: 실거래 시작 | - | **약 3주 후** |

---

## 긴급 연락처 / 참고 자료

- Binance Testnet: https://testnet.binancefuture.com
- Binance API 문서: https://binance-docs.github.io/apidocs/futures/en/
- LightGBM 문서: https://lightgbm.readthedocs.io/
- stable-baselines3 문서: https://stable-baselines3.readthedocs.io/

---

*마지막 업데이트: 2026-02-03*
