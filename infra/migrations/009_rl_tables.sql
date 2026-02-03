-- RL 모델 및 결정 기록 테이블

-- RL 모델 테이블
CREATE TABLE IF NOT EXISTS rl_models (
    model_id VARCHAR(32) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    algorithm VARCHAR(20) NOT NULL DEFAULT 'PPO',
    train_start TIMESTAMPTZ NOT NULL,
    train_end TIMESTAMPTZ NOT NULL,
    metrics JSONB,
    model_path VARCHAR(255),
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_models_symbol ON rl_models(symbol);
CREATE INDEX IF NOT EXISTS idx_rl_models_production ON rl_models(is_production);

-- RL 결정 기록 테이블
CREATE TABLE IF NOT EXISTS rl_decisions (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_id VARCHAR(32) REFERENCES rl_models(model_id),
    action INTEGER NOT NULL,  -- 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
    action_name VARCHAR(10) NOT NULL,
    confidence FLOAT,
    action_probs JSONB,  -- 각 액션별 확률
    value_estimate FLOAT,  -- 상태 가치 추정
    model_predictions JSONB,  -- 입력된 모델 예측값
    observation JSONB,  -- 전체 관측 벡터
    position_before FLOAT,
    position_after FLOAT,
    executed BOOLEAN DEFAULT FALSE,  -- 실제 실행 여부
    pnl_result FLOAT,  -- 결과 손익 (후에 업데이트)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_decisions_ts ON rl_decisions(ts DESC);
CREATE INDEX IF NOT EXISTS idx_rl_decisions_symbol ON rl_decisions(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_rl_decisions_model ON rl_decisions(model_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_rl_decisions_action ON rl_decisions(action);

-- RL 학습 에피소드 기록
CREATE TABLE IF NOT EXISTS rl_episodes (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(32) REFERENCES rl_models(model_id),
    episode_num INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    total_reward FLOAT,
    final_balance FLOAT,
    total_trades INTEGER,
    winning_trades INTEGER,
    max_drawdown FLOAT,
    episode_length INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_episodes_model ON rl_episodes(model_id);

-- Production 모델 설정 함수
CREATE OR REPLACE FUNCTION set_rl_production_model(p_model_id VARCHAR, p_symbol VARCHAR)
RETURNS VOID AS $$
BEGIN
    -- 해당 심볼의 기존 production 해제
    UPDATE rl_models
    SET is_production = FALSE
    WHERE symbol = p_symbol AND is_production = TRUE;

    -- 새 모델을 production으로 설정
    UPDATE rl_models
    SET is_production = TRUE
    WHERE model_id = p_model_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE rl_models IS 'RL 에이전트 모델 정보';
COMMENT ON TABLE rl_decisions IS 'RL 에이전트 결정 기록 (모든 결정 저장)';
COMMENT ON TABLE rl_episodes IS 'RL 학습 에피소드 기록';
