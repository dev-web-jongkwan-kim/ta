-- Multi-timeframe support: 15m and 1h candles, features, labels

-- 15분 캔들
CREATE TABLE IF NOT EXISTS candles_15m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (symbol, ts)
);

-- 1시간 캔들
CREATE TABLE IF NOT EXISTS candles_1h (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (symbol, ts)
);

-- 15분 피처
CREATE TABLE IF NOT EXISTS features_15m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  schema_version INT NOT NULL,
  features JSONB NOT NULL,
  atr DOUBLE PRECISION,
  funding_z DOUBLE PRECISION,
  btc_regime INT,
  PRIMARY KEY (symbol, ts)
);

-- 1시간 피처
CREATE TABLE IF NOT EXISTS features_1h (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  schema_version INT NOT NULL,
  features JSONB NOT NULL,
  atr DOUBLE PRECISION,
  funding_z DOUBLE PRECISION,
  btc_regime INT,
  PRIMARY KEY (symbol, ts)
);

-- 15분 라벨 (Long)
CREATE TABLE IF NOT EXISTS labels_long_15m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  spec_hash TEXT NOT NULL,
  y SMALLINT NOT NULL,
  ret_net DOUBLE PRECISION NOT NULL,
  mae DOUBLE PRECISION NOT NULL,
  mfe DOUBLE PRECISION NOT NULL,
  time_to_event_min INT NOT NULL,
  PRIMARY KEY (symbol, ts, spec_hash)
);

-- 15분 라벨 (Short)
CREATE TABLE IF NOT EXISTS labels_short_15m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  spec_hash TEXT NOT NULL,
  y SMALLINT NOT NULL,
  ret_net DOUBLE PRECISION NOT NULL,
  mae DOUBLE PRECISION NOT NULL,
  mfe DOUBLE PRECISION NOT NULL,
  time_to_event_min INT NOT NULL,
  PRIMARY KEY (symbol, ts, spec_hash)
);

-- 1시간 라벨 (Long)
CREATE TABLE IF NOT EXISTS labels_long_1h (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  spec_hash TEXT NOT NULL,
  y SMALLINT NOT NULL,
  ret_net DOUBLE PRECISION NOT NULL,
  mae DOUBLE PRECISION NOT NULL,
  mfe DOUBLE PRECISION NOT NULL,
  time_to_event_min INT NOT NULL,
  PRIMARY KEY (symbol, ts, spec_hash)
);

-- 1시간 라벨 (Short)
CREATE TABLE IF NOT EXISTS labels_short_1h (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  spec_hash TEXT NOT NULL,
  y SMALLINT NOT NULL,
  ret_net DOUBLE PRECISION NOT NULL,
  mae DOUBLE PRECISION NOT NULL,
  mfe DOUBLE PRECISION NOT NULL,
  time_to_event_min INT NOT NULL,
  PRIMARY KEY (symbol, ts, spec_hash)
);

-- 멀티 타임프레임 조합 피처 (1분 기준 + 15분/1시간 피처 조인)
CREATE TABLE IF NOT EXISTS features_multi_tf (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  schema_version INT NOT NULL,
  features_1m JSONB NOT NULL,
  features_15m JSONB,
  features_1h JSONB,
  atr DOUBLE PRECISION,
  PRIMARY KEY (symbol, ts)
);

-- 인덱스 추가 (쿼리 성능 향상)
CREATE INDEX IF NOT EXISTS idx_candles_15m_symbol_ts ON candles_15m(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_candles_1h_symbol_ts ON candles_1h(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_features_15m_symbol_ts ON features_15m(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_features_1h_symbol_ts ON features_1h(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_labels_long_15m_spec ON labels_long_15m(spec_hash, ts);
CREATE INDEX IF NOT EXISTS idx_labels_short_15m_spec ON labels_short_15m(spec_hash, ts);
CREATE INDEX IF NOT EXISTS idx_labels_long_1h_spec ON labels_long_1h(spec_hash, ts);
CREATE INDEX IF NOT EXISTS idx_labels_short_1h_spec ON labels_short_1h(spec_hash, ts);