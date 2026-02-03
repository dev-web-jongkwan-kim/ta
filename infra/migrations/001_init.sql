CREATE TABLE IF NOT EXISTS instruments (
  symbol TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'active',
  liquidity_tier TEXT NOT NULL DEFAULT 'A',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS candles_1m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS premium_index (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  mark_price DOUBLE PRECISION NOT NULL,
  index_price DOUBLE PRECISION,
  last_price DOUBLE PRECISION,
  last_funding_rate DOUBLE PRECISION,
  next_funding_time TIMESTAMPTZ,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS funding_rates (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  funding_rate DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS features_1m (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  schema_version INT NOT NULL,
  features JSONB NOT NULL,
  atr DOUBLE PRECISION,
  funding_z DOUBLE PRECISION,
  btc_regime INT,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS labels_long_1m (
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

CREATE TABLE IF NOT EXISTS labels_short_1m (
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

CREATE TABLE IF NOT EXISTS training_jobs (
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

CREATE TABLE IF NOT EXISTS models (
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

CREATE TABLE IF NOT EXISTS signals (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  model_id UUID NOT NULL REFERENCES models(model_id),
  ev_long DOUBLE PRECISION NOT NULL,
  ev_short DOUBLE PRECISION NOT NULL,
  er_long DOUBLE PRECISION,
  er_short DOUBLE PRECISION,
  q05_long DOUBLE PRECISION,
  q05_short DOUBLE PRECISION,
  e_mae_long DOUBLE PRECISION,
  e_mae_short DOUBLE PRECISION,
  e_hold_long_min INT,
  e_hold_short_min INT,
  decision TEXT NOT NULL,
  size_notional DOUBLE PRECISION,
  leverage INT,
  sl_price DOUBLE PRECISION,
  tp_price DOUBLE PRECISION,
  block_reason_codes JSONB,
  explain JSONB,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS account_snapshots (
  ts TIMESTAMPTZ PRIMARY KEY,
  equity DOUBLE PRECISION NOT NULL,
  wallet_balance DOUBLE PRECISION NOT NULL,
  unrealized_pnl DOUBLE PRECISION NOT NULL,
  available_margin DOUBLE PRECISION NOT NULL,
  used_margin DOUBLE PRECISION NOT NULL,
  margin_ratio DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  ts TIMESTAMPTZ NOT NULL,
  side TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS orders (
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

CREATE TABLE IF NOT EXISTS fills (
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

CREATE TABLE IF NOT EXISTS risk_events (
  ts TIMESTAMPTZ NOT NULL,
  event_type TEXT NOT NULL,
  symbol TEXT NOT NULL DEFAULT '',
  severity INT NOT NULL,
  message TEXT NOT NULL,
  details JSONB,
  PRIMARY KEY (ts, event_type, symbol)
);

CREATE TABLE IF NOT EXISTS drift_metrics (
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  schema_version INT NOT NULL,
  psi DOUBLE PRECISION,
  missing_rate DOUBLE PRECISION,
  latency_ms DOUBLE PRECISION,
  PRIMARY KEY (symbol, ts, schema_version)
);
