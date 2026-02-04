-- Trading sessions table for manual start/stop control
CREATE TABLE IF NOT EXISTS trading_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mode VARCHAR(10) NOT NULL CHECK (mode IN ('shadow', 'live')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stopped_at TIMESTAMPTZ,
    initial_capital DECIMAL(18,8),
    final_capital DECIMAL(18,8),
    total_trades INT DEFAULT 0,
    wins INT DEFAULT 0,
    losses INT DEFAULT 0,
    total_pnl DECIMAL(18,8) DEFAULT 0,
    gross_profit DECIMAL(18,8) DEFAULT 0,
    gross_loss DECIMAL(18,8) DEFAULT 0,
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(8,4),
    avg_hold_min INT,
    best_trade DECIMAL(18,8),
    worst_trade DECIMAL(18,8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for active session lookup
CREATE INDEX IF NOT EXISTS idx_trading_sessions_active
ON trading_sessions (stopped_at) WHERE stopped_at IS NULL;

-- Trades table for completed trade tracking
CREATE TABLE IF NOT EXISTS trades (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES trading_sessions(session_id),
    trade_group_id UUID NOT NULL,
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    entry_price DECIMAL(18,8) NOT NULL,
    exit_price DECIMAL(18,8),
    qty DECIMAL(18,8) NOT NULL,
    pnl DECIMAL(18,8),
    return_pct DECIMAL(8,4),
    hold_min INT,
    exit_reason TEXT,  -- 'TP', 'SL', 'MANUAL', 'TIMEOUT'
    is_shadow BOOLEAN NOT NULL DEFAULT FALSE,
    entry_order_id BIGINT,
    exit_order_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for trade queries
CREATE INDEX IF NOT EXISTS idx_trades_session ON trades (session_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_shadow ON trades (is_shadow);

-- Add is_shadow column to orders if not exists
ALTER TABLE orders ADD COLUMN IF NOT EXISTS is_shadow BOOLEAN DEFAULT FALSE;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS session_id UUID REFERENCES trading_sessions(session_id);

-- Add session_id to signals
ALTER TABLE signals ADD COLUMN IF NOT EXISTS session_id UUID REFERENCES trading_sessions(session_id);
