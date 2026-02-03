-- Open Interest 테이블
CREATE TABLE IF NOT EXISTS open_interest (
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    ts TIMESTAMPTZ NOT NULL,
    open_interest DOUBLE PRECISION NOT NULL,      -- OI 수량
    open_interest_value DOUBLE PRECISION,         -- OI 가치 (USD)
    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_open_interest_symbol_ts ON open_interest(symbol, ts DESC);

-- Long/Short 비율 테이블 (전체 + Top Trader + Taker)
CREATE TABLE IF NOT EXISTS long_short_ratio (
    symbol TEXT NOT NULL REFERENCES instruments(symbol),
    ts TIMESTAMPTZ NOT NULL,
    -- Global Long/Short Account Ratio
    long_short_ratio DOUBLE PRECISION,            -- Long/Short 비율
    long_account DOUBLE PRECISION,                -- Long 계정 비율 (0-1)
    short_account DOUBLE PRECISION,               -- Short 계정 비율 (0-1)
    -- Top Trader Long/Short Position Ratio
    top_long_short_ratio DOUBLE PRECISION,        -- Top Trader L/S 비율
    top_long_account DOUBLE PRECISION,            -- Top Trader Long 비율
    top_short_account DOUBLE PRECISION,           -- Top Trader Short 비율
    -- Taker Buy/Sell Volume Ratio
    taker_buy_sell_ratio DOUBLE PRECISION,        -- 매수/매도 비율
    taker_buy_vol DOUBLE PRECISION,               -- 매수 체결량
    taker_sell_vol DOUBLE PRECISION,              -- 매도 체결량
    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_long_short_ratio_symbol_ts ON long_short_ratio(symbol, ts DESC);
