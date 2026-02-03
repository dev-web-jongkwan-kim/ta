CREATE EXTENSION IF NOT EXISTS "pgcrypto";

ALTER TABLE signals ADD COLUMN IF NOT EXISTS trade_group_id UUID;
UPDATE signals SET trade_group_id = gen_random_uuid() WHERE trade_group_id IS NULL;
ALTER TABLE signals ALTER COLUMN trade_group_id SET NOT NULL;

ALTER TABLE orders ADD COLUMN IF NOT EXISTS trade_group_id UUID;
UPDATE orders SET trade_group_id = gen_random_uuid() WHERE trade_group_id IS NULL;
ALTER TABLE orders ALTER COLUMN trade_group_id SET NOT NULL;

ALTER TABLE fills ADD COLUMN IF NOT EXISTS trade_group_id UUID;
UPDATE fills SET trade_group_id = gen_random_uuid() WHERE trade_group_id IS NULL;
ALTER TABLE fills ALTER COLUMN trade_group_id SET NOT NULL;

ALTER TABLE positions ADD COLUMN IF NOT EXISTS trade_group_id UUID;
UPDATE positions SET trade_group_id = gen_random_uuid() WHERE trade_group_id IS NULL;
ALTER TABLE positions ALTER COLUMN trade_group_id SET NOT NULL;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS event_type TEXT DEFAULT 'FINAL';

ALTER TABLE positions DROP CONSTRAINT IF EXISTS positions_pkey;
ALTER TABLE positions ADD PRIMARY KEY (trade_group_id, symbol, ts, side);
