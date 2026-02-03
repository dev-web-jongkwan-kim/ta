-- 라벨 테이블 인덱스 추가
-- 증분 라벨링 시 symbol + spec_hash 조회 성능 최적화

-- labels_long_1m 인덱스
CREATE INDEX IF NOT EXISTS idx_labels_long_symbol_spec
ON labels_long_1m(symbol, spec_hash);

CREATE INDEX IF NOT EXISTS idx_labels_long_spec_ts
ON labels_long_1m(spec_hash, ts);

-- labels_short_1m 인덱스
CREATE INDEX IF NOT EXISTS idx_labels_short_symbol_spec
ON labels_short_1m(symbol, spec_hash);

CREATE INDEX IF NOT EXISTS idx_labels_short_spec_ts
ON labels_short_1m(spec_hash, ts);
