"""
30개 심볼 전체 데이터 파이프라인
1. 캔들 데이터 다운로드 (누락된 심볼)
2. 피처 생성 (1m + 15m + 1h)
3. 라벨 생성 (15m ATR 기반)
4. 모델 재학습

Usage:
    python scripts/full_pipeline_30_symbols.py
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# 22개 심볼 (제외: XAUUSDT, XAGUSDT, HYPEUSDT, ZORAUSDT, PUMPUSDT, PAXGUSDT, ALPHAUSDT, BNXUSDT)
ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "BCHUSDT", "SUIUSDT", "AAVEUSDT", "1000PEPEUSDT",
    "FILUSDT", "AXSUSDT", "ENAUSDT", "ZKUSDT", "ZECUSDT",
    "TRUMPUSDT", "NEARUSDT",
]


def step1_download_candles():
    """Step 1: 누락된 심볼 캔들 다운로드"""
    # download_binance_vision의 함수들을 직접 import
    import io
    import zipfile
    import httpx
    import pandas as pd
    from packages.common.db import bulk_upsert, get_conn, fetch_all

    BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/daily/klines"

    def download_and_parse_zip(url: str):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        return pd.read_csv(f)
        except Exception as e:
            logger.warning(f"Error downloading {url}: {e}")
            return None

    def process_symbol_candles(symbol: str, start_date: datetime, end_date: datetime) -> int:
        total_rows = 0
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            url = f"{BINANCE_VISION_BASE}/{symbol}/1m/{symbol}-1m-{date_str}.zip"
            df = download_and_parse_zip(url)
            if df is not None and not df.empty:
                rows = []
                for _, row in df.iterrows():
                    ts = pd.to_datetime(row["open_time"], unit="ms", utc=True)
                    rows.append((symbol, ts, float(row["open"]), float(row["high"]),
                                float(row["low"]), float(row["close"]), float(row["volume"])))
                if rows:
                    bulk_upsert("candles_1m", ["symbol", "ts", "open", "high", "low", "close", "volume"],
                               rows, conflict_cols=["symbol", "ts"],
                               update_cols=["open", "high", "low", "close", "volume"])
                    total_rows += len(rows)
            current += timedelta(days=1)
        return total_rows

    def ensure_instruments(symbols):
        with get_conn() as conn:
            with conn.cursor() as cur:
                for symbol in symbols:
                    cur.execute("INSERT INTO instruments (symbol, status, liquidity_tier) VALUES (%s, 'active', 1) ON CONFLICT (symbol) DO UPDATE SET status = 'active'", (symbol,))
            conn.commit()

    def generate_premium_index_from_candles(symbols):
        total_rows = 0
        with get_conn() as conn:
            with conn.cursor() as cur:
                for symbol in symbols:
                    cur.execute("""
                        INSERT INTO premium_index (symbol, ts, mark_price, index_price, last_funding_rate, next_funding_time)
                        SELECT symbol, ts, close, close, 0.0, NULL FROM candles_1m WHERE symbol = %s
                        ON CONFLICT (symbol, ts) DO UPDATE SET mark_price = EXCLUDED.mark_price, index_price = EXCLUDED.index_price
                    """, (symbol,))
                    total_rows += cur.rowcount
                conn.commit()
        return total_rows

    logger.info("=" * 60)
    logger.info("Step 1: 캔들 데이터 다운로드")
    logger.info("=" * 60)

    # 현재 캔들 데이터 상태 확인
    rows = fetch_all("""
        SELECT symbol, COUNT(*) as cnt
        FROM candles_1m
        GROUP BY symbol
    """)
    existing = {r[0]: r[1] for r in rows}

    # 다운로드 필요한 심볼 확인 (데이터가 50만개 미만인 것)
    need_download = []
    for symbol in ALL_SYMBOLS:
        cnt = existing.get(symbol, 0)
        if cnt < 500000:
            need_download.append(symbol)
            logger.info(f"  {symbol}: {cnt:,}개 (다운로드 필요)")
        else:
            logger.info(f"  {symbol}: {cnt:,}개 (OK)")

    if not need_download:
        logger.info("모든 심볼 캔들 데이터 충분")
        return

    logger.info(f"\n다운로드 필요: {len(need_download)}개 심볼")

    # 종목 등록
    ensure_instruments(need_download)

    # 1년치 다운로드
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    for i, symbol in enumerate(need_download, 1):
        logger.info(f"\n[{i}/{len(need_download)}] {symbol} 다운로드 중...")
        try:
            count = process_symbol_candles(symbol, start, end)
            logger.info(f"  → {count:,}개 캔들 저장")
        except Exception as e:
            logger.error(f"  → 실패: {e}")

    # Premium index 생성
    logger.info("\nPremium Index 생성 중...")
    generate_premium_index_from_candles(need_download)
    logger.info("Premium Index 완료")


def step2_generate_features():
    """Step 2: 피처 생성 (1m, 15m, 1h)"""
    import json
    import math
    import pandas as pd
    from packages.common.db import fetch_all, bulk_upsert
    from services.features.compute import compute_features_for_symbol

    def _coerce_value(value):
        if value is None:
            return None
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return value

    def _safe_json(payload):
        def _default(obj):
            try:
                return float(obj)
            except Exception:
                return str(obj)
        return json.dumps(payload, default=_default)

    def compute_and_store_for_timeframe(symbol, candle_table, feature_table, timeframe, btc_candles):
        """Compute features for a symbol/timeframe and store to DB."""
        # Load candles
        rows = fetch_all(f"""
            SELECT ts, open, high, low, close, volume
            FROM {candle_table}
            WHERE symbol = %s
            ORDER BY ts
        """, (symbol,))
        if not rows:
            return 0

        candles = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        candles['ts'] = pd.to_datetime(candles['ts'], utc=True)

        # Load premium index
        premium_rows = fetch_all("""
            SELECT ts, mark_price, index_price, last_funding_rate, last_price
            FROM premium_index
            WHERE symbol = %s
            ORDER BY ts
        """, (symbol,))
        if premium_rows:
            premium = pd.DataFrame(premium_rows, columns=['ts', 'mark_price', 'index_price', 'last_funding_rate', 'last_price'])
            premium['ts'] = pd.to_datetime(premium['ts'], utc=True)
        else:
            premium = pd.DataFrame(columns=['ts', 'mark_price', 'index_price', 'last_funding_rate', 'last_price'])

        # Load open interest
        oi_rows = fetch_all("""
            SELECT ts, open_interest
            FROM open_interest
            WHERE symbol = %s
            ORDER BY ts
        """, (symbol,))
        if oi_rows:
            oi_df = pd.DataFrame(oi_rows, columns=['ts', 'open_interest'])
            oi_df['ts'] = pd.to_datetime(oi_df['ts'], utc=True)
        else:
            oi_df = None

        # Load long/short ratio
        ls_rows = fetch_all("""
            SELECT ts, long_short_ratio, long_account, short_account,
                   top_long_short_ratio, top_long_account, top_short_account,
                   taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
            FROM long_short_ratio
            WHERE symbol = %s
            ORDER BY ts
        """, (symbol,))
        if ls_rows:
            ls_df = pd.DataFrame(ls_rows, columns=[
                'ts', 'long_short_ratio', 'long_account', 'short_account',
                'top_long_short_ratio', 'top_long_account', 'top_short_account',
                'taker_buy_sell_ratio', 'taker_buy_vol', 'taker_sell_vol'
            ])
            ls_df['ts'] = pd.to_datetime(ls_df['ts'], utc=True)
        else:
            ls_df = None

        # Compute features
        feats = compute_features_for_symbol(
            symbol,
            candles,
            premium,
            btc_candles,
            open_interest=oi_df,
            long_short_ratio=ls_df,
            timeframe=timeframe,
        )

        if feats.empty:
            return 0

        # Store features - process in batches
        batch_size = 5000
        total_stored = 0

        for i in range(0, len(feats), batch_size):
            batch = feats.iloc[i:i+batch_size]
            rows_to_insert = []

            for _, row in batch.iterrows():
                features_clean = {k: _coerce_value(v) for k, v in row["features"].items()}
                rows_to_insert.append((
                    row["symbol"],
                    row["ts"],
                    int(row["schema_version"]),
                    _safe_json(features_clean),
                    _coerce_value(row.get("atr")),
                    _coerce_value(row.get("funding_z")),
                    int(row["btc_regime"]) if pd.notna(row.get("btc_regime")) else None,
                ))

            bulk_upsert(
                feature_table,
                ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
                rows_to_insert,
                conflict_cols=["symbol", "ts"],
                update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
            )
            total_stored += len(rows_to_insert)

        return total_stored

    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 피처 생성")
    logger.info("=" * 60)

    # Load BTC candles for each timeframe
    btc_candles_cache = {}
    for tf, table in [("1m", "candles_1m"), ("15m", "candles_15m"), ("1h", "candles_1h")]:
        rows = fetch_all(f"""
            SELECT ts, open, high, low, close, volume
            FROM {table}
            WHERE symbol = 'BTCUSDT'
            ORDER BY ts
        """)
        if rows:
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            btc_candles_cache[tf] = df
        else:
            btc_candles_cache[tf] = pd.DataFrame()

    for symbol in ALL_SYMBOLS:
        logger.info(f"\n{symbol} 피처 생성 중...")
        try:
            count_1m = compute_and_store_for_timeframe(
                symbol, "candles_1m", "features_1m", "1m", btc_candles_cache["1m"]
            )
            logger.info(f"  1m: {count_1m:,}개")
        except Exception as e:
            logger.error(f"  1m 실패: {e}")

        try:
            count_15m = compute_and_store_for_timeframe(
                symbol, "candles_15m", "features_15m", "15m", btc_candles_cache["15m"]
            )
            logger.info(f"  15m: {count_15m:,}개")
        except Exception as e:
            logger.error(f"  15m 실패: {e}")

        try:
            count_1h = compute_and_store_for_timeframe(
                symbol, "candles_1h", "features_1h", "1h", btc_candles_cache["1h"]
            )
            logger.info(f"  1h: {count_1h:,}개")
        except Exception as e:
            logger.error(f"  1h 실패: {e}")


def step2b_aggregate_candles():
    """Step 2b: 1m 캔들을 15m, 1h로 집계"""
    from packages.common.db import get_conn

    logger.info("\n" + "=" * 60)
    logger.info("Step 2b: 캔들 집계 (1m → 15m, 1h)")
    logger.info("=" * 60)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # 15m 캔들 집계
            logger.info("15m 캔들 집계 중...")
            cur.execute("""
                INSERT INTO candles_15m (symbol, ts, open, high, low, close, volume)
                SELECT
                    symbol,
                    ts_bucket + interval '14 minutes 59.999 seconds' as ts,
                    (array_agg(open ORDER BY ts))[1] as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    (array_agg(close ORDER BY ts DESC))[1] as close,
                    SUM(volume) as volume
                FROM (
                    SELECT *,
                        date_trunc('hour', ts) + interval '1 minute' * (floor(extract(minute from ts) / 15) * 15) as ts_bucket
                    FROM candles_1m
                    WHERE symbol = ANY(%s)
                ) sub
                GROUP BY symbol, ts_bucket
                HAVING COUNT(*) >= 13
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, (ALL_SYMBOLS,))
            logger.info(f"  15m: {cur.rowcount:,}개")

            # 1h 캔들 집계
            logger.info("1h 캔들 집계 중...")
            cur.execute("""
                INSERT INTO candles_1h (symbol, ts, open, high, low, close, volume)
                SELECT
                    symbol,
                    ts_bucket + interval '59 minutes 59.999 seconds' as ts,
                    (array_agg(open ORDER BY ts))[1] as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    (array_agg(close ORDER BY ts DESC))[1] as close,
                    SUM(volume) as volume
                FROM (
                    SELECT *, date_trunc('hour', ts) as ts_bucket
                    FROM candles_1m
                    WHERE symbol = ANY(%s)
                ) sub
                GROUP BY symbol, ts_bucket
                HAVING COUNT(*) >= 55
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, (ALL_SYMBOLS,))
            logger.info(f"  1h: {cur.rowcount:,}개")

            conn.commit()


def step3_generate_labels():
    """Step 3: 라벨 생성 (15m ATR 기반)"""
    from services.labeling.pipeline import LabelingConfig, label_symbol_incremental

    logger.info("\n" + "=" * 60)
    logger.info("Step 3: 라벨 생성 (15m ATR 기반)")
    logger.info("=" * 60)

    config = LabelingConfig(
        k_tp=1.5,
        k_sl=1.0,
        h_bars=360,
        risk_mae_atr=3.0,
        timeframe="1m",
        atr_timeframe="15m",  # 15m ATR 사용
    )
    spec = config.spec()  # LabelSpec 객체로 변환

    for symbol in ALL_SYMBOLS:
        logger.info(f"\n{symbol} 라벨 생성 중...")
        try:
            result = label_symbol_incremental(symbol, spec, timeframe="1m", atr_timeframe="15m")
            total = result.get('long', 0) + result.get('short', 0)
            logger.info(f"  → Long: {result.get('long', 0):,}, Short: {result.get('short', 0):,}")
        except Exception as e:
            logger.error(f"  → 실패: {e}")


def step4_train_model():
    """Step 4: 모델 학습"""
    from services.training.train import TrainConfig, run_training_job

    logger.info("\n" + "=" * 60)
    logger.info("Step 4: 모델 학습")
    logger.info("=" * 60)

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    # 폭락 구간 포함하여 전체 데이터로 학습
    val_start = yesterday - timedelta(days=2)  # 검증은 최근 2일만

    cfg = TrainConfig(
        label_spec_hash='8c8d570e2343c185',  # 15m ATR 기반 라벨
        feature_schema_version=4,
        train_start='2025-02-02',
        train_end=yesterday.strftime('%Y-%m-%d'),  # 어제까지 전체 학습
        val_start=val_start.strftime('%Y-%m-%d'),
        val_end=yesterday.strftime('%Y-%m-%d'),
        targets=(
            'er_long', 'q05_long', 'e_mae_long', 'e_hold_long',
            'er_short', 'q05_short', 'e_mae_short', 'e_hold_short',
        ),
        algo='lgbm',
        timeframe='1m',
        use_multi_tf=True,
        label_spec={
            'k_tp': 1.5,
            'k_sl': 1.0,
            'h_bars': 360,
            'fee_rate': 0.0004,
            'slippage_k': 0.15,
            'atr_timeframe': '15m',
        }
    )

    logger.info(f"Train: {cfg.train_start} ~ {cfg.train_end}")
    logger.info(f"Val: {cfg.val_start} ~ {cfg.val_end}")
    logger.info("학습 중...")

    result = run_training_job(cfg)

    if result.get('status') == 'empty':
        logger.error(f"학습 실패: {result.get('message')}")
        return

    model_id = result.get('model_id')
    logger.info(f"\n모델 학습 완료: {model_id}")

    # 메트릭 출력
    metrics = result.get('metrics', {})
    filtered_long = metrics.get('filtered_long', {})
    filtered_short = metrics.get('filtered_short', {})

    logger.info("\n=== ER 필터별 성능 ===")
    logger.info(f"{'Filter':<12} {'Long PF':>10} {'Short PF':>10}")
    logger.info("-" * 35)
    for key in ['er>0', 'er>0.0005', 'er>0.001', 'er>0.0015', 'er>0.002']:
        long_pf = filtered_long.get(key, {}).get('profit_factor', 0)
        short_pf = filtered_short.get(key, {}).get('profit_factor', 0)
        logger.info(f"{key:<12} {long_pf:>10.2f} {short_pf:>10.2f}")

    return model_id


def verify_step(step_name: str, expected_symbols: int = None):
    if expected_symbols is None:
        expected_symbols = len(ALL_SYMBOLS)
    """각 단계 후 검증"""
    from packages.common.db import fetch_all

    logger.info(f"\n{'='*40}")
    logger.info(f"[검증] {step_name}")
    logger.info(f"{'='*40}")

    if step_name == "candles":
        rows = fetch_all("""
            SELECT symbol, COUNT(*), MIN(ts), MAX(ts)
            FROM candles_1m
            WHERE symbol = ANY(%s)
            GROUP BY symbol
            ORDER BY symbol
        """, (ALL_SYMBOLS,))
        logger.info(f"캔들 데이터 심볼 수: {len(rows)}/{expected_symbols}")
        missing = set(ALL_SYMBOLS) - set(r[0] for r in rows)
        if missing:
            logger.warning(f"캔들 누락: {missing}")
        return len(rows) >= expected_symbols

    elif step_name == "candles_15m":
        rows = fetch_all("""
            SELECT symbol, COUNT(*)
            FROM candles_15m
            WHERE symbol = ANY(%s)
            GROUP BY symbol
        """, (ALL_SYMBOLS,))
        logger.info(f"15분봉 심볼 수: {len(rows)}/{expected_symbols}")
        missing = set(ALL_SYMBOLS) - set(r[0] for r in rows)
        if missing:
            logger.warning(f"15분봉 누락: {missing}")
        return len(rows) >= expected_symbols

    elif step_name == "features":
        rows = fetch_all("""
            SELECT symbol, COUNT(*) as cnt
            FROM features_1m
            WHERE symbol = ANY(%s)
            GROUP BY symbol
        """, (ALL_SYMBOLS,))
        logger.info(f"피처(1m) 심볼 수: {len(rows)}/{expected_symbols}")
        for r in rows:
            logger.info(f"  {r[0]}: {r[1]:,}개")
        missing = set(ALL_SYMBOLS) - set(r[0] for r in rows)
        if missing:
            logger.warning(f"피처 누락: {missing}")
        return len(rows) >= expected_symbols

    elif step_name == "labels":
        rows_long = fetch_all("""
            SELECT symbol, COUNT(*) as cnt
            FROM labels_long_1m
            WHERE symbol = ANY(%s) AND spec_hash = '8c8d570e2343c185'
            GROUP BY symbol
        """, (ALL_SYMBOLS,))
        rows_short = fetch_all("""
            SELECT symbol, COUNT(*) as cnt
            FROM labels_short_1m
            WHERE symbol = ANY(%s) AND spec_hash = '8c8d570e2343c185'
            GROUP BY symbol
        """, (ALL_SYMBOLS,))
        logger.info(f"라벨(Long) 심볼 수: {len(rows_long)}/{expected_symbols}")
        logger.info(f"라벨(Short) 심볼 수: {len(rows_short)}/{expected_symbols}")
        for r in rows_long:
            logger.info(f"  {r[0]}: Long {r[1]:,}개")
        missing_long = set(ALL_SYMBOLS) - set(r[0] for r in rows_long)
        missing_short = set(ALL_SYMBOLS) - set(r[0] for r in rows_short)
        if missing_long:
            logger.warning(f"Long 라벨 누락: {missing_long}")
        if missing_short:
            logger.warning(f"Short 라벨 누락: {missing_short}")
        return len(rows_long) >= expected_symbols and len(rows_short) >= expected_symbols

    return True


def main():
    logger.info("=" * 60)
    logger.info(f"{len(ALL_SYMBOLS)}개 심볼 전체 파이프라인 시작")
    logger.info("=" * 60)
    logger.info(f"대상 심볼: {len(ALL_SYMBOLS)}개")
    logger.info(f"심볼 목록: {ALL_SYMBOLS}")
    logger.info(f"시작 시간: {datetime.now()}")
    logger.info("")

    try:
        # Step 1: 캔들 다운로드 (이미 완료)
        # step1_download_candles()
        # if not verify_step("candles"):
        #     raise Exception("Step 1 검증 실패: 캔들 데이터 누락")

        # Step 2b: 캔들 집계 (15m, 1h) (이미 완료)
        # step2b_aggregate_candles()
        # if not verify_step("candles_15m"):
        #     raise Exception("Step 2b 검증 실패: 15분봉 데이터 누락")

        # Step 2: 피처 생성 (이미 완료)
        # step2_generate_features()
        # if not verify_step("features"):
        #     raise Exception("Step 2 검증 실패: 피처 데이터 누락")

        # Step 3: 라벨 생성
        step3_generate_labels()
        if not verify_step("labels"):
            raise Exception("Step 3 검증 실패: 라벨 데이터 누락")

        # Step 4: 모델 학습
        model_id = step4_train_model()

        logger.info("\n" + "=" * 60)
        logger.info("파이프라인 완료!")
        logger.info("=" * 60)
        if model_id:
            logger.info(f"새 모델 ID: {model_id}")
            logger.info("프로덕션 적용하려면:")
            logger.info(f"  UPDATE models SET is_production = false WHERE is_production = true;")
            logger.info(f"  UPDATE models SET is_production = true WHERE model_id = '{model_id}';")

    except Exception as e:
        logger.error(f"파이프라인 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
