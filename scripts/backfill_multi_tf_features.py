"""Backfill 15m and 1h features for the gap period."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from packages.common.db import bulk_upsert, fetch_all
from services.features.compute import compute_features_for_symbol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _coerce_value(v):
    """Convert value to JSON-safe format."""
    if v is None or (isinstance(v, float) and (pd.isna(v) or v != v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _safe_json(d):
    """Convert dict to JSON-safe format."""
    import json
    return json.dumps({k: _coerce_value(v) for k, v in d.items()})


def load_candles(symbol: str, table: str, limit: int = 500) -> pd.DataFrame:
    """Load candles from database."""
    rows = fetch_all(
        f"""
        SELECT symbol, ts, open, high, low, close, volume
        FROM {table}
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    return df.sort_values("ts")


def load_premium(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Load premium data."""
    rows = fetch_all(
        """
        SELECT symbol, ts, mark_price, index_price, last_price, last_funding_rate, next_funding_time
        FROM premium_index
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(
        rows,
        columns=["symbol", "ts", "mark_price", "index_price", "last_price", "last_funding_rate", "next_funding_time"],
    )
    return df.sort_values("ts")


def load_open_interest(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Load open interest data."""
    rows = fetch_all(
        """
        SELECT symbol, ts, open_interest, open_interest_value
        FROM open_interest
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open_interest", "open_interest_value"])
    return df.sort_values("ts")


def load_long_short_ratio(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Load long/short ratio data."""
    rows = fetch_all(
        """
        SELECT symbol, ts, long_short_ratio, long_account, short_account,
               top_long_short_ratio, top_long_account, top_short_account,
               taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
        FROM long_short_ratio
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(
        rows,
        columns=[
            "symbol", "ts", "long_short_ratio", "long_account", "short_account",
            "top_long_short_ratio", "top_long_account", "top_short_account",
            "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol",
        ],
    )
    return df.sort_values("ts")


def get_latest_feature_ts(symbol: str, table: str) -> datetime | None:
    """Get the latest feature timestamp for a symbol."""
    rows = fetch_all(
        f"SELECT MAX(ts) FROM {table} WHERE symbol = %s",
        (symbol,),
    )
    if rows and rows[0][0]:
        return rows[0][0]
    return None


def compute_and_store_features(
    symbol: str,
    candle_table: str,
    feature_table: str,
    timeframe: str,
) -> int:
    """Compute and store features for a symbol and timeframe."""
    # Get latest feature timestamp
    latest_feature_ts = get_latest_feature_ts(symbol, feature_table)

    # Load candles
    candles = load_candles(symbol, candle_table, limit=500)
    if candles.empty or len(candles) < 30:
        logger.warning(f"Not enough {candle_table} data for {symbol}: {len(candles)}")
        return 0

    # Load other data
    premium = load_premium(symbol)
    btc_candles = load_candles("BTCUSDT", candle_table, limit=500) if symbol != "BTCUSDT" else candles
    open_interest_df = load_open_interest(symbol)
    long_short_ratio_df = load_long_short_ratio(symbol)

    # Compute features
    feats = compute_features_for_symbol(
        symbol,
        candles,
        premium,
        btc_candles,
        open_interest=open_interest_df,
        long_short_ratio=long_short_ratio_df,
        timeframe=timeframe,
    )

    if feats.empty:
        logger.warning(f"No features computed for {symbol}")
        return 0

    # Filter to only new features (after latest_feature_ts)
    if latest_feature_ts:
        feats = feats[feats["ts"] > latest_feature_ts]

    if feats.empty:
        logger.info(f"No new features to insert for {symbol}")
        return 0

    # Prepare rows for insertion
    rows = []
    for _, row in feats.iterrows():
        features_clean = {k: _coerce_value(v) for k, v in row["features"].items()}
        rows.append((
            row["symbol"],
            row["ts"],
            int(row["schema_version"]),
            _safe_json(features_clean),
            _coerce_value(row.get("atr")),
            _coerce_value(row.get("funding_z")),
            int(row["btc_regime"]) if pd.notna(row.get("btc_regime")) else None,
        ))

    # Bulk insert
    bulk_upsert(
        feature_table,
        ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
        rows,
        conflict_cols=["symbol", "ts"],
        update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
    )

    logger.info(f"Inserted {len(rows)} features for {symbol} into {feature_table}")
    return len(rows)


def main():
    """Main function to backfill features."""
    # Get universe
    rows = fetch_all("SELECT symbol FROM instruments ORDER BY symbol")
    symbols = [r[0] for r in rows]

    logger.info(f"Backfilling features for {len(symbols)} symbols")

    total_15m = 0
    total_1h = 0

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")

        # 15m features
        count = compute_and_store_features(
            symbol,
            candle_table="candles_15m",
            feature_table="features_15m",
            timeframe="15m",
        )
        total_15m += count

        # 1h features
        count = compute_and_store_features(
            symbol,
            candle_table="candles_1h",
            feature_table="features_1h",
            timeframe="1h",
        )
        total_1h += count

    logger.info(f"Backfill complete: {total_15m} 15m features, {total_1h} 1h features")


if __name__ == "__main__":
    main()
