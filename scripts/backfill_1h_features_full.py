"""Backfill ALL 1h features from the beginning."""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
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
    return json.dumps({k: _coerce_value(v) for k, v in d.items()})


def load_all_candles(symbol: str, table: str) -> pd.DataFrame:
    """Load ALL candles from database (no limit)."""
    rows = fetch_all(
        f"""
        SELECT symbol, ts, open, high, low, close, volume
        FROM {table}
        WHERE symbol = %s
        ORDER BY ts
        """,
        (symbol,),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    return df


def load_all_premium(symbol: str) -> pd.DataFrame:
    """Load ALL premium data."""
    rows = fetch_all(
        """
        SELECT symbol, ts, mark_price, index_price, last_price, last_funding_rate, next_funding_time
        FROM premium_index
        WHERE symbol = %s
        ORDER BY ts
        """,
        (symbol,),
    )
    df = pd.DataFrame(
        rows,
        columns=["symbol", "ts", "mark_price", "index_price", "last_price", "last_funding_rate", "next_funding_time"],
    )
    return df


def load_all_open_interest(symbol: str) -> pd.DataFrame:
    """Load ALL open interest data."""
    rows = fetch_all(
        """
        SELECT symbol, ts, open_interest, open_interest_value
        FROM open_interest
        WHERE symbol = %s
        ORDER BY ts
        """,
        (symbol,),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open_interest", "open_interest_value"])
    return df


def load_all_long_short_ratio(symbol: str) -> pd.DataFrame:
    """Load ALL long/short ratio data."""
    rows = fetch_all(
        """
        SELECT symbol, ts, long_short_ratio, long_account, short_account,
               top_long_short_ratio, top_long_account, top_short_account,
               taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
        FROM long_short_ratio
        WHERE symbol = %s
        ORDER BY ts
        """,
        (symbol,),
    )
    df = pd.DataFrame(
        rows,
        columns=[
            "symbol", "ts", "long_short_ratio", "long_account", "short_account",
            "top_long_short_ratio", "top_long_account", "top_short_account",
            "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol",
        ],
    )
    return df


def compute_and_store_features_full(
    symbol: str,
    candle_table: str,
    feature_table: str,
    timeframe: str,
    btc_candles: pd.DataFrame,
) -> int:
    """Compute and store ALL features for a symbol and timeframe."""
    # Load ALL candles
    candles = load_all_candles(symbol, candle_table)
    if candles.empty or len(candles) < 30:
        logger.warning(f"Not enough {candle_table} data for {symbol}: {len(candles)}")
        return 0

    logger.info(f"  Loaded {len(candles):,} candles for {symbol}")

    # Load other data
    premium = load_all_premium(symbol)
    open_interest_df = load_all_open_interest(symbol)
    long_short_ratio_df = load_all_long_short_ratio(symbol)

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

    # Delete existing features for this symbol (full rebuild)
    from packages.common.db import execute
    execute(f"DELETE FROM {feature_table} WHERE symbol = %s", (symbol,))

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

    # Bulk insert in batches
    batch_size = 10000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        bulk_upsert(
            feature_table,
            ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
            batch,
            conflict_cols=["symbol", "ts"],
            update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
        )
        logger.info(f"    Inserted batch {i//batch_size + 1}: {len(batch)} rows")

    logger.info(f"  Total: {len(rows)} features for {symbol}")
    return len(rows)


def main():
    """Main function to backfill ALL 1h features."""
    # Get universe
    rows = fetch_all("SELECT symbol FROM instruments ORDER BY symbol")
    symbols = [r[0] for r in rows]

    logger.info(f"Backfilling 1h features for {len(symbols)} symbols (FULL REBUILD)")

    # Load BTC candles once (for all symbols)
    btc_candles = load_all_candles("BTCUSDT", "candles_1h")
    logger.info(f"Loaded {len(btc_candles):,} BTC 1h candles as reference")

    total_1h = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")

        count = compute_and_store_features_full(
            symbol,
            candle_table="candles_1h",
            feature_table="features_1h",
            timeframe="1h",
            btc_candles=btc_candles,
        )
        total_1h += count

    logger.info(f"Backfill complete: {total_1h:,} total 1h features")


if __name__ == "__main__":
    main()
