"""Websocket-based real-time trading worker."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import httpx
import pandas as pd

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, fetch_all
from packages.common.runtime import set_collector_ok
from packages.common.symbol_map import to_internal
from packages.common.time import ms_to_dt
from packages.common.trade_group import build_trade_group_id
from packages.common.bus import RedisBus
from packages.common.ws_status import set_ws_streams
from services.collector.buffer import IngestBuffer
from services.collector.market_data import (
    fetch_premium_index,
    fetch_open_interest,
    fetch_long_short_ratio,
    fetch_top_long_short_ratio,
    fetch_taker_buy_sell_ratio,
)
from services.features.compute import compute_features_for_symbol
from services.inference.predictor import Predictor
from services.policy.decide import PolicyConfig, decide
from services.registry.ensure_default import ensure_default_model
from services.websocket.connection import WebsocketManager
from services.websocket.stream_router import StreamRouter
from services.websocket.handlers.kline_handler import KlineHandler
from services.websocket.handlers.mark_price_handler import MarkPriceHandler
from services.websocket.handlers.book_ticker_handler import BookTickerHandler
from services.engine.session_manager import session_manager
from services.engine.position_manager import position_manager
from packages.common.runtime import get_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Data freshness settings (timeframe-specific max gaps)
DATA_FRESHNESS_MAX_GAP = {
    "1m": 5,      # 5 minutes max gap for 1m candles
    "15m": 30,    # 30 minutes max gap for 15m candles
    "1h": 90,     # 90 minutes max gap for 1h candles (allows for current hour)
}
REQUIRED_HISTORY_HOURS = 60  # 60 hours of history required for features
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def check_data_freshness(symbol: str, timeframe: str = "15m") -> Tuple[bool, float]:
    """
    Check if candle data is fresh enough for trading.

    Returns:
        Tuple of (is_fresh, gap_minutes)
    """
    table = f"candles_{timeframe}"
    rows = fetch_all(
        f"SELECT MAX(ts) FROM {table} WHERE symbol = %s",
        (symbol,),
    )

    if not rows or not rows[0][0]:
        return False, float('inf')

    latest_ts = rows[0][0]
    now = datetime.now(timezone.utc)
    gap = (now - latest_ts).total_seconds() / 60

    max_gap = DATA_FRESHNESS_MAX_GAP.get(timeframe, 30)
    is_fresh = gap <= max_gap
    return is_fresh, gap


def check_sufficient_history(symbol: str, timeframe: str = "15m") -> Tuple[bool, int]:
    """
    Check if we have enough historical data for feature calculation.

    Returns:
        Tuple of (has_enough, candle_count)
    """
    table = f"candles_{timeframe}"
    # Calculate required candles based on timeframe
    if timeframe == "15m":
        required_candles = int(REQUIRED_HISTORY_HOURS * 60 / 15)  # 240 candles
    elif timeframe == "1h":
        required_candles = int(REQUIRED_HISTORY_HOURS)  # 60 candles
    else:
        required_candles = int(REQUIRED_HISTORY_HOURS * 60)  # 3600 candles for 1m

    rows = fetch_all(
        f"SELECT COUNT(*) FROM {table} WHERE symbol = %s",
        (symbol,),
    )

    count = rows[0][0] if rows else 0
    return count >= required_candles, count


def fetch_binance_klines(symbol: str, interval: str, limit: int = 1500) -> List[Dict]:
    """Fetch klines from Binance Futures API."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                BINANCE_KLINES_URL,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch klines for {symbol} {interval}: {e}")
        return []


def backfill_candles_for_symbol(symbol: str, timeframe: str = "15m") -> int:
    """
    Backfill missing candles for a symbol from Binance API.

    Returns:
        Number of candles inserted
    """
    table = f"candles_{timeframe}"

    # Map timeframe to Binance interval
    interval_map = {"1m": "1m", "15m": "15m", "1h": "1h"}
    interval = interval_map.get(timeframe, "15m")

    # Calculate how many candles we need
    if timeframe == "15m":
        required_candles = int(REQUIRED_HISTORY_HOURS * 60 / 15) + 50  # Extra buffer
    elif timeframe == "1h":
        required_candles = int(REQUIRED_HISTORY_HOURS) + 10
    else:
        required_candles = 1500  # Max for 1m

    # Fetch from Binance
    klines = fetch_binance_klines(symbol, interval, min(required_candles, 1500))

    if not klines:
        return 0

    rows = []
    for k in klines:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        rows.append((
            symbol,
            ts,
            float(k[1]),  # open
            float(k[2]),  # high
            float(k[3]),  # low
            float(k[4]),  # close
            float(k[5]),  # volume
        ))

    if rows:
        bulk_upsert(
            table,
            ["symbol", "ts", "open", "high", "low", "close", "volume"],
            rows,
            conflict_cols=["symbol", "ts"],
            update_cols=["open", "high", "low", "close", "volume"],
        )

    return len(rows)


def ensure_data_ready(symbols: List[str]) -> bool:
    """
    Ensure all symbols have fresh and sufficient data before trading.
    Downloads missing data if needed.

    Returns:
        True if all symbols are ready, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Checking data readiness...")
    logger.info("=" * 60)

    all_ready = True
    timeframes = ["1m", "15m", "1h"]

    for symbol in symbols:
        symbol_ready = True

        for tf in timeframes:
            # Check freshness
            is_fresh, gap = check_data_freshness(symbol, tf)
            has_history, count = check_sufficient_history(symbol, tf)

            if not is_fresh or not has_history:
                logger.info(f"[BACKFILL] {symbol} {tf}: fresh={is_fresh} (gap={gap:.1f}min), history={has_history} ({count} candles)")

                # Backfill if needed
                inserted = backfill_candles_for_symbol(symbol, tf)
                logger.info(f"[BACKFILL] {symbol} {tf}: Downloaded {inserted} candles")

                # Re-check
                is_fresh, gap = check_data_freshness(symbol, tf)
                has_history, count = check_sufficient_history(symbol, tf)

                if not is_fresh:
                    logger.warning(f"[BACKFILL] {symbol} {tf}: Still not fresh after backfill (gap={gap:.1f}min)")
                    symbol_ready = False

                if not has_history:
                    logger.warning(f"[BACKFILL] {symbol} {tf}: Still insufficient history ({count} candles)")
                    symbol_ready = False

        if symbol_ready:
            logger.info(f"[READY] {symbol}: All timeframes ready ✓")
        else:
            all_ready = False

    logger.info("=" * 60)
    if all_ready:
        logger.info("All symbols ready for trading!")
    else:
        logger.warning("Some symbols have data issues - trading may be affected")
    logger.info("=" * 60)

    return all_ready


def _load_candles(symbol: str, limit: int = 500) -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles_1m
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    return df.sort_values("ts")


def _load_candles_15m(symbol: str, limit: int = 200) -> pd.DataFrame:
    """Load 15m candles for feature computation."""
    rows = fetch_all(
        """
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles_15m
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    return df.sort_values("ts")


def _load_candles_1h(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Load 1h candles for feature computation."""
    rows = fetch_all(
        """
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles_1h
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    return df.sort_values("ts")


def _active_universe(defaults: List[str]) -> List[str]:
    rows = fetch_all("SELECT symbol FROM instruments WHERE status='active' ORDER BY symbol")
    if not rows:
        return defaults
    return [r[0] for r in rows]


def _load_premium(symbol: str, limit: int = 500) -> pd.DataFrame:
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


def _load_open_interest(symbol: str, limit: int = 500) -> pd.DataFrame:
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


def _load_long_short_ratio(symbol: str, limit: int = 500) -> pd.DataFrame:
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


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return value


def _safe_json(payload: Any) -> str:
    def _default(obj: Any) -> Any:
        try:
            return float(obj)
        except Exception:  # noqa: BLE001
            return str(obj)
    return json.dumps(payload, default=_default)


def _load_latest_features(symbol: str, table: str) -> Dict[str, float]:
    """Load latest features from DB for a symbol."""
    rows = fetch_all(
        f"SELECT features FROM {table} WHERE symbol = %s ORDER BY ts DESC LIMIT 1",
        (symbol,),
    )
    if not rows:
        return {}
    features = rows[0][0]
    if isinstance(features, (bytes, bytearray)):
        features = features.decode()
    if isinstance(features, str):
        import json as _json
        features = _json.loads(features)
    if not isinstance(features, dict):
        return {}
    return {k: _coerce_value(v) for k, v in features.items()}


def _merge_multi_tf_features(
    features_1m: Dict[str, float],
    symbol: str,
) -> Dict[str, float]:
    """Merge 1m, 15m, 1h features with prefixes for model inference."""
    # Add f_1m_ prefix to 1m features
    merged = {f"f_1m_{k}": v for k, v in features_1m.items()}

    # Load and add 15m features
    features_15m = _load_latest_features(symbol, "features_15m")
    for k, v in features_15m.items():
        merged[f"f_15m_{k}"] = v

    # Load and add 1h features
    features_1h = _load_latest_features(symbol, "features_1h")
    for k, v in features_1h.items():
        merged[f"f_1h_{k}"] = v

    return merged


def _compute_and_store_features_15m(symbol: str) -> None:
    """Compute and store 15m features when a 15m candle closes."""
    try:
        candles = _load_candles_15m(symbol)
        if candles.empty or len(candles) < 20:
            logger.debug(f"Not enough 15m candles for {symbol}: {len(candles)}")
            return

        premium = _load_premium(symbol)
        btc_candles = _load_candles_15m("BTCUSDT") if symbol != "BTCUSDT" else candles
        open_interest_df = _load_open_interest(symbol)
        long_short_ratio_df = _load_long_short_ratio(symbol)

        feats = compute_features_for_symbol(
            symbol,
            candles,
            premium,
            btc_candles,
            open_interest=open_interest_df,
            long_short_ratio=long_short_ratio_df,
            timeframe="15m",
        )

        if feats.empty:
            return

        latest = feats.iloc[-1]
        features_clean = {k: _coerce_value(v) for k, v in latest["features"].items()}

        bulk_upsert(
            "features_15m",
            ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
            [
                (
                    latest["symbol"],
                    latest["ts"],
                    int(latest["schema_version"]),
                    _safe_json(features_clean),
                    _coerce_value(latest.get("atr")),
                    _coerce_value(latest.get("funding_z")),
                    int(latest["btc_regime"]) if pd.notna(latest.get("btc_regime")) else None,
                )
            ],
            conflict_cols=["symbol", "ts"],
            update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
        )
        logger.info(f"[FEATURES] Computed and stored 15m features for {symbol}")

    except Exception as e:
        logger.error(f"Failed to compute 15m features for {symbol}: {e}")


def _compute_and_store_features_1h(symbol: str) -> None:
    """Compute and store 1h features when a 1h candle closes."""
    try:
        candles = _load_candles_1h(symbol)
        if candles.empty or len(candles) < 20:
            logger.debug(f"Not enough 1h candles for {symbol}: {len(candles)}")
            return

        premium = _load_premium(symbol)
        btc_candles = _load_candles_1h("BTCUSDT") if symbol != "BTCUSDT" else candles
        open_interest_df = _load_open_interest(symbol)
        long_short_ratio_df = _load_long_short_ratio(symbol)

        feats = compute_features_for_symbol(
            symbol,
            candles,
            premium,
            btc_candles,
            open_interest=open_interest_df,
            long_short_ratio=long_short_ratio_df,
            timeframe="1h",
        )

        if feats.empty:
            return

        latest = feats.iloc[-1]
        features_clean = {k: _coerce_value(v) for k, v in latest["features"].items()}

        bulk_upsert(
            "features_1h",
            ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
            [
                (
                    latest["symbol"],
                    latest["ts"],
                    int(latest["schema_version"]),
                    _safe_json(features_clean),
                    _coerce_value(latest.get("atr")),
                    _coerce_value(latest.get("funding_z")),
                    int(latest["btc_regime"]) if pd.notna(latest.get("btc_regime")) else None,
                )
            ],
            conflict_cols=["symbol", "ts"],
            update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
        )
        logger.info(f"[FEATURES] Computed and stored 1h features for {symbol}")

    except Exception as e:
        logger.error(f"Failed to compute 1h features for {symbol}: {e}")


class RealtimeWorker:
    """Websocket-based real-time trading worker."""

    def __init__(self):
        """Initialize the worker."""
        self.settings = get_settings()
        self.bus = RedisBus()
        self.buffer = IngestBuffer()
        self.predictor = Predictor()
        self.model_id = ensure_default_model()

        # Websocket components
        self.ws_manager: WebsocketManager | None = None
        self.router = StreamRouter()
        self.kline_handler: KlineHandler | None = None
        self.mark_price_handler: MarkPriceHandler | None = None
        self.book_ticker_handler: BookTickerHandler | None = None

        # Background tasks
        self.rest_poller_task: asyncio.Task | None = None
        self.buffer_flusher_task: asyncio.Task | None = None
        self.premium_aggregator_task: asyncio.Task | None = None

        # Lock to prevent concurrent position entries (race condition fix)
        self._position_entry_lock = asyncio.Lock()

    def on_1m_candle_close(self, symbol: str, close_time: datetime) -> None:
        """
        Callback triggered when a 1m candle closes.

        This is where inference is triggered.
        """
        logger.info(f"1m candle closed: {symbol} @ {close_time}")

        try:
            # Run inference in background
            asyncio.create_task(self.run_inference(symbol))
        except Exception as e:
            logger.error(f"Failed to trigger inference for {symbol}: {e}")

    def on_15m_candle_close(self, symbol: str, close_time: datetime) -> None:
        """
        Callback triggered when a 15m candle closes.

        Computes and stores 15m features.
        """
        try:
            # Flush buffer first to ensure candle is in DB
            self.buffer.flush()
            # Compute features in background
            asyncio.get_event_loop().run_in_executor(
                None, _compute_and_store_features_15m, symbol
            )
        except Exception as e:
            logger.error(f"Failed to compute 15m features for {symbol}: {e}")

    def on_1h_candle_close(self, symbol: str, close_time: datetime) -> None:
        """
        Callback triggered when a 1h candle closes.

        Computes and stores 1h features.
        """
        try:
            # Flush buffer first to ensure candle is in DB
            self.buffer.flush()
            # Compute features in background
            asyncio.get_event_loop().run_in_executor(
                None, _compute_and_store_features_1h, symbol
            )
        except Exception as e:
            logger.error(f"Failed to compute 1h features for {symbol}: {e}")

    async def run_inference(self, symbol: str) -> None:
        """Run inference for a symbol."""
        try:
            # Check data freshness before trading
            is_fresh, gap = check_data_freshness(symbol, "15m")
            if not is_fresh:
                logger.warning(
                    f"[STALE DATA] {symbol}: 15m data is {gap:.1f} min old (max={DATA_FRESHNESS_MAX_GAP.get('15m', 30)}min), skipping trade"
                )
                return

            candles = _load_candles(symbol)
            if candles.empty:
                return

            premium = _load_premium(symbol)
            btc_candles = _load_candles("BTCUSDT") if symbol != "BTCUSDT" else candles
            open_interest_df = _load_open_interest(symbol)
            long_short_ratio_df = _load_long_short_ratio(symbol)

            feats = compute_features_for_symbol(
                symbol,
                candles,
                premium,
                btc_candles,
                open_interest=open_interest_df,
                long_short_ratio=long_short_ratio_df,
            )

            if feats.empty:
                return

            latest = feats.iloc[-1]
            features_clean = {k: _coerce_value(v) for k, v in latest["features"].items()}

            # Store features
            bulk_upsert(
                "features_1m",
                ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
                [
                    (
                        latest["symbol"],
                        latest["ts"],
                        int(latest["schema_version"]),
                        _safe_json(features_clean),
                        _coerce_value(latest.get("atr")),
                        _coerce_value(latest.get("funding_z")),
                        int(latest["btc_regime"]) if pd.notna(latest.get("btc_regime")) else None,
                    )
                ],
                conflict_cols=["symbol", "ts"],
                update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
            )

            # Run prediction with multi-timeframe features (model trained on 1m + 15m + 1h)
            # Load features from each timeframe with appropriate prefixes
            model_features = {}

            # 1m features (from current candle processing)
            for k, v in features_clean.items():
                model_features[f"f_1m_{k}"] = v

            # 15m features
            features_15m = _load_latest_features(symbol, "features_15m")
            if features_15m:
                for k, v in features_15m.items():
                    model_features[f"f_15m_{k}"] = v

            # 1h features
            features_1h = _load_latest_features(symbol, "features_1h")
            if features_1h:
                for k, v in features_1h.items():
                    model_features[f"f_1h_{k}"] = v

            preds = self.predictor.predict(model_features)

            # Make decision (simplified - no risk checks for now)
            policy_cfg = PolicyConfig(
                self.settings.ev_min,
                self.settings.q05_min,
                self.settings.mae_max,
                self.settings.max_positions,
            )
            last_close = float(candles.iloc[-1]["close"]) if not candles.empty else 0.0

            # Use 15m ATR for SL/TP calculation (wider stops, matches training data)
            features_15m = _load_latest_features(symbol, "features_15m")
            atr_15m = float(features_15m.get("atr", 0)) if features_15m else 0.0
            # Fallback to 1m ATR * 10 if 15m ATR not available
            if atr_15m == 0:
                atr_1m = float(latest["atr"]) if pd.notna(latest.get("atr")) else 0.0
                atr_15m = atr_1m * 10  # Approximate 15m ATR

            # Preliminary decision to determine direction
            state = {"equity": 10000.0, "sl_price": 0, "tp_price": 0}
            decision = decide(symbol, preds, state, policy_cfg)

            # Calculate SL/TP based on direction (k_sl=1.0, k_tp=1.5 matching training labels)
            # Using 15m ATR for wider stops that match training labels
            k_sl = 1.0
            k_tp = 1.5
            if decision["decision"] == "LONG":
                sl_price = last_close - atr_15m * k_sl  # SL below for LONG
                tp_price = last_close + atr_15m * k_tp  # TP above for LONG
            elif decision["decision"] == "SHORT":
                sl_price = last_close + atr_15m * k_sl  # SL above for SHORT
                tp_price = last_close - atr_15m * k_tp  # TP below for SHORT
            else:
                sl_price = last_close - atr_15m * k_sl
                tp_price = last_close + atr_15m * k_tp

            # Update decision with correct SL/TP
            decision["sl_price"] = sl_price
            decision["tp_price"] = tp_price

            trade_group_id = build_trade_group_id(symbol, latest["ts"], decision["decision"])

            signal_row = {
                "symbol": symbol,
                "ts": latest["ts"],
                "model_id": self.model_id,
                "ev_long": decision["ev_long"],
                "ev_short": decision["ev_short"],
                "er_long": preds.get("er_long"),
                "er_short": preds.get("er_short"),
                "q05_long": preds.get("q05_long"),
                "q05_short": preds.get("q05_short"),
                "e_mae_long": preds.get("e_mae_long"),
                "e_mae_short": preds.get("e_mae_short"),
                "e_hold_long_min": int(preds.get("e_hold_long", 0)),
                "e_hold_short_min": int(preds.get("e_hold_short", 0)),
                "decision": decision["decision"],
                "size_notional": decision.get("size_notional"),
                "leverage": decision.get("leverage"),
                "sl_price": decision.get("sl_price"),
                "tp_price": decision.get("tp_price"),
                "block_reason_codes": json.dumps(decision.get("block_reasons", [])),
                "explain": json.dumps({}),
                "trade_group_id": trade_group_id,
            }

            # Store signal
            bulk_upsert(
                "signals",
                list(signal_row.keys()),
                [tuple(signal_row.values())],
                conflict_cols=["symbol", "ts"],
                update_cols=list(signal_row.keys()),
            )

            # Publish to redis
            self.bus.publish_json(
                "signal_updates",
                {
                    "symbol": symbol,
                    "ts": str(latest["ts"]),
                    "decision": decision["decision"],
                    "ev_long": decision["ev_long"],
                    "ev_short": decision["ev_short"],
                    "block_reasons": decision.get("block_reasons", []),
                },
            )

            logger.info(
                f"[SIGNAL] {symbol} -> {decision['decision']} | "
                f"er_long={preds.get('er_long', 0):.4f} er_short={preds.get('er_short', 0):.4f} | "
                f"SL={sl_price:.2f} TP={tp_price:.2f}"
            )

            # Execute shadow trade if conditions met
            await self._execute_shadow_trade(
                symbol=symbol,
                decision=decision,
                signal_row=signal_row,
                last_close=last_close,
            )

        except Exception as e:
            logger.error(f"Inference failed for {symbol}: {e}", exc_info=True)

    async def _execute_shadow_trade(
        self,
        symbol: str,
        decision: Dict[str, Any],
        signal_row: Dict[str, Any],
        last_close: float,
    ) -> None:
        """
        Execute shadow trade if conditions are met.

        This registers the position with position_manager so SL/TP can be
        monitored via markPrice stream. When SL/TP is hit, the position
        is automatically closed and recorded.
        """
        # Only execute if trading session is active in shadow mode
        if not session_manager.is_trading:
            return

        if session_manager.mode != "shadow":
            return

        # Only execute LONG or SHORT decisions
        trade_decision = decision.get("decision", "HOLD")
        if trade_decision not in ("LONG", "SHORT"):
            logger.debug(f"[SHADOW] Skipping {symbol}: decision={trade_decision}")
            return

        logger.info(f"[SHADOW] Processing {symbol} {trade_decision} signal")

        # Check for block reasons
        block_reasons = decision.get("block_reasons", [])
        if block_reasons:
            logger.info(
                f"[SHADOW] Trade blocked for {symbol}: {block_reasons}"
            )
            return

        # Use lock to prevent race condition when multiple symbols close at the same time
        async with self._position_entry_lock:
            # Check if already have a position for this symbol (in-memory check first for speed)
            existing = position_manager.get_position(symbol)
            if existing:
                logger.info(
                    f"[SHADOW] Already have position for {symbol} (memory), skipping"
                )
                return

            # Double-check against DB to prevent duplicates after restarts
            if position_manager.has_open_position_in_db(symbol):
                logger.info(
                    f"[SHADOW] Already have position for {symbol} (DB), skipping"
                )
                return

            # Check max positions limit using DB count for accuracy
            current_positions = position_manager.count_open_positions_in_db()
            if current_positions >= self.settings.max_positions:
                logger.info(
                    f"[SHADOW] Max positions reached ({current_positions}/{self.settings.max_positions}), skipping {symbol}"
                )
                return

            # Determine fill price
            fill_price = last_close
            if self.book_ticker_handler:
                book = self.book_ticker_handler.get_latest_book(symbol)
                if book:
                    # Use ask for LONG (buying), bid for SHORT (selling)
                    if trade_decision == "LONG":
                        fill_price = book.get("ask", last_close)
                    else:
                        fill_price = book.get("bid", last_close)

            # Calculate position size (use notional from decision or default)
            size_notional = decision.get("size_notional", 100.0)  # Default $100
            qty = size_notional / fill_price if fill_price > 0 else 0

            # Get SL/TP from signal
            sl_price = signal_row.get("sl_price")
            tp_price = signal_row.get("tp_price")
            trade_group_id = signal_row.get("trade_group_id")

            # Register position with position_manager
            # This enables SL/TP monitoring via markPrice stream
            position_manager.update_position_from_cache(
                symbol=symbol,
                side=trade_decision,
                amt=qty if trade_decision == "LONG" else -qty,
                entry_price=fill_price,
                trade_group_id=str(trade_group_id) if trade_group_id else None,
                sl_price=sl_price,
                tp_price=tp_price,
            )

            sl_str = f"{sl_price:.4f}" if sl_price else "N/A"
            tp_str = f"{tp_price:.4f}" if tp_price else "N/A"
            logger.info(
                f"[SHADOW] Entered {trade_decision} {symbol} @ {fill_price:.4f} | "
                f"qty={qty:.6f} notional=${size_notional:.2f} | "
                f"SL={sl_str} TP={tp_str}"
            )

            # Publish position opened event for real-time UI updates
            self.bus.publish_json(
                "trading_events",
                {
                    "type": "position_opened",
                    "symbol": symbol,
                    "side": trade_decision,
                    "qty": qty,
                    "entry_price": fill_price,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "notional": size_notional,
                    "is_shadow": True,
                },
            )

    async def rest_poller(self) -> None:
        """Poll REST APIs for data not available via websocket."""
        logger.info("Starting REST poller...")
        last_ls_poll = 0  # Track last L/S ratio poll time

        while True:
            try:
                universe = _active_universe(self.settings.universe_list())
                now_ts = datetime.now(timezone.utc)
                current_time = now_ts.timestamp()

                # L/S ratios polling: every 5 minutes (300 seconds)
                should_poll_ls = (current_time - last_ls_poll) >= 300

                for symbol in universe:
                    # Open Interest: every 1 minute
                    try:
                        oi_data = fetch_open_interest(symbol)
                        if oi_data:
                            oi_value = float(oi_data.get("openInterest", 0.0))
                            # Get mark price from cached markPrice handler
                            mark_price = 1.0
                            if self.mark_price_handler:
                                price_data = self.mark_price_handler.get_latest_price(symbol)
                                if price_data:
                                    mark_price = price_data.get("mark_price", 1.0)

                            self.buffer.add_open_interest(
                                (
                                    symbol,
                                    now_ts,
                                    oi_value,
                                    oi_value * mark_price,
                                )
                            )
                    except Exception as e:
                        logger.debug(f"Failed to fetch OI for {symbol}: {e}")

                    # Long/Short ratios: every 5 minutes
                    if should_poll_ls:
                        try:
                            ls_data = fetch_long_short_ratio(symbol, period="5m", limit=1)
                            top_ls_data = fetch_top_long_short_ratio(symbol, period="5m", limit=1)
                            taker_data = fetch_taker_buy_sell_ratio(symbol, period="5m", limit=1)

                            ls = ls_data[0] if ls_data else {}
                            top_ls = top_ls_data[0] if top_ls_data else {}
                            taker = taker_data[0] if taker_data else {}

                            self.buffer.add_long_short_ratio(
                                (
                                    symbol,
                                    now_ts,
                                    float(ls.get("longShortRatio", 1.0)),
                                    float(ls.get("longAccount", 0.5)),
                                    float(ls.get("shortAccount", 0.5)),
                                    float(top_ls.get("longShortRatio", 1.0)),
                                    float(top_ls.get("longAccount", 0.5)),
                                    float(top_ls.get("shortAccount", 0.5)),
                                    float(taker.get("buySellRatio", 1.0)),
                                    float(taker.get("buyVol", 0.0)),
                                    float(taker.get("sellVol", 0.0)),
                                )
                            )
                        except Exception as e:
                            logger.debug(f"Failed to fetch L/S for {symbol}: {e}")

                if should_poll_ls:
                    last_ls_poll = current_time

                set_collector_ok(True)

            except Exception as e:
                logger.error(f"REST poller error: {e}")
                set_collector_ok(False)

            # Poll every 60 seconds (OI every minute, L/S every 5 minutes)
            await asyncio.sleep(60)

    async def buffer_flusher(self) -> None:
        """Periodically flush the buffer."""
        logger.info("Starting buffer flusher...")

        while True:
            try:
                if self.buffer.should_flush():
                    self.buffer.flush()
            except Exception as e:
                logger.error(f"Buffer flush error: {e}")

            await asyncio.sleep(5)

    async def premium_aggregator(self) -> None:
        """Aggregate markPrice stream data to premium_index table every minute."""
        logger.info("Starting premium aggregator...")

        while True:
            try:
                await asyncio.sleep(60)  # Aggregate every 1 minute

                if not self.mark_price_handler:
                    continue

                now_ts = datetime.now(timezone.utc)
                prices = self.mark_price_handler.get_all_prices()

                for symbol, price_data in prices.items():
                    # Only save if data is recent (within last 5 seconds)
                    updated_at = price_data.get("updated_at")
                    if updated_at and (now_ts - updated_at).total_seconds() < 5:
                        self.buffer.add_premium(
                            (
                                symbol,
                                now_ts,
                                price_data.get("mark_price", 0.0),
                                price_data.get("index_price", 0.0),
                                price_data.get("mark_price", 0.0),  # last_price (use mark as proxy)
                                price_data.get("funding_rate", 0.0),
                                price_data.get("next_funding_time", now_ts),
                            )
                        )

            except Exception as e:
                logger.error(f"Premium aggregator error: {e}")

    def setup_websocket(self) -> None:
        """Setup websocket connections."""
        universe = _active_universe(self.settings.universe_list())

        # Build stream list - 5 streams per symbol
        streams = []
        for symbol in universe:
            s = symbol.lower().replace("_", "")  # BTCUSDT -> btcusdt
            streams.extend([
                f"{s}@kline_1m",
                f"{s}@kline_15m",
                f"{s}@kline_1h",
                f"{s}@markPrice@1s",
                f"{s}@bookTicker",
            ])

        logger.info(f"Setting up websocket with {len(streams)} streams for {len(universe)} symbols")
        logger.info(f"Streams per symbol: 5 (kline_1m, kline_15m, kline_1h, markPrice, bookTicker)")

        # Set stream counts
        set_ws_streams(len(streams), len(streams))

        # Setup handlers
        self.kline_handler = KlineHandler(
            self.buffer,
            on_candle_close=self.on_1m_candle_close,
            on_candle_close_15m=self.on_15m_candle_close,
            on_candle_close_1h=self.on_1h_candle_close,
        )
        self.book_ticker_handler = BookTickerHandler()
        self.mark_price_handler = MarkPriceHandler(
            position_manager=position_manager,
            book_ticker_handler=self.book_ticker_handler,
            session_manager=session_manager,
        )

        # Load positions from database on startup
        position_manager.load_positions_from_db()

        # Register handlers
        self.router.register("kline_1m", self.kline_handler.handle_kline_1m)
        self.router.register("kline_15m", self.kline_handler.handle_kline_15m)
        self.router.register("kline_1h", self.kline_handler.handle_kline_1h)
        self.router.register("markPrice@1s", self.mark_price_handler.handle)
        self.router.register("bookTicker", self.book_ticker_handler.handle)

        # Setup websocket manager
        ws_url = self.settings.binance_futures_ws_url
        self.ws_manager = WebsocketManager(ws_url, self.router.route)
        self.ws_manager.add_streams(streams)

    async def start(self) -> None:
        """Start the worker."""
        logger.info("Starting websocket-based realtime worker...")

        # Get universe for data preparation
        universe = _active_universe(self.settings.universe_list())

        # Ensure data is ready before starting (backfill if needed)
        logger.info(f"Preparing data for {len(universe)} symbols...")
        ensure_data_ready(universe)

        # Setup websocket
        self.setup_websocket()

        # Start background tasks
        self.rest_poller_task = asyncio.create_task(self.rest_poller())
        self.buffer_flusher_task = asyncio.create_task(self.buffer_flusher())
        self.premium_aggregator_task = asyncio.create_task(self.premium_aggregator())

        # Start websocket manager (blocks until stopped)
        if self.ws_manager:
            await self.ws_manager.start()


def main() -> None:
    """Main entry point."""
    worker = RealtimeWorker()
    asyncio.run(worker.start())


if __name__ == "__main__":
    main()
