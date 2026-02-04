"""Websocket-based real-time trading worker."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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

    async def run_inference(self, symbol: str) -> None:
        """Run inference for a symbol."""
        try:
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

            # Run prediction
            preds = self.predictor.predict(latest["features"])

            # Make decision (simplified - no risk checks for now)
            policy_cfg = PolicyConfig(
                self.settings.ev_min,
                self.settings.q05_min,
                self.settings.mae_max,
                self.settings.max_positions,
            )
            last_close = float(candles.iloc[-1]["close"]) if not candles.empty else 0.0
            atr = float(latest["atr"]) if pd.notna(latest.get("atr")) else 0.0
            sl_price = last_close - atr * 1.0
            tp_price = last_close + atr * 1.5
            state = {"equity": 10000.0, "sl_price": sl_price, "tp_price": tp_price}
            decision = decide(symbol, preds, state, policy_cfg)

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

            logger.info(f"Inference complete: {symbol} -> {decision['decision']}")

        except Exception as e:
            logger.error(f"Inference failed for {symbol}: {e}", exc_info=True)

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
        self.kline_handler = KlineHandler(self.buffer, self.on_1m_candle_close)
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
