"""Handler for kline (candlestick) websocket messages."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from packages.common.symbol_map import to_internal
from services.collector.buffer import IngestBuffer

logger = logging.getLogger(__name__)


class KlineHandler:
    """Processes kline websocket messages."""

    def __init__(
        self,
        buffer: IngestBuffer,
        on_candle_close: Callable[[str, datetime], None] | None = None,
        on_candle_close_15m: Callable[[str, datetime], None] | None = None,
        on_candle_close_1h: Callable[[str, datetime], None] | None = None,
    ):
        """
        Initialize kline handler.

        Args:
            buffer: Ingest buffer for storing candles
            on_candle_close: Callback for 1m candle completion (symbol, close_time)
            on_candle_close_15m: Callback for 15m candle completion
            on_candle_close_1h: Callback for 1h candle completion
        """
        self.buffer = buffer
        self.on_candle_close = on_candle_close
        self.on_candle_close_15m = on_candle_close_15m
        self.on_candle_close_1h = on_candle_close_1h
        self.last_1m_close: dict[str, datetime] = {}
        self.last_15m_close: dict[str, datetime] = {}
        self.last_1h_close: dict[str, datetime] = {}

    def handle_kline_1m(self, data: dict) -> None:
        """
        Handle 1m kline message.

        Message format:
        {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1699200000000,  # Open time
                "T": 1699200059999,  # Close time
                "i": "1m",
                "o": "36000.00",
                "h": "36100.00",
                "l": "35900.00",
                "c": "36050.00",
                "v": "100.5",
                "x": true  # Is candle closed?
            }
        }
        """
        k = data.get("k", {})
        symbol = to_internal(data.get("s", ""), "ws")
        is_closed = k.get("x", False)

        # Only process closed candles
        if not is_closed:
            return

        close_time_ms = k.get("T", 0)
        close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)

        # Deduplicate: ignore if we've already processed this candle
        if self.last_1m_close.get(symbol) == close_time:
            return

        self.last_1m_close[symbol] = close_time

        # Store candle in buffer
        try:
            self.buffer.add_candle(
                (
                    symbol,
                    close_time,
                    float(k.get("o", 0)),
                    float(k.get("h", 0)),
                    float(k.get("l", 0)),
                    float(k.get("c", 0)),
                    float(k.get("v", 0)),
                )
            )
            logger.debug(f"1m candle closed: {symbol} @ {close_time}")

            # Trigger inference callback
            if self.on_candle_close:
                self.on_candle_close(symbol, close_time)

        except Exception as e:
            logger.error(f"Failed to process 1m kline for {symbol}: {e}")

    def handle_kline_15m(self, data: dict) -> None:
        """
        Handle 15m kline message.

        Stores candle and triggers feature computation.
        """
        k = data.get("k", {})
        symbol = to_internal(data.get("s", ""), "ws")
        is_closed = k.get("x", False)

        if not is_closed:
            return

        close_time_ms = k.get("T", 0)
        close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)

        # Deduplicate
        if self.last_15m_close.get(symbol) == close_time:
            return

        self.last_15m_close[symbol] = close_time

        try:
            self.buffer.add_candle_15m(
                (
                    symbol,
                    close_time,
                    float(k.get("o", 0)),
                    float(k.get("h", 0)),
                    float(k.get("l", 0)),
                    float(k.get("c", 0)),
                    float(k.get("v", 0)),
                )
            )
            logger.info(f"15m candle closed: {symbol} @ {close_time}")

            # Trigger feature computation callback
            if self.on_candle_close_15m:
                self.on_candle_close_15m(symbol, close_time)

        except Exception as e:
            logger.error(f"Failed to process 15m kline for {symbol}: {e}")

    def handle_kline_1h(self, data: dict) -> None:
        """
        Handle 1h kline message.

        Stores candle and triggers feature computation.
        """
        k = data.get("k", {})
        symbol = to_internal(data.get("s", ""), "ws")
        is_closed = k.get("x", False)

        if not is_closed:
            return

        close_time_ms = k.get("T", 0)
        close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)

        # Deduplicate
        if self.last_1h_close.get(symbol) == close_time:
            return

        self.last_1h_close[symbol] = close_time

        try:
            self.buffer.add_candle_1h(
                (
                    symbol,
                    close_time,
                    float(k.get("o", 0)),
                    float(k.get("h", 0)),
                    float(k.get("l", 0)),
                    float(k.get("c", 0)),
                    float(k.get("v", 0)),
                )
            )
            logger.info(f"1h candle closed: {symbol} @ {close_time}")

            # Trigger feature computation callback
            if self.on_candle_close_1h:
                self.on_candle_close_1h(symbol, close_time)

        except Exception as e:
            logger.error(f"Failed to process 1h kline for {symbol}: {e}")
