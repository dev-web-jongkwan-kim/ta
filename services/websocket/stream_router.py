"""Routes websocket messages to appropriate handlers."""
from __future__ import annotations

import logging
from typing import Callable, Dict

from packages.common.ws_status import increment_ws_message

logger = logging.getLogger(__name__)


class StreamRouter:
    """Routes websocket stream messages to handlers."""

    def __init__(self):
        """Initialize stream router."""
        self.handlers: Dict[str, Callable[[dict], None]] = {}
        self.stats = {
            "total_messages": 0,
            "routed_messages": 0,
            "unhandled_messages": 0,
        }

    def register(self, stream_pattern: str, handler: Callable[[dict], None]) -> None:
        """
        Register a handler for a stream pattern.

        Args:
            stream_pattern: Stream pattern (e.g., 'kline_1m', 'markPrice')
            handler: Handler function
        """
        self.handlers[stream_pattern] = handler
        logger.info(f"Registered handler for pattern: {stream_pattern}")

    def route(self, message: dict) -> None:
        """
        Route a message to the appropriate handler.

        Expected message format:
        {
            "stream": "btcusdt@kline_1m",
            "data": {...}
        }
        """
        self.stats["total_messages"] += 1

        stream = message.get("stream", "")
        data = message.get("data", {})

        if not stream:
            self.stats["unhandled_messages"] += 1
            return

        # Extract stream type (e.g., "kline_1m" from "btcusdt@kline_1m")
        # Handle multi-part types like "markPrice@1s" -> "markPrice@1s"
        parts = stream.split("@")
        if len(parts) < 2:
            self.stats["unhandled_messages"] += 1
            return

        # Join all parts after the symbol
        stream_type = "@".join(parts[1:])  # "kline_1m", "markPrice@1s", etc.

        # Find matching handler
        handler = self.handlers.get(stream_type)
        if handler:
            try:
                handler(data)
                self.stats["routed_messages"] += 1
                increment_ws_message(stream_type)
            except Exception as e:
                logger.error(f"Handler error for {stream_type}: {e}")
        else:
            self.stats["unhandled_messages"] += 1
            if self.stats["unhandled_messages"] % 100 == 0:
                logger.debug(
                    f"No handler for stream type: {stream_type} "
                    f"(unhandled: {self.stats['unhandled_messages']})"
                )

    def get_stats(self) -> dict:
        """Get routing statistics."""
        return self.stats.copy()
