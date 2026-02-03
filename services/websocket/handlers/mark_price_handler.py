"""Handler for markPrice websocket messages."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from packages.common.symbol_map import to_internal

logger = logging.getLogger(__name__)


class MarkPriceHandler:
    """Processes markPrice websocket messages for SL/TP monitoring."""

    def __init__(self):
        """Initialize mark price handler."""
        self.latest_prices: dict[str, dict] = {}

    def handle(self, data: dict) -> None:
        """
        Handle markPrice@1s message.

        Message format:
        {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "p": "36050.00",      # Mark price
            "i": "36045.00",      # Index price
            "r": "0.00010000",    # Funding rate
            "T": 1699200000000    # Next funding time
        }
        """
        try:
            symbol = to_internal(data.get("s", ""), "ws")
            mark_price = float(data.get("p", 0))
            index_price = float(data.get("i", 0))
            funding_rate = float(data.get("r", 0))
            next_funding_time_ms = data.get("T", 0)

            # Cache latest price for SL/TP checks
            self.latest_prices[symbol] = {
                "mark_price": mark_price,
                "index_price": index_price,
                "funding_rate": funding_rate,
                "next_funding_time": datetime.fromtimestamp(
                    next_funding_time_ms / 1000, tz=timezone.utc
                ),
                "updated_at": datetime.now(timezone.utc),
            }

            # TODO: Check SL/TP for open positions
            # This would trigger position closure if price hits SL or TP

        except Exception as e:
            logger.error(f"Failed to process markPrice: {e}")

    def get_latest_price(self, symbol: str) -> dict | None:
        """Get latest cached price for a symbol."""
        return self.latest_prices.get(symbol)

    def get_all_prices(self) -> dict[str, dict]:
        """Get all cached prices."""
        return self.latest_prices.copy()
