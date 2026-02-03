"""Handler for bookTicker websocket messages."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from packages.common.symbol_map import to_internal

logger = logging.getLogger(__name__)


class BookTickerHandler:
    """Processes bookTicker websocket messages for best bid/ask."""

    def __init__(self):
        """Initialize book ticker handler."""
        self.latest_book: dict[str, dict] = {}

    def handle(self, data: dict) -> None:
        """
        Handle bookTicker message.

        Message format:
        {
            "u": 400900217,     # order book updateId
            "s": "BTCUSDT",     # symbol
            "b": "36049.00",    # best bid price
            "B": "10.5",        # best bid qty
            "a": "36050.00",    # best ask price
            "A": "8.2",         # best ask qty
            "T": 1699200000000  # transaction time
        }
        """
        try:
            symbol = to_internal(data.get("s", ""), "ws")
            best_bid = float(data.get("b", 0))
            best_bid_qty = float(data.get("B", 0))
            best_ask = float(data.get("a", 0))
            best_ask_qty = float(data.get("A", 0))

            # Cache latest book for slippage calculation
            self.latest_book[symbol] = {
                "bid": best_bid,
                "bid_qty": best_bid_qty,
                "ask": best_ask,
                "ask_qty": best_ask_qty,
                "spread": best_ask - best_bid,
                "spread_bps": ((best_ask - best_bid) / best_bid * 10000) if best_bid > 0 else 0,
                "updated_at": datetime.now(timezone.utc),
            }

        except Exception as e:
            logger.error(f"Failed to process bookTicker: {e}")

    def get_latest_book(self, symbol: str) -> dict | None:
        """Get latest cached book for a symbol."""
        return self.latest_book.get(symbol)

    def estimate_fill_price(self, symbol: str, side: str, qty: float) -> float | None:
        """
        Estimate fill price for market order.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            qty: Order quantity

        Returns:
            Estimated fill price or None if no data
        """
        book = self.latest_book.get(symbol)
        if not book:
            return None

        if side == "BUY":
            # Buy at ask price
            return book["ask"]
        else:
            # Sell at bid price
            return book["bid"]

    def get_all_books(self) -> dict[str, dict]:
        """Get all cached order books."""
        return self.latest_book.copy()
