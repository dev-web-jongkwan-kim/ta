"""Handler for markPrice websocket messages with SL/TP monitoring."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional, TYPE_CHECKING

from packages.common.symbol_map import to_internal

if TYPE_CHECKING:
    from services.websocket.handlers.book_ticker_handler import BookTickerHandler
    from services.engine.position_manager import PositionManager
    from services.engine.session_manager import TradingSessionManager

logger = logging.getLogger(__name__)


class MarkPriceHandler:
    """Processes markPrice websocket messages for SL/TP monitoring."""

    def __init__(
        self,
        position_manager: Optional["PositionManager"] = None,
        book_ticker_handler: Optional["BookTickerHandler"] = None,
        session_manager: Optional["TradingSessionManager"] = None,
    ):
        """
        Initialize mark price handler.

        Args:
            position_manager: Position manager for SL/TP checking
            book_ticker_handler: Book ticker handler for fill price estimation
            session_manager: Session manager for trade recording
        """
        self.latest_prices: dict[str, dict] = {}
        self._position_manager = position_manager
        self._book_ticker_handler = book_ticker_handler
        self._session_manager = session_manager

        # Callback for SL/TP hits
        self._on_sl_tp_hit_callbacks: list[Callable] = []

    def set_position_manager(self, pm: "PositionManager") -> None:
        """Set position manager (for late binding)."""
        self._position_manager = pm

    def set_book_ticker_handler(self, bth: "BookTickerHandler") -> None:
        """Set book ticker handler (for late binding)."""
        self._book_ticker_handler = bth

    def set_session_manager(self, sm: "TradingSessionManager") -> None:
        """Set session manager (for late binding)."""
        self._session_manager = sm

    def on_sl_tp_hit(self, callback: Callable) -> None:
        """Register callback for SL/TP hit events."""
        self._on_sl_tp_hit_callbacks.append(callback)

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

            # Check SL/TP for open positions
            self._check_sl_tp(symbol, mark_price)

        except Exception as e:
            logger.error(f"Failed to process markPrice: {e}")

    def _check_sl_tp(self, symbol: str, mark_price: float) -> None:
        """
        Check SL/TP levels for open positions.

        If SL or TP is hit, triggers position close.
        """
        if not self._position_manager:
            return

        # Check if SL/TP hit
        result = self._position_manager.check_sl_tp(symbol, mark_price)
        if not result:
            return

        position, exit_reason = result

        # Determine fill price
        fill_price = mark_price
        if self._book_ticker_handler:
            book = self._book_ticker_handler.get_latest_book(symbol)
            if book:
                # Use bid for closing LONG (selling), ask for closing SHORT (buying)
                if position.side == "LONG":
                    fill_price = book.get("bid", mark_price)
                else:
                    fill_price = book.get("ask", mark_price)

        # Determine if shadow mode
        is_shadow = True
        if self._session_manager:
            is_shadow = self._session_manager.mode == "shadow"

        # Close position
        trade_record = self._position_manager.close_position(
            symbol=symbol,
            exit_price=fill_price,
            exit_reason=exit_reason,
            is_shadow=is_shadow,
        )

        if not trade_record:
            return

        # Record trade in session
        if self._session_manager and self._session_manager.is_trading:
            try:
                from uuid import uuid4
                self._session_manager.record_trade(
                    trade_group_id=trade_record.get("trade_group_id") or uuid4(),
                    symbol=trade_record["symbol"],
                    side=trade_record["side"],
                    entry_time=trade_record["entry_time"],
                    entry_price=trade_record["entry_price"],
                    qty=trade_record["qty"],
                    exit_time=trade_record["exit_time"],
                    exit_price=trade_record["exit_price"],
                    pnl=trade_record["pnl"],
                    exit_reason=trade_record["exit_reason"],
                )
            except Exception as e:
                logger.error(f"Failed to record trade: {e}")

        # Notify callbacks
        for callback in self._on_sl_tp_hit_callbacks:
            try:
                callback({
                    "type": "sl_tp_hit",
                    "symbol": symbol,
                    "exit_reason": exit_reason,
                    "mark_price": mark_price,
                    "fill_price": fill_price,
                    "pnl": trade_record["pnl"],
                    "is_shadow": is_shadow,
                })
            except Exception as e:
                logger.error(f"SL/TP hit callback error: {e}")

        logger.info(
            f"{'[SHADOW] ' if is_shadow else ''}Position closed by {exit_reason}: "
            f"{symbol} PnL={trade_record['pnl']:.2f}"
        )

    def get_latest_price(self, symbol: str) -> dict | None:
        """Get latest cached price for a symbol."""
        return self.latest_prices.get(symbol)

    def get_all_prices(self) -> dict[str, dict]:
        """Get all cached prices."""
        return self.latest_prices.copy()
