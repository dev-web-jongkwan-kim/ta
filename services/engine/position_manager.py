"""Position Manager for SL/TP monitoring and position lifecycle management."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Callable
from uuid import UUID

from packages.common.db import fetch_all, get_conn

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Open position with SL/TP levels."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    amt: float
    entry_price: float
    entry_time: datetime
    trade_group_id: Optional[UUID]
    sl_price: Optional[float]
    tp_price: Optional[float]

    def check_sl_hit(self, mark_price: float) -> bool:
        """Check if SL is hit."""
        if self.sl_price is None:
            return False
        if self.side == "LONG":
            return mark_price <= self.sl_price
        else:  # SHORT
            return mark_price >= self.sl_price

    def check_tp_hit(self, mark_price: float) -> bool:
        """Check if TP is hit."""
        if self.tp_price is None:
            return False
        if self.side == "LONG":
            return mark_price >= self.tp_price
        else:  # SHORT
            return mark_price <= self.tp_price

    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate PnL based on exit price."""
        if self.side == "LONG":
            return (exit_price - self.entry_price) * self.amt
        else:  # SHORT
            return (self.entry_price - exit_price) * self.amt


class PositionManager:
    """
    Manages position lifecycle including SL/TP monitoring.

    This class integrates with:
    - UserDataHandler for real-time position updates
    - Database for SL/TP levels from signals table
    - SessionManager for trade recording
    """

    def __init__(self):
        """Initialize position manager."""
        # Cached positions keyed by symbol
        self._positions: Dict[str, OpenPosition] = {}

        # Track last SL/TP hit time to prevent duplicate triggers
        self._last_hit_time: Dict[str, datetime] = {}

        # Callback for position close
        self._on_close_callbacks: List[Callable] = []

        # Minimum time between hits for same position (seconds)
        self._hit_cooldown_sec = 5

    def on_position_close(self, callback: Callable) -> None:
        """Register callback for position close events."""
        self._on_close_callbacks.append(callback)

    def update_position_from_cache(
        self,
        symbol: str,
        side: str,
        amt: float,
        entry_price: float,
        trade_group_id: Optional[str] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> None:
        """
        Update cached position from UserDataHandler.

        Called when userData stream receives position update.
        """
        if amt == 0 or abs(amt) < 1e-8:
            # Position closed
            self._positions.pop(symbol, None)
            return

        # Use provided SL/TP or fetch from signals table
        if sl_price is None or tp_price is None:
            fetched_sl, fetched_tp = self._fetch_sl_tp(symbol)
            sl_price = sl_price or fetched_sl
            tp_price = tp_price or fetched_tp

        position = OpenPosition(
            symbol=symbol,
            side="LONG" if amt > 0 else "SHORT",
            amt=abs(amt),
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            trade_group_id=UUID(trade_group_id) if trade_group_id else None,
            sl_price=sl_price,
            tp_price=tp_price,
        )
        self._positions[symbol] = position

        # Persist to database for cross-container visibility
        self._persist_position(position)

        logger.debug(f"Position updated: {symbol} {side} amt={amt} SL={sl_price} TP={tp_price}")

    def _persist_position(self, position: OpenPosition, event_type: str = "entry") -> None:
        """Persist position to database."""
        try:
            from uuid import uuid4
            # Use trade_group_id from position or generate a new one
            trade_group_id = position.trade_group_id or uuid4()

            with get_conn() as conn:
                with conn.cursor() as cur:
                    # Use the actual primary key for conflict resolution
                    cur.execute("""
                        INSERT INTO positions (symbol, ts, side, amt, entry_price, trade_group_id, event_type, sl_price, tp_price)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (trade_group_id, symbol, ts, side) DO UPDATE SET
                            amt = EXCLUDED.amt,
                            entry_price = EXCLUDED.entry_price,
                            event_type = EXCLUDED.event_type,
                            sl_price = EXCLUDED.sl_price,
                            tp_price = EXCLUDED.tp_price
                    """, (
                        position.symbol,
                        position.entry_time,
                        position.side,
                        position.amt if position.side == "LONG" else -position.amt,
                        position.entry_price,
                        str(trade_group_id),
                        event_type,
                        position.sl_price,
                        position.tp_price,
                    ))
                conn.commit()
                logger.debug(f"Persisted position: {position.symbol} {position.side} event_type={event_type}")
        except Exception as e:
            logger.error(f"Failed to persist position: {e}")

    def check_sl_tp(
        self,
        symbol: str,
        mark_price: float,
    ) -> Optional[Tuple[OpenPosition, str]]:
        """
        Check if SL/TP is hit for a symbol.

        Returns:
            Tuple of (position, exit_reason) if hit, None otherwise.
            exit_reason is either "SL" or "TP"
        """
        position = self._positions.get(symbol)
        if not position:
            return None

        # Check cooldown to prevent duplicate triggers
        last_hit = self._last_hit_time.get(symbol)
        if last_hit:
            elapsed = (datetime.now(timezone.utc) - last_hit).total_seconds()
            if elapsed < self._hit_cooldown_sec:
                return None

        # Check SL first (higher priority)
        if position.check_sl_hit(mark_price):
            self._last_hit_time[symbol] = datetime.now(timezone.utc)
            logger.info(
                f"SL HIT: {symbol} {position.side} "
                f"mark={mark_price} sl={position.sl_price}"
            )
            return (position, "SL")

        # Check TP
        if position.check_tp_hit(mark_price):
            self._last_hit_time[symbol] = datetime.now(timezone.utc)
            logger.info(
                f"TP HIT: {symbol} {position.side} "
                f"mark={mark_price} tp={position.tp_price}"
            )
            return (position, "TP")

        return None

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        is_shadow: bool = True,
    ) -> Optional[Dict]:
        """
        Close a position and record the trade.

        Args:
            symbol: Trading symbol
            exit_price: Price at which position is closed
            exit_reason: "SL", "TP", "MANUAL", or "TIMEOUT"
            is_shadow: Whether this is shadow mode

        Returns:
            Trade record dict if successful, None otherwise
        """
        position = self._positions.get(symbol)
        if not position:
            logger.warning(f"No open position to close for {symbol}")
            return None

        # Calculate PnL
        pnl = position.calculate_pnl(exit_price)
        exit_time = datetime.now(timezone.utc)
        hold_min = int((exit_time - position.entry_time).total_seconds() / 60)

        # Calculate return percentage
        notional = position.entry_price * position.amt
        return_pct = (pnl / notional * 100) if notional > 0 else 0

        trade_record = {
            "symbol": symbol,
            "side": position.side,
            "entry_time": position.entry_time,
            "exit_time": exit_time,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "qty": position.amt,
            "pnl": pnl,
            "return_pct": return_pct,
            "hold_min": hold_min,
            "exit_reason": exit_reason,
            "is_shadow": is_shadow,
            "trade_group_id": position.trade_group_id,
        }

        # Persist final position state to database
        # For FINAL record, store exit_price in entry_price column
        final_position = OpenPosition(
            symbol=position.symbol,
            side=position.side,
            amt=position.amt,
            entry_price=exit_price,  # Store actual exit price for FINAL record
            entry_time=exit_time,  # Use exit time for FINAL record timestamp
            trade_group_id=position.trade_group_id,
            sl_price=position.sl_price,
            tp_price=position.tp_price,
        )
        self._persist_position(final_position, event_type="FINAL")

        # Remove from cache
        self._positions.pop(symbol, None)
        self._last_hit_time.pop(symbol, None)

        # Notify callbacks
        for callback in self._on_close_callbacks:
            try:
                callback(trade_record)
            except Exception as e:
                logger.error(f"Position close callback error: {e}")

        logger.info(
            f"Position closed: {symbol} {position.side} "
            f"PnL={pnl:.2f} reason={exit_reason}"
        )

        return trade_record

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        """Get cached position for a symbol."""
        return self._positions.get(symbol)

    def has_open_position_in_db(self, symbol: str) -> bool:
        """
        Check if symbol has an open position in the database.

        This is a more reliable check than in-memory state, used to prevent
        duplicate entries when the in-memory state might be stale.
        """
        try:
            rows = fetch_all("""
                SELECT 1
                FROM positions p
                WHERE p.symbol = %s
                AND p.event_type = 'entry'
                AND NOT EXISTS (
                    SELECT 1 FROM positions p2
                    WHERE p2.trade_group_id = p.trade_group_id
                    AND p2.event_type = 'FINAL'
                )
                LIMIT 1
            """, (symbol,))
            return len(rows) > 0
        except Exception as e:
            logger.error(f"Failed to check open position in DB for {symbol}: {e}")
            return False

    def count_open_positions_in_db(self) -> int:
        """
        Count total open positions in the database.

        This is a more reliable count than in-memory state.
        """
        try:
            rows = fetch_all("""
                SELECT COUNT(DISTINCT symbol)
                FROM positions p
                WHERE p.event_type = 'entry'
                AND NOT EXISTS (
                    SELECT 1 FROM positions p2
                    WHERE p2.trade_group_id = p.trade_group_id
                    AND p2.event_type = 'FINAL'
                )
            """)
            return rows[0][0] if rows else 0
        except Exception as e:
            logger.error(f"Failed to count open positions in DB: {e}")
            return len(self._positions)

    def count_directional_positions_in_db(self, side: str) -> int:
        """
        Count open positions by direction (LONG/SHORT) in the database.

        Args:
            side: "LONG" or "SHORT"

        Returns:
            Number of open positions in the specified direction.
        """
        try:
            rows = fetch_all("""
                SELECT COUNT(DISTINCT symbol)
                FROM positions p
                WHERE p.event_type = 'entry'
                AND p.side = %s
                AND NOT EXISTS (
                    SELECT 1 FROM positions p2
                    WHERE p2.trade_group_id = p.trade_group_id
                    AND p2.event_type = 'FINAL'
                )
            """, (side,))
            return rows[0][0] if rows else 0
        except Exception as e:
            logger.error(f"Failed to count {side} positions in DB: {e}")
            return 0

    def get_all_positions(self) -> Dict[str, OpenPosition]:
        """Get all open positions from database.

        Returns entries without corresponding FINAL records (actual open positions).
        If a symbol has multiple open entries, returns the latest one.
        """
        positions: Dict[str, OpenPosition] = {}
        try:
            # Get open positions: entries without corresponding FINAL records
            rows = fetch_all("""
                WITH open_entries AS (
                    SELECT
                        p.symbol, p.side, p.amt, p.entry_price, p.ts, p.trade_group_id, p.sl_price, p.tp_price,
                        ROW_NUMBER() OVER (PARTITION BY p.symbol ORDER BY p.ts DESC) as rn
                    FROM positions p
                    WHERE p.event_type = 'entry'
                    AND p.amt != 0
                    AND NOT EXISTS (
                        SELECT 1 FROM positions p2
                        WHERE p2.trade_group_id = p.trade_group_id
                        AND p2.event_type = 'FINAL'
                    )
                )
                SELECT symbol, side, amt, entry_price, ts, trade_group_id, sl_price, tp_price
                FROM open_entries
                WHERE rn = 1
            """)
            for row in rows:
                symbol, side, amt, entry_price, ts, trade_group_id, sl_price, tp_price = row
                if amt and abs(amt) > 1e-8:
                    positions[symbol] = OpenPosition(
                        symbol=symbol,
                        side=side,
                        amt=abs(amt),
                        entry_price=float(entry_price) if entry_price else 0,
                        entry_time=ts,
                        trade_group_id=trade_group_id,
                        sl_price=float(sl_price) if sl_price else None,
                        tp_price=float(tp_price) if tp_price else None,
                    )
        except Exception as e:
            logger.error(f"Failed to get positions from DB: {e}")
        return positions

    def _cleanup_orphaned_entries(self) -> None:
        """
        Clean up orphaned entries on startup.

        When the worker restarts, older entries may exist without FINAL records.
        This marks older duplicates (per symbol) as FINAL to keep the DB clean.
        """
        try:
            from datetime import datetime, timezone

            # Find symbols with multiple open entries (entry without FINAL)
            duplicates = fetch_all("""
                WITH open_entries AS (
                    SELECT
                        p.symbol, p.ts, p.side, p.trade_group_id, p.entry_price, p.amt,
                        p.sl_price, p.tp_price,
                        ROW_NUMBER() OVER (PARTITION BY p.symbol ORDER BY p.ts DESC) as rn
                    FROM positions p
                    WHERE p.event_type = 'entry'
                    AND NOT EXISTS (
                        SELECT 1 FROM positions p2
                        WHERE p2.trade_group_id = p.trade_group_id
                        AND p2.event_type = 'FINAL'
                    )
                )
                SELECT symbol, ts, side, trade_group_id, entry_price, amt, sl_price, tp_price
                FROM open_entries
                WHERE rn > 1
            """)

            if not duplicates:
                return

            logger.info(f"Cleaning up {len(duplicates)} orphaned position entries...")

            with get_conn() as conn:
                with conn.cursor() as cur:
                    for row in duplicates:
                        symbol, ts, side, trade_group_id, entry_price, amt, sl_price, tp_price = row
                        exit_time = datetime.now(timezone.utc)

                        # Insert FINAL record to close this orphaned position
                        cur.execute("""
                            INSERT INTO positions (symbol, ts, side, amt, entry_price, trade_group_id, event_type, sl_price, tp_price)
                            VALUES (%s, %s, %s, %s, %s, %s, 'FINAL', %s, %s)
                            ON CONFLICT (trade_group_id, symbol, ts, side) DO NOTHING
                        """, (symbol, exit_time, side, amt, entry_price, str(trade_group_id), sl_price, tp_price))

                        logger.info(f"Closed orphaned entry: {symbol} @ {ts}")

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned entries: {e}")

    def load_positions_from_db(self) -> None:
        """Load open positions from database on startup.

        Loads entries without corresponding FINAL records (actual open positions).
        Also cleans up orphaned duplicate entries first.
        """
        # First, clean up any orphaned duplicate entries from previous runs
        self._cleanup_orphaned_entries()

        try:
            # Get open positions: entries without corresponding FINAL records
            # Use the LATEST entry per symbol (in case of duplicates)
            rows = fetch_all("""
                WITH open_entries AS (
                    SELECT
                        p.symbol, p.side, p.amt, p.entry_price, p.ts, p.trade_group_id, p.sl_price, p.tp_price,
                        ROW_NUMBER() OVER (PARTITION BY p.symbol ORDER BY p.ts DESC) as rn
                    FROM positions p
                    WHERE p.event_type = 'entry'
                    AND p.amt != 0
                    AND NOT EXISTS (
                        SELECT 1 FROM positions p2
                        WHERE p2.trade_group_id = p.trade_group_id
                        AND p2.event_type = 'FINAL'
                    )
                )
                SELECT symbol, side, amt, entry_price, ts, trade_group_id, sl_price, tp_price
                FROM open_entries
                WHERE rn = 1
            """)

            for row in rows:
                symbol, side, amt, entry_price, ts, trade_group_id, sl_price, tp_price = row
                if amt and abs(amt) > 1e-8:
                    # If SL/TP not in positions table, try fetching from signals
                    if sl_price is None or tp_price is None:
                        fetched_sl, fetched_tp = self._fetch_sl_tp(symbol)
                        sl_price = sl_price or fetched_sl
                        tp_price = tp_price or fetched_tp
                    self._positions[symbol] = OpenPosition(
                        symbol=symbol,
                        side=side,
                        amt=abs(amt),
                        entry_price=float(entry_price) if entry_price else 0,
                        entry_time=ts,
                        trade_group_id=trade_group_id,
                        sl_price=float(sl_price) if sl_price else None,
                        tp_price=float(tp_price) if tp_price else None,
                    )

            logger.info(f"Loaded {len(self._positions)} positions from database")

        except Exception as e:
            logger.error(f"Failed to load positions from DB: {e}")

    def refresh_sl_tp(self, symbol: str) -> None:
        """Refresh SL/TP levels from signals table for a position."""
        position = self._positions.get(symbol)
        if not position:
            return

        sl_price, tp_price = self._fetch_sl_tp(symbol)
        position.sl_price = sl_price
        position.tp_price = tp_price

        logger.debug(f"SL/TP refreshed for {symbol}: SL={sl_price} TP={tp_price}")

    def _fetch_sl_tp(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch SL/TP levels from signals table."""
        try:
            rows = fetch_all(
                """
                SELECT sl_price, tp_price
                FROM signals
                WHERE symbol = %s
                    AND sl_price IS NOT NULL
                    AND tp_price IS NOT NULL
                ORDER BY ts DESC
                LIMIT 1
                """,
                (symbol,)
            )

            if rows and rows[0]:
                sl_price = float(rows[0][0]) if rows[0][0] else None
                tp_price = float(rows[0][1]) if rows[0][1] else None
                return sl_price, tp_price

            return None, None

        except Exception as e:
            logger.error(f"Failed to fetch SL/TP for {symbol}: {e}")
            return None, None


# Singleton instance
position_manager = PositionManager()
