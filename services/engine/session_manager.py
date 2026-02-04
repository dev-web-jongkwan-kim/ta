"""Trading Session Manager for manual start/stop control."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Literal
from uuid import UUID, uuid4

from packages.common.db import get_conn, fetch_all

logger = logging.getLogger(__name__)

TradingMode = Literal["off", "shadow", "live"]


@dataclass
class TradingSession:
    """Trading session data."""
    session_id: UUID
    mode: str
    started_at: datetime
    stopped_at: Optional[datetime]
    initial_capital: Optional[Decimal]
    final_capital: Optional[Decimal]
    total_trades: int
    wins: int
    losses: int
    total_pnl: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    win_rate: Optional[float]
    profit_factor: Optional[float]
    avg_hold_min: Optional[int]
    best_trade: Optional[Decimal]
    worst_trade: Optional[Decimal]

    @property
    def is_active(self) -> bool:
        return self.stopped_at is None

    @property
    def running_time_sec(self) -> int:
        end = self.stopped_at or datetime.utcnow()
        return int((end - self.started_at).total_seconds())

    def to_dict(self) -> Dict:
        return {
            "session_id": str(self.session_id),
            "mode": self.mode,
            "started_at": self.started_at.isoformat(),
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "initial_capital": float(self.initial_capital) if self.initial_capital else None,
            "final_capital": float(self.final_capital) if self.final_capital else None,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": float(self.total_pnl),
            "gross_profit": float(self.gross_profit),
            "gross_loss": float(self.gross_loss),
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_hold_min": self.avg_hold_min,
            "best_trade": float(self.best_trade) if self.best_trade else None,
            "worst_trade": float(self.worst_trade) if self.worst_trade else None,
            "is_active": self.is_active,
            "running_time_sec": self.running_time_sec,
        }


@dataclass
class Trade:
    """Completed trade data."""
    trade_id: UUID
    session_id: Optional[UUID]
    trade_group_id: UUID
    symbol: str
    side: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: Decimal
    exit_price: Optional[Decimal]
    qty: Decimal
    pnl: Optional[Decimal]
    return_pct: Optional[float]
    hold_min: Optional[int]
    exit_reason: Optional[str]
    is_shadow: bool

    def to_dict(self) -> Dict:
        return {
            "trade_id": str(self.trade_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "trade_group_id": str(self.trade_group_id),
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "qty": float(self.qty),
            "pnl": float(self.pnl) if self.pnl else None,
            "return_pct": self.return_pct,
            "hold_min": self.hold_min,
            "exit_reason": self.exit_reason,
            "is_shadow": self.is_shadow,
        }


class TradingSessionManager:
    """Manages trading sessions for manual start/stop control."""

    _instance: Optional["TradingSessionManager"] = None
    _current_session: Optional[TradingSession] = None
    _mode: TradingMode = "off"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_trading(self) -> bool:
        return self._mode != "off" and self._current_session is not None

    @property
    def current_session(self) -> Optional[TradingSession]:
        return self._current_session

    def start(self, mode: Literal["shadow", "live"], initial_capital: Optional[float] = None) -> TradingSession:
        """Start a new trading session."""
        if self.is_trading:
            raise RuntimeError(f"Trading already active in {self._mode} mode. Stop first.")

        # Validate mode
        if mode not in ("shadow", "live"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'shadow' or 'live'")

        # Perform safety checks for live mode
        if mode == "live":
            self._validate_live_start()

        session_id = uuid4()
        now = datetime.utcnow()

        # Insert into database
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trading_sessions (session_id, mode, started_at, initial_capital)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (str(session_id), mode, now, initial_capital)
                )
            conn.commit()

        self._current_session = TradingSession(
            session_id=session_id,
            mode=mode,
            started_at=now,
            stopped_at=None,
            initial_capital=Decimal(str(initial_capital)) if initial_capital else None,
            final_capital=None,
            total_trades=0,
            wins=0,
            losses=0,
            total_pnl=Decimal("0"),
            gross_profit=Decimal("0"),
            gross_loss=Decimal("0"),
            win_rate=None,
            profit_factor=None,
            avg_hold_min=None,
            best_trade=None,
            worst_trade=None,
        )
        self._mode = mode

        logger.info(f"Trading session started: {session_id} in {mode} mode")
        return self._current_session

    def stop(self, final_capital: Optional[float] = None) -> TradingSession:
        """Stop the current trading session."""
        if not self.is_trading:
            raise RuntimeError("No active trading session to stop")

        session = self._current_session
        now = datetime.utcnow()

        # Calculate final stats from trades
        stats = self._calculate_session_stats(session.session_id)

        # Update database
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE trading_sessions SET
                        stopped_at = %s,
                        final_capital = %s,
                        total_trades = %s,
                        wins = %s,
                        losses = %s,
                        total_pnl = %s,
                        gross_profit = %s,
                        gross_loss = %s,
                        win_rate = %s,
                        profit_factor = %s,
                        avg_hold_min = %s,
                        best_trade = %s,
                        worst_trade = %s
                    WHERE session_id = %s
                    """,
                    (
                        now,
                        final_capital,
                        stats["total_trades"],
                        stats["wins"],
                        stats["losses"],
                        stats["total_pnl"],
                        stats["gross_profit"],
                        stats["gross_loss"],
                        stats["win_rate"],
                        stats["profit_factor"],
                        stats["avg_hold_min"],
                        stats["best_trade"],
                        stats["worst_trade"],
                        str(session.session_id),
                    )
                )
            conn.commit()

        # Update local session object
        session.stopped_at = now
        session.final_capital = Decimal(str(final_capital)) if final_capital else None
        session.total_trades = stats["total_trades"]
        session.wins = stats["wins"]
        session.losses = stats["losses"]
        session.total_pnl = Decimal(str(stats["total_pnl"]))
        session.gross_profit = Decimal(str(stats["gross_profit"]))
        session.gross_loss = Decimal(str(stats["gross_loss"]))
        session.win_rate = stats["win_rate"]
        session.profit_factor = stats["profit_factor"]
        session.avg_hold_min = stats["avg_hold_min"]
        session.best_trade = Decimal(str(stats["best_trade"])) if stats["best_trade"] else None
        session.worst_trade = Decimal(str(stats["worst_trade"])) if stats["worst_trade"] else None

        self._mode = "off"
        result = session
        self._current_session = None

        logger.info(f"Trading session stopped: {session.session_id}")
        return result

    def get_status(self) -> Dict:
        """Get current trading status."""
        if not self.is_trading:
            return {
                "is_running": False,
                "mode": "off",
                "started_at": None,
                "running_time_sec": 0,
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": None,
            }

        # Get live stats
        stats = self._calculate_session_stats(self._current_session.session_id)

        return {
            "is_running": True,
            "mode": self._mode,
            "session_id": str(self._current_session.session_id),
            "started_at": self._current_session.started_at.isoformat(),
            "running_time_sec": self._current_session.running_time_sec,
            "total_trades": stats["total_trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total_pnl": stats["total_pnl"],
            "gross_profit": stats["gross_profit"],
            "gross_loss": stats["gross_loss"],
            "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "avg_hold_min": stats["avg_hold_min"],
            "best_trade": stats["best_trade"],
            "worst_trade": stats["worst_trade"],
        }

    def get_stats(self, session_id: Optional[str] = None) -> Dict:
        """Get trading statistics for a session."""
        if session_id:
            sid = UUID(session_id)
        elif self._current_session:
            sid = self._current_session.session_id
        else:
            # Return stats for most recent session
            rows = fetch_all(
                "SELECT session_id FROM trading_sessions ORDER BY started_at DESC LIMIT 1"
            )
            if not rows:
                return self._empty_stats()
            sid = UUID(rows[0][0])

        stats = self._calculate_session_stats(sid)

        # Get session info
        rows = fetch_all(
            "SELECT started_at, mode FROM trading_sessions WHERE session_id = %s",
            (str(sid),)
        )
        if rows:
            started_at, mode = rows[0]
            stats["started_at"] = started_at.isoformat()
            stats["mode"] = mode
            stats["running_time_sec"] = int((datetime.utcnow() - started_at).total_seconds())

        return stats

    def get_trades(
        self,
        session_id: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
        symbol: Optional[str] = None,
    ) -> Dict:
        """Get trade history with pagination."""
        offset = (page - 1) * limit

        # Build query
        where_clauses = []
        params = []

        if session_id:
            where_clauses.append("session_id = %s")
            params.append(session_id)
        elif self._current_session:
            where_clauses.append("session_id = %s")
            params.append(str(self._current_session.session_id))

        if symbol:
            where_clauses.append("symbol = %s")
            params.append(symbol)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM trades WHERE {where_sql}"
        total = fetch_all(count_query, tuple(params))[0][0]

        # Get trades
        query = f"""
            SELECT trade_id, session_id, trade_group_id, symbol, side,
                   entry_time, exit_time, entry_price, exit_price, qty,
                   pnl, return_pct, hold_min, exit_reason, is_shadow
            FROM trades
            WHERE {where_sql}
            ORDER BY entry_time DESC
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        rows = fetch_all(query, tuple(params))

        trades = []
        for row in rows:
            trade = Trade(
                trade_id=UUID(row[0]) if isinstance(row[0], str) else row[0],
                session_id=UUID(row[1]) if row[1] else None,
                trade_group_id=UUID(row[2]) if isinstance(row[2], str) else row[2],
                symbol=row[3],
                side=row[4],
                entry_time=row[5],
                exit_time=row[6],
                entry_price=row[7],
                exit_price=row[8],
                qty=row[9],
                pnl=row[10],
                return_pct=float(row[11]) if row[11] else None,
                hold_min=row[12],
                exit_reason=row[13],
                is_shadow=row[14],
            )
            trades.append(trade.to_dict())

        return {
            "trades": trades,
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit if total > 0 else 0,
            "limit": limit,
        }

    def record_trade(
        self,
        trade_group_id: UUID,
        symbol: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        qty: float,
        exit_time: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> Trade:
        """Record a trade (entry or complete)."""
        session_id = str(self._current_session.session_id) if self._current_session else None
        is_shadow = self._mode == "shadow"
        trade_id = uuid4()

        hold_min = None
        return_pct = None
        if exit_time and entry_time:
            hold_min = int((exit_time - entry_time).total_seconds() / 60)
        if pnl and entry_price and qty:
            notional = entry_price * qty
            return_pct = (pnl / notional) * 100 if notional > 0 else 0

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trades (
                        trade_id, session_id, trade_group_id, symbol, side,
                        entry_time, exit_time, entry_price, exit_price, qty,
                        pnl, return_pct, hold_min, exit_reason, is_shadow
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(trade_id), session_id, str(trade_group_id), symbol, side,
                        entry_time, exit_time, entry_price, exit_price, qty,
                        pnl, return_pct, hold_min, exit_reason, is_shadow
                    )
                )
            conn.commit()

        return Trade(
            trade_id=trade_id,
            session_id=UUID(session_id) if session_id else None,
            trade_group_id=trade_group_id,
            symbol=symbol,
            side=side,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=Decimal(str(entry_price)),
            exit_price=Decimal(str(exit_price)) if exit_price else None,
            qty=Decimal(str(qty)),
            pnl=Decimal(str(pnl)) if pnl else None,
            return_pct=return_pct,
            hold_min=hold_min,
            exit_reason=exit_reason,
            is_shadow=is_shadow,
        )

    def _calculate_session_stats(self, session_id: UUID) -> Dict:
        """Calculate stats for a session from trades table."""
        rows = fetch_all(
            """
            SELECT
                COUNT(*) as total_trades,
                COUNT(*) FILTER (WHERE pnl > 0) as wins,
                COUNT(*) FILTER (WHERE pnl <= 0) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(pnl) FILTER (WHERE pnl > 0), 0) as gross_profit,
                COALESCE(ABS(SUM(pnl) FILTER (WHERE pnl < 0)), 0) as gross_loss,
                AVG(hold_min) as avg_hold_min,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades
            WHERE session_id = %s AND exit_time IS NOT NULL
            """,
            (str(session_id),)
        )

        if not rows or rows[0][0] == 0:
            return self._empty_stats()

        row = rows[0]
        total_trades = row[0]
        wins = row[1]
        losses = row[2]
        total_pnl = float(row[3])
        gross_profit = float(row[4])
        gross_loss = float(row[5])
        avg_hold_min = int(row[6]) if row[6] else None
        best_trade = float(row[7]) if row[7] else None
        worst_trade = float(row[8]) if row[8] else None

        win_rate = (wins / total_trades * 100) if total_trades > 0 else None
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "win_rate": round(win_rate, 2) if win_rate else None,
            "profit_factor": round(profit_factor, 2) if profit_factor else None,
            "avg_hold_min": avg_hold_min,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
        }

    def _empty_stats(self) -> Dict:
        """Return empty stats structure."""
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_hold_min": None,
            "best_trade": None,
            "worst_trade": None,
        }

    def _validate_live_start(self) -> None:
        """Validate conditions for starting live trading."""
        # TODO: Implement validation checks
        # - API key valid
        # - Sufficient balance
        # - Model loaded
        # - WebSocket connected
        pass


# Singleton instance
session_manager = TradingSessionManager()
