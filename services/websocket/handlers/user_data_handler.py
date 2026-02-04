"""Handler for userData websocket messages (order/position updates)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Dict, List, Optional

from packages.common.db import get_conn
from packages.common.symbol_map import to_internal

logger = logging.getLogger(__name__)


class UserDataHandler:
    """
    Processes userData websocket stream messages for real-time order/position updates.

    Event types:
    - ACCOUNT_UPDATE: Account balance and position changes
    - ORDER_TRADE_UPDATE: Order status updates and fills
    - ACCOUNT_CONFIG_UPDATE: Leverage and margin mode changes
    """

    def __init__(self):
        """Initialize user data handler."""
        self.positions: Dict[str, Dict] = {}
        self.open_orders: Dict[int, Dict] = {}
        self.account_balance: Optional[Dict] = None

        # Callbacks for order events
        self._on_order_fill_callbacks: List[Callable] = []
        self._on_position_update_callbacks: List[Callable] = []

    def on_order_fill(self, callback: Callable) -> None:
        """Register callback for order fill events."""
        self._on_order_fill_callbacks.append(callback)

    def on_position_update(self, callback: Callable) -> None:
        """Register callback for position update events."""
        self._on_position_update_callbacks.append(callback)

    def handle(self, data: dict) -> None:
        """
        Handle userData stream message.

        Routes to appropriate handler based on event type.
        """
        try:
            event_type = data.get("e")

            if event_type == "ORDER_TRADE_UPDATE":
                self._handle_order_update(data)
            elif event_type == "ACCOUNT_UPDATE":
                self._handle_account_update(data)
            elif event_type == "ACCOUNT_CONFIG_UPDATE":
                self._handle_config_update(data)
            else:
                logger.debug(f"Unknown userData event type: {event_type}")

        except Exception as e:
            logger.error(f"Failed to process userData: {e}")

    def _handle_order_update(self, data: dict) -> None:
        """
        Handle ORDER_TRADE_UPDATE event.

        Message format:
        {
            "e": "ORDER_TRADE_UPDATE",
            "T": 1699200000000,       # Transaction time
            "o": {
                "s": "BTCUSDT",        # Symbol
                "c": "my_order_id",    # Client order ID
                "S": "BUY",            # Side
                "o": "LIMIT",          # Order type
                "f": "GTC",            # Time in force
                "q": "0.001",          # Original quantity
                "p": "36000.00",       # Original price
                "ap": "0",             # Average price
                "sp": "0",             # Stop price
                "x": "NEW",            # Execution type
                "X": "NEW",            # Order status
                "i": 123456789,        # Order ID
                "l": "0",              # Last filled quantity
                "z": "0",              # Cumulative filled quantity
                "L": "0",              # Last filled price
                "n": "0",              # Commission
                "N": "USDT",           # Commission asset
                "T": 1699200000000,    # Order trade time
                "t": 0,                # Trade ID
                "rp": "0"              # Realized profit
            }
        }
        """
        order_data = data.get("o", {})

        symbol = to_internal(order_data.get("s", ""), "ws")
        order_id = order_data.get("i")
        client_order_id = order_data.get("c", "")
        side = order_data.get("S")
        order_type = order_data.get("o")
        status = order_data.get("X")
        execution_type = order_data.get("x")

        qty = Decimal(order_data.get("q", "0"))
        price = Decimal(order_data.get("p", "0"))
        avg_price = Decimal(order_data.get("ap", "0"))
        stop_price = Decimal(order_data.get("sp", "0"))

        filled_qty = Decimal(order_data.get("z", "0"))
        last_filled_qty = Decimal(order_data.get("l", "0"))
        last_filled_price = Decimal(order_data.get("L", "0"))

        commission = Decimal(order_data.get("n", "0"))
        commission_asset = order_data.get("N", "USDT")
        realized_pnl = Decimal(order_data.get("rp", "0"))

        trade_time = datetime.fromtimestamp(
            order_data.get("T", 0) / 1000, tz=timezone.utc
        )
        trade_id = order_data.get("t")

        # Update orders cache
        if status in ("FILLED", "CANCELED", "EXPIRED"):
            self.open_orders.pop(order_id, None)
        else:
            self.open_orders[order_id] = {
                "order_id": order_id,
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "status": status,
                "qty": float(qty),
                "price": float(price),
                "filled_qty": float(filled_qty),
                "avg_price": float(avg_price),
                "updated_at": trade_time,
            }

        # Update database
        self._update_order_in_db(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=status,
            qty=qty,
            price=price,
            stop_price=stop_price,
            updated_at=trade_time,
        )

        # If this is a fill, record it and notify callbacks
        if execution_type == "TRADE" and float(last_filled_qty) > 0:
            self._record_fill(
                trade_id=trade_id,
                order_id=order_id,
                symbol=symbol,
                price=last_filled_price,
                qty=last_filled_qty,
                commission=commission,
                commission_asset=commission_asset,
                realized_pnl=realized_pnl,
                ts=trade_time,
            )

            # Notify callbacks
            fill_event = {
                "type": "fill",
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "price": float(last_filled_price),
                "qty": float(last_filled_qty),
                "commission": float(commission),
                "realized_pnl": float(realized_pnl),
                "ts": trade_time,
            }
            for callback in self._on_order_fill_callbacks:
                try:
                    callback(fill_event)
                except Exception as e:
                    logger.error(f"Order fill callback error: {e}")

        logger.info(
            f"Order update: {symbol} {side} {status} "
            f"(id={order_id}, filled={filled_qty}/{qty})"
        )

    def _handle_account_update(self, data: dict) -> None:
        """
        Handle ACCOUNT_UPDATE event.

        Message format:
        {
            "e": "ACCOUNT_UPDATE",
            "T": 1699200000000,
            "a": {
                "B": [                 # Balances
                    {"a": "USDT", "wb": "100.00", "cw": "100.00"}
                ],
                "P": [                 # Positions
                    {
                        "s": "BTCUSDT",
                        "pa": "0.001",      # Position amount
                        "ep": "36000.00",   # Entry price
                        "up": "10.00",      # Unrealized PnL
                        "mt": "cross",      # Margin type
                        "iw": "0",          # Isolated wallet
                        "ps": "LONG"        # Position side
                    }
                ]
            }
        }
        """
        account_data = data.get("a", {})

        # Update balances
        balances = account_data.get("B", [])
        for balance in balances:
            asset = balance.get("a")
            wallet_balance = Decimal(balance.get("wb", "0"))
            cross_wallet = Decimal(balance.get("cw", "0"))

            if asset == "USDT":
                self.account_balance = {
                    "asset": asset,
                    "wallet_balance": float(wallet_balance),
                    "cross_wallet": float(cross_wallet),
                    "updated_at": datetime.now(timezone.utc),
                }

        # Update positions
        positions = account_data.get("P", [])
        for pos in positions:
            symbol = to_internal(pos.get("s", ""), "ws")
            position_amt = Decimal(pos.get("pa", "0"))
            entry_price = Decimal(pos.get("ep", "0"))
            unrealized_pnl = Decimal(pos.get("up", "0"))
            margin_type = pos.get("mt", "cross")
            position_side = pos.get("ps", "BOTH")

            position_data = {
                "symbol": symbol,
                "amt": float(position_amt),
                "entry_price": float(entry_price),
                "unrealized_pnl": float(unrealized_pnl),
                "margin_type": margin_type,
                "side": position_side,
                "updated_at": datetime.now(timezone.utc),
            }

            if float(position_amt) != 0:
                self.positions[symbol] = position_data
            else:
                self.positions.pop(symbol, None)

            # Notify callbacks
            for callback in self._on_position_update_callbacks:
                try:
                    callback(position_data)
                except Exception as e:
                    logger.error(f"Position update callback error: {e}")

        logger.debug(f"Account update: {len(positions)} positions")

    def _handle_config_update(self, data: dict) -> None:
        """
        Handle ACCOUNT_CONFIG_UPDATE event (leverage/margin mode changes).
        """
        logger.debug(f"Config update received: {data}")

    def _update_order_in_db(
        self,
        order_id: int,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        status: str,
        qty: Decimal,
        price: Decimal,
        stop_price: Decimal,
        updated_at: datetime,
    ) -> None:
        """Update order in database."""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO orders (
                            order_id, client_order_id, symbol, side, type,
                            status, reduce_only, price, stop_price, qty,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (order_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            order_id, client_order_id, symbol, side, order_type,
                            status, False, float(price), float(stop_price), float(qty),
                            updated_at, updated_at
                        )
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update order in DB: {e}")

    def _record_fill(
        self,
        trade_id: int,
        order_id: int,
        symbol: str,
        price: Decimal,
        qty: Decimal,
        commission: Decimal,
        commission_asset: str,
        realized_pnl: Decimal,
        ts: datetime,
    ) -> None:
        """Record fill in database."""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO fills (
                            trade_id, order_id, symbol, price, qty,
                            fee, fee_asset, realized_pnl, ts
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (trade_id) DO NOTHING
                        """,
                        (
                            trade_id, order_id, symbol, float(price), float(qty),
                            float(commission), commission_asset, float(realized_pnl), ts
                        )
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record fill in DB: {e}")

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get cached position for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all cached positions."""
        return self.positions.copy()

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders, optionally filtered by symbol."""
        orders = list(self.open_orders.values())
        if symbol:
            orders = [o for o in orders if o.get("symbol") == symbol]
        return orders

    def get_account_balance(self) -> Optional[Dict]:
        """Get cached account balance."""
        return self.account_balance
