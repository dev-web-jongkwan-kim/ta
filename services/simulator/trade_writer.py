from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg2 import sql

from packages.common.db import get_conn


class TradeGroupWriter:
    @staticmethod
    def _insert_signal(cur: Any, row: Dict[str, Any]) -> None:
        query = sql.SQL(
            """
            INSERT INTO signals (
                symbol, ts, model_id, ev_long, ev_short, er_long, er_short, q05_long, q05_short,
                e_mae_long, e_mae_short, e_hold_long_min, e_hold_short_min, decision,
                size_notional, leverage, sl_price, tp_price, block_reason_codes, explain, trade_group_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, ts) DO UPDATE SET
            ev_long = EXCLUDED.ev_long,
            ev_short = EXCLUDED.ev_short,
            er_long = EXCLUDED.er_long,
            er_short = EXCLUDED.er_short,
            q05_long = EXCLUDED.q05_long,
            q05_short = EXCLUDED.q05_short,
            e_mae_long = EXCLUDED.e_mae_long,
            e_mae_short = EXCLUDED.e_mae_short,
            e_hold_long_min = EXCLUDED.e_hold_long_min,
            e_hold_short_min = EXCLUDED.e_hold_short_min,
            decision = EXCLUDED.decision,
            size_notional = EXCLUDED.size_notional,
            leverage = EXCLUDED.leverage,
            sl_price = EXCLUDED.sl_price,
            tp_price = EXCLUDED.tp_price,
            block_reason_codes = EXCLUDED.block_reason_codes,
            explain = EXCLUDED.explain,
            trade_group_id = EXCLUDED.trade_group_id
            """
        )
        cur.execute(
            query,
            (
                row["symbol"],
                row["ts"],
                row["model_id"],
                row["ev_long"],
                row["ev_short"],
                row.get("er_long"),
                row.get("er_short"),
                row.get("q05_long"),
                row.get("q05_short"),
                row.get("e_mae_long"),
                row.get("e_mae_short"),
                row.get("e_hold_long_min"),
                row.get("e_hold_short_min"),
                row["decision"],
                row.get("size_notional"),
                row.get("leverage"),
                row.get("sl_price"),
                row.get("tp_price"),
                row.get("block_reason_codes"),
                row.get("explain"),
                row["trade_group_id"],
            ),
        )

    @staticmethod
    def _insert_order(cur: Any, order: Dict[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO orders (order_id, client_order_id, symbol, side, type, status, reduce_only,
                price, stop_price, qty, created_at, updated_at, trade_group_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (order_id) DO UPDATE SET updated_at = EXCLUDED.updated_at
            """,
            (
                order["order_id"],
                order.get("client_order_id"),
                order["symbol"],
                order["side"],
                order["type"],
                order["status"],
                order.get("reduce_only", False),
                order.get("price"),
                order.get("stop_price"),
                order["qty"],
                order.get("created_at"),
                order.get("updated_at"),
                order.get("trade_group_id"),
            ),
        )

    @staticmethod
    def _insert_fill(cur: Any, fill: Dict[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO fills (trade_id, order_id, symbol, price, qty, fee, fee_asset, realized_pnl, ts, trade_group_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (trade_id) DO NOTHING
            """,
            (
                fill["trade_id"],
                fill["order_id"],
                fill["symbol"],
                fill["price"],
                fill["qty"],
                fill.get("fee", 0.0),
                fill.get("fee_asset"),
                fill.get("realized_pnl"),
                fill.get("ts"),
                fill.get("trade_group_id"),
            ),
        )

    @staticmethod
    def _insert_position_event(cur: Any, event: Dict[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO positions (
                trade_group_id, symbol, ts, side, amt, entry_price, mark_price, leverage,
                margin_type, liquidation_price, notional, unrealized_pnl, event_type
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (trade_group_id, symbol, ts, side) DO UPDATE SET
                amt = EXCLUDED.amt,
                entry_price = EXCLUDED.entry_price,
                mark_price = EXCLUDED.mark_price,
                leverage = EXCLUDED.leverage,
                margin_type = EXCLUDED.margin_type,
                liquidation_price = EXCLUDED.liquidation_price,
                notional = EXCLUDED.notional,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                event_type = EXCLUDED.event_type
            """,
            (
                event["trade_group_id"],
                event["symbol"],
                event["ts"],
                event["side"],
                event.get("amt"),
                event.get("entry_price"),
                event.get("mark_price"),
                event.get("leverage"),
                event.get("margin_type"),
                event.get("liquidation_price"),
                event.get("notional"),
                event.get("unrealized_pnl"),
                event.get("event_type", "FINAL"),
            ),
        )

    @classmethod
    def persist_trade(
        cls,
        signal: Dict[str, Any],
        orders: List[Dict[str, Any]],
        fills: List[Dict[str, Any]],
        position_events: List[Dict[str, Any]],
    ) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cls._insert_signal(cur, signal)
                for order in orders:
                    cls._insert_order(cur, order)
                for fill in fills:
                    cls._insert_fill(cur, fill)
                for event in position_events:
                    cls._insert_position_event(cur, event)
            conn.commit()
