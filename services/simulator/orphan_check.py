from __future__ import annotations

from typing import List, Tuple

from packages.common.db import fetch_all
from packages.common.risk import log_risk_event


def _fetch_orphan_fills() -> List[Tuple]:
    return fetch_all(
        """
        SELECT f.trade_id, f.order_id, f.symbol, f.trade_group_id
        FROM fills f
        LEFT JOIN orders o ON o.order_id = f.order_id
        WHERE o.order_id IS NULL
        """
    )


def _fetch_incomplete_trades() -> List[Tuple]:
    return fetch_all(
        """
        SELECT s.trade_group_id, s.symbol, s.ts,
            COALESCE(o.order_count, 0) AS order_count,
            COALESCE(f.fill_count, 0) AS fill_count
        FROM signals s
        LEFT JOIN (
            SELECT trade_group_id, COUNT(*) AS order_count
            FROM orders
            GROUP BY trade_group_id
        ) o ON o.trade_group_id = s.trade_group_id
        LEFT JOIN (
            SELECT trade_group_id, COUNT(*) AS fill_count
            FROM fills
            GROUP BY trade_group_id
        ) f ON f.trade_group_id = s.trade_group_id
        WHERE COALESCE(o.order_count, 0) < 2 OR COALESCE(f.fill_count, 0) < 2
        ORDER BY s.ts DESC
        LIMIT %s
        """,
        (20,),
    )


def _count_incomplete_trade_groups() -> int:
    row = fetch_all(
        """
        SELECT COUNT(*)
        FROM (
            SELECT s.trade_group_id
            FROM signals s
            LEFT JOIN (
                SELECT trade_group_id, COUNT(*) AS order_count
                FROM orders
                GROUP BY trade_group_id
            ) o ON o.trade_group_id = s.trade_group_id
            LEFT JOIN (
                SELECT trade_group_id, COUNT(*) AS fill_count
                FROM fills
                GROUP BY trade_group_id
            ) f ON f.trade_group_id = s.trade_group_id
            WHERE COALESCE(o.order_count, 0) < 2 OR COALESCE(f.fill_count, 0) < 2
        ) q
        """
    )
    return row[0][0] if row and row[0] else 0


def log_orphan_findings() -> None:
    orphan_fills = _fetch_orphan_fills()
    if orphan_fills:
        log_risk_event(
            "ORPHAN_FILL",
            f"{len(orphan_fills)} fills have no matching order",
            severity=4,
            details={"sample": orphan_fills[:5]},
        )
        print(f"Detected {len(orphan_fills)} orphan fills (sample first 5 printed)")
        for row in orphan_fills[:5]:
            print("  ", row)

    total_groups_row = fetch_all("SELECT COUNT(DISTINCT trade_group_id) FROM signals")
    total_groups = total_groups_row[0][0] if total_groups_row and total_groups_row[0] else 0
    incomplete = _count_incomplete_trade_groups()
    sample = _fetch_incomplete_trades()
    if incomplete:
        log_risk_event(
            "INCOMPLETE_TRADE_GROUP",
            f"{incomplete} of {total_groups} trade groups missing order/fill records",
            severity=3,
            details={"total_trade_groups": total_groups, "incomplete": incomplete},
        )
        print(f"{incomplete} trade groups incomplete out of {total_groups}")
        for row in sample:
            print("  sample:", row)


def main() -> None:
    print("Running simulator orphan check...")
    total_groups_row = fetch_all("SELECT COUNT(DISTINCT trade_group_id) FROM signals")
    total_groups = total_groups_row[0][0] if total_groups_row and total_groups_row[0] else 0
    print(f"Total trade groups observed: {total_groups}")
    log_orphan_findings()


if __name__ == "__main__":
    main()
