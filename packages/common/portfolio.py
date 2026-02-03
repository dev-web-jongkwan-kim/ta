from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from packages.common.db import fetch_all


def _today_midnight() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def get_portfolio_metrics() -> Dict[str, float]:
    rows = fetch_all(
        """
        SELECT
            COUNT(*) FILTER (WHERE event_type = 'OPEN' AND amt > 0) AS open_positions,
            COALESCE(SUM(CASE WHEN event_type = 'OPEN' AND amt > 0 THEN notional ELSE 0 END), 0) AS total_notional,
            COALESCE(SUM(CASE WHEN event_type = 'OPEN' AND amt > 0 AND side = 'BUY' THEN notional ELSE 0 END), 0) AS long_notional,
            COALESCE(SUM(CASE WHEN event_type = 'OPEN' AND amt > 0 AND side = 'SELL' THEN notional ELSE 0 END), 0) AS short_notional
        FROM positions
        """
    )
    open_positions = rows[0][0] if rows else 0
    total = float(rows[0][1] or 0.0)
    long = float(rows[0][2] or 0.0)
    short = float(rows[0][3] or 0.0)

    today = _today_midnight()
    pnl_rows = fetch_all(
        """
        SELECT
            COALESCE(SUM(realized_pnl), 0) AS realized_pnl,
            COALESCE(SUM(CASE WHEN realized_pnl < 0 THEN -realized_pnl ELSE 0 END), 0) AS negative_pnl
        FROM fills
        WHERE ts >= %s
        """,
        (today,),
    )
    realized = float(pnl_rows[0][0] if pnl_rows else 0.0)
    negative = float(pnl_rows[0][1] if pnl_rows else 0.0)

    return {
        "open_positions": float(open_positions),
        "total_notional": total,
        "long_notional": long,
        "short_notional": short,
        "daily_pnl": realized,
        "daily_loss": negative,
    }
