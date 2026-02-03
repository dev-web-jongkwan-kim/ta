from __future__ import annotations

import json
from typing import Any, Dict, Optional

from packages.common.db import execute


def log_risk_event(
    event_type: str,
    message: str,
    symbol: str = "",
    severity: int = 2,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    execute(
        """
        INSERT INTO risk_events (ts, event_type, symbol, severity, message, details)
        VALUES (now(), %s, %s, %s, %s, %s)
        """,
        (event_type, symbol, severity, message, json.dumps(details or {})),
    )
