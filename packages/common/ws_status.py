"""WebSocket status tracking."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Dict

from packages.common.db import execute, fetch_one

# In-memory status (updated by websocket worker)
_status = {
    "connected": False,
    "started_at": None,
    "last_message_at": None,
    "reconnect_count": 0,
    "message_counts": {},
    "streams_active": 0,
    "streams_total": 0,
}


def set_ws_connected(connected: bool) -> None:
    """Set websocket connection status."""
    _status["connected"] = connected
    if connected and not _status["started_at"]:
        _status["started_at"] = time.time()


def set_ws_reconnect_count(count: int) -> None:
    """Set reconnect attempt count."""
    _status["reconnect_count"] = count


def set_ws_streams(active: int, total: int) -> None:
    """Set stream counts."""
    _status["streams_active"] = active
    _status["streams_total"] = total


def increment_ws_message(stream_type: str) -> None:
    """Increment message count for a stream type."""
    _status["message_counts"][stream_type] = _status["message_counts"].get(stream_type, 0) + 1
    _status["last_message_at"] = time.time()


def get_ws_status() -> dict:
    """Get current websocket status."""
    now = time.time()
    started_at = _status.get("started_at")
    last_message_at = _status.get("last_message_at")

    uptime_sec = int(now - started_at) if started_at else 0
    last_message_ago = now - last_message_at if last_message_at else None

    return {
        "connected": _status["connected"],
        "uptime_sec": uptime_sec,
        "last_message_ago_sec": last_message_ago,
        "reconnect_count": _status["reconnect_count"],
        "streams_active": _status["streams_active"],
        "streams_total": _status["streams_total"],
        "message_counts": _status["message_counts"].copy(),
        "total_messages": sum(_status["message_counts"].values()),
    }


def reset_ws_status() -> None:
    """Reset status (on disconnect)."""
    _status["connected"] = False
    _status["last_message_at"] = None
