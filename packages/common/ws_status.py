"""WebSocket status tracking via Redis for cross-container sharing."""
from __future__ import annotations

import json
import time
from typing import Dict, Optional

import redis

from packages.common.config import get_settings

_redis_client: Optional[redis.Redis] = None
_WS_STATUS_KEY = "ws:status"

# TTL extended to 30 minutes - heartbeat will refresh this periodically
_STATUS_TTL_SEC = 1800


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


def _get_status() -> Dict:
    """Get status from Redis."""
    try:
        r = _get_redis()
        data = r.get(_WS_STATUS_KEY)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return {
        "connected": False,
        "started_at": None,
        "last_message_at": None,
        "last_heartbeat_at": None,
        "reconnect_count": 0,
        "message_counts": {},
        "streams_active": 0,
        "streams_total": 0,
        "connection_count": 0,
    }


def _set_status(status: Dict) -> None:
    """Save status to Redis with extended TTL."""
    try:
        r = _get_redis()
        r.set(_WS_STATUS_KEY, json.dumps(status), ex=_STATUS_TTL_SEC)
    except Exception:
        pass


def set_ws_connected(connected: bool) -> None:
    """Set websocket connection status."""
    status = _get_status()
    status["connected"] = connected
    if connected:
        if not status["started_at"]:
            status["started_at"] = time.time()
        status["last_heartbeat_at"] = time.time()
    _set_status(status)


def set_ws_reconnect_count(count: int) -> None:
    """Set reconnect attempt count."""
    status = _get_status()
    status["reconnect_count"] = count
    _set_status(status)


def set_ws_streams(active: int, total: int) -> None:
    """Set stream counts."""
    status = _get_status()
    status["streams_active"] = active
    status["streams_total"] = total
    _set_status(status)


def set_ws_connection_count(count: int) -> None:
    """Set number of active WebSocket connections."""
    status = _get_status()
    status["connection_count"] = count
    _set_status(status)


def increment_ws_message(stream_type: str) -> None:
    """Increment message count for a stream type."""
    status = _get_status()
    status["message_counts"][stream_type] = status["message_counts"].get(stream_type, 0) + 1
    status["last_message_at"] = time.time()
    _set_status(status)


def refresh_ws_heartbeat() -> None:
    """
    Refresh heartbeat timestamp and TTL without changing other values.

    This should be called periodically (e.g., every 30 seconds) by the worker
    to keep the status alive in Redis even when no messages are arriving.
    """
    status = _get_status()
    if status["connected"]:
        status["last_heartbeat_at"] = time.time()
        _set_status(status)


def get_ws_status() -> dict:
    """Get current websocket status with computed fields."""
    status = _get_status()
    now = time.time()
    started_at = status.get("started_at")
    last_message_at = status.get("last_message_at")
    last_heartbeat_at = status.get("last_heartbeat_at")

    uptime_sec = int(now - started_at) if started_at else 0
    last_message_ago = now - last_message_at if last_message_at else None
    last_heartbeat_ago = now - last_heartbeat_at if last_heartbeat_at else None

    # Connection is considered alive if heartbeat was within last 60 seconds
    # This handles the case where Redis key exists but connection might be stale
    is_alive = (
        status["connected"] and
        last_heartbeat_at is not None and
        (now - last_heartbeat_at) < 60
    )

    return {
        "connected": is_alive,
        "uptime_sec": uptime_sec,
        "last_message_ago_sec": last_message_ago,
        "last_heartbeat_ago_sec": last_heartbeat_ago,
        "reconnect_count": status["reconnect_count"],
        "streams_active": status["streams_active"],
        "streams_total": status["streams_total"],
        "connection_count": status.get("connection_count", 0),
        "message_counts": status["message_counts"].copy(),
        "total_messages": sum(status["message_counts"].values()),
    }


def reset_ws_status() -> None:
    """
    Reset status on disconnect.

    Preserves started_at for uptime tracking across reconnections.
    """
    status = _get_status()
    status["connected"] = False
    # Don't clear started_at - we want uptime to persist across reconnections
    # Don't clear last_message_at - it's useful for debugging
    status["last_heartbeat_at"] = None
    _set_status(status)
