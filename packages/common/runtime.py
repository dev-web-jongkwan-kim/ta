from __future__ import annotations

import json

import redis

from packages.common.config import get_settings


MODE_KEY = "ta:mode"
USERSTREAM_KEY = "ta:userstream_ok"
COLLECTOR_KEY = "ta:collector_ok"
DRIFT_STATUS_KEY = "ta:drift_status"
DRIFT_METRICS_KEY = "ta:drift_metrics"


def _client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def get_mode() -> str:
    client = _client()
    mode = client.get(MODE_KEY)
    return mode or get_settings().mode


def set_mode(mode: str) -> None:
    _client().set(MODE_KEY, mode)


def set_userstream_ok(ok: bool) -> None:
    _client().set(USERSTREAM_KEY, "1" if ok else "0")


def get_userstream_ok() -> bool:
    val = _client().get(USERSTREAM_KEY)
    return val == "1" if val is not None else True


def set_collector_ok(ok: bool) -> None:
    _client().set(COLLECTOR_KEY, "1" if ok else "0")


def get_collector_ok() -> bool:
    val = _client().get(COLLECTOR_KEY)
    return val == "1" if val is not None else True


def set_drift_status(status: str) -> None:
    _client().set(DRIFT_STATUS_KEY, status)


def get_drift_status() -> str:
    status = _client().get(DRIFT_STATUS_KEY)
    return status or "ok"


def set_drift_metrics(metrics: Dict[str, Any]) -> None:
    _client().set(DRIFT_METRICS_KEY, json.dumps(metrics))


def get_drift_metrics() -> Dict[str, Any]:
    payload = _client().get(DRIFT_METRICS_KEY)
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}
