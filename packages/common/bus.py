from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple

import redis

from packages.common.config import get_settings


class RedisBus:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def publish(self, stream: str, payload: Dict[str, str]) -> str:
        return self.client.xadd(stream, payload)

    def publish_json(self, stream: str, payload: Dict) -> str:
        as_str = {"data": json.dumps(payload)}
        return self.publish(stream, as_str)

    def read(self, stream: str, last_id: str = "0-0", count: int = 100, block_ms: int = 1000) -> List[Tuple[str, Dict[str, str]]]:
        res = self.client.xread({stream: last_id}, count=count, block=block_ms)
        if not res:
            return []
        _, entries = res[0]
        return entries
