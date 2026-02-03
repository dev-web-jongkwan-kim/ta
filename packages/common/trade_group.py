from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Union
import uuid


def build_trade_group_id(symbol: str, ts: Union[str, datetime], side: str) -> str:
    if isinstance(ts, datetime):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)
    normalized = f"{symbol.upper()}:{ts_str}:{side.upper()}"
    digest = sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return str(uuid.UUID(digest))
