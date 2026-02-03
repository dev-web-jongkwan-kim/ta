from __future__ import annotations

from typing import Literal


def _strip_separators(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").replace("-", "").replace("_", "")


def validate_internal(internal: str) -> None:
    if not internal or internal.upper() != internal:
        raise ValueError(f"Invalid internal symbol casing: {internal}")
    if any(ch in internal for ch in ["/", ":", "-", "_"]):
        raise ValueError(f"Invalid internal symbol separators: {internal}")


def to_internal(symbol: str, source: Literal["rest", "ws", "ccxt"]) -> str:
    if source == "ws":
        internal = _strip_separators(symbol).upper()
    elif source == "ccxt":
        internal = _strip_separators(symbol).upper()
        if internal.endswith("USDTUSDT"):
            internal = internal.replace("USDTUSDT", "USDT")
    else:
        internal = _strip_separators(symbol).upper()
    validate_internal(internal)
    return internal


def to_ws(internal: str) -> str:
    validate_internal(internal)
    return internal.lower()


def to_rest(internal: str) -> str:
    validate_internal(internal)
    return internal
