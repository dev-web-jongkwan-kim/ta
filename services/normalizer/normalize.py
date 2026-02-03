from __future__ import annotations

from packages.common.symbol_map import to_internal


def normalize_symbol(symbol: str, source: str) -> str:
    return to_internal(symbol, source)
