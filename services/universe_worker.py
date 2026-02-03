from __future__ import annotations

import time

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, execute


def sync_fixed_universe() -> list[str]:
    """고정된 UNIVERSE를 instruments 테이블에 동기화"""
    settings = get_settings()
    symbols = settings.universe_list()

    # instruments 테이블에 등록
    rows = [(s, "active", "A") for s in symbols]
    bulk_upsert(
        "instruments",
        ["symbol", "status", "liquidity_tier"],
        rows,
        conflict_cols=["symbol"],
        update_cols=["status", "liquidity_tier"],
    )

    # UNIVERSE에 없는 종목은 inactive로
    if symbols:
        execute(
            "UPDATE instruments SET status='inactive' WHERE symbol NOT IN %s",
            (tuple(symbols),),
        )

    return symbols


def main() -> None:
    settings = get_settings()
    print(f"Fixed Universe Mode: {len(settings.universe_list())} symbols")
    print(f"Symbols: {settings.universe_list()}")

    while True:
        try:
            symbols = sync_fixed_universe()
            print(f"Universe synced: {len(symbols)} symbols (fixed)")
        except Exception as exc:  # noqa: BLE001
            print(f"Universe sync failed: {exc}")
        # 1시간마다만 동기화 (고정이므로 자주 할 필요 없음)
        time.sleep(3600)


if __name__ == "__main__":
    main()
