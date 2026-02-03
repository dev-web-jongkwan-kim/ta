from packages.common.config import get_settings
from packages.common.db import bulk_upsert


def main() -> None:
    settings = get_settings()
    symbols = settings.universe_list()
    rows = [(s, "active", "A") for s in symbols]
    bulk_upsert(
        "instruments",
        ["symbol", "status", "liquidity_tier"],
        rows,
        conflict_cols=["symbol"],
        update_cols=["status", "liquidity_tier"],
    )
    print(f"Inserted/updated {len(rows)} instruments")


if __name__ == "__main__":
    main()
