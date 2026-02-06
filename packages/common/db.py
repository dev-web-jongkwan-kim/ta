from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_values

from packages.common.config import get_settings


@contextmanager
def get_conn():
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()


def bulk_upsert(
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str],
) -> int:
    if not rows:
        return 0

    cols_sql = ",".join(columns)
    conflict_sql = ",".join(conflict_cols)
    update_sql = ",".join([f"{col}=EXCLUDED.{col}" for col in update_cols])
    query = (
        f"INSERT INTO {table} ({cols_sql}) VALUES %s "
        f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {update_sql}"
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)
        conn.commit()
    return len(rows)


def fetch_all(query: str, params: Tuple | None = None) -> List[Tuple]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchall()


def fetch_one(query: str, params: Tuple | None = None) -> Tuple | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchone()


def execute(query: str, params: Tuple | None = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
        conn.commit()
