from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine


READ_ONLY_PREFIXES = ("select", "with")


class SQLExecutionError(Exception):
    pass


def ensure_read_only(sql: str) -> None:
    normalized = sql.strip().lower()
    if not normalized.startswith(READ_ONLY_PREFIXES):
        raise SQLExecutionError("Only read-only SELECT/CTE queries are allowed.")


def execute_query(engine: Engine, sql: str, max_rows: int = 200) -> list[dict]:
    ensure_read_only(sql)

    limited_sql = f"SELECT * FROM ({sql}) AS subq LIMIT {max_rows}"

    with engine.connect() as conn:
        result = conn.execute(text(limited_sql))
        columns: Sequence[str] = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]
