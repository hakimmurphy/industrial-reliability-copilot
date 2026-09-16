from dataclasses import dataclass


@dataclass
class EvaluationResult:
    query_non_empty: bool
    is_read_only: bool


READ_ONLY_PREFIXES = ("select", "with")


def evaluate_sql_quality(sql: str) -> EvaluationResult:
    normalized = sql.strip().lower()
    return EvaluationResult(
        query_non_empty=bool(normalized),
        is_read_only=normalized.startswith(READ_ONLY_PREFIXES),
    )
