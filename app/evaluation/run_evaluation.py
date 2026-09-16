import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.investigation_agent import investigate
from app.agents.sql_agent import llm_sql
from app.db import engine_scope
from app.evaluation.basic_eval import evaluate_sql_quality
from app.rag.retriever import retrieve
from app.tools.sql_tool import SQLExecutionError


DEFAULT_CASES_PATH = Path("app/evaluation/benchmark_cases.json")
DEFAULT_REPORT_PATH = Path("app/evaluation/step10_eval_report.json")


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: dict[str, Any]


@dataclass
class SectionSummary:
    total: int
    passed: int
    failed: int
    pass_rate: float


def _load_cases(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_all_tokens(text: str, tokens: list[str]) -> bool:
    lower = text.lower()
    return all(token.lower() in lower for token in tokens)


def evaluate_sql_cases(sql_cases: list[dict[str, Any]]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in sql_cases:
        question = case["question"]
        generated = llm_sql(question)
        quality = evaluate_sql_quality(generated.sql)
        must_include = case.get("must_include", [])
        keyword_match = _contains_all_tokens(generated.sql, must_include)

        passed = quality.query_non_empty and quality.is_read_only and keyword_match
        results.append(
            CaseResult(
                name=case["name"],
                passed=passed,
                details={
                    "question": question,
                    "generated_sql": generated.sql,
                    "query_non_empty": quality.query_non_empty,
                    "is_read_only": quality.is_read_only,
                    "must_include": must_include,
                    "must_include_match": keyword_match,
                },
            )
        )
    return results


def evaluate_rag_cases(rag_cases: list[dict[str, Any]]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in rag_cases:
        query = case["question"]
        top_k = int(case.get("top_k", 3))
        chunks = retrieve(query, top_k=top_k)

        merged_text = "\n".join(chunk.text for chunk in chunks).lower()
        expected_terms = case.get("expected_terms", [])
        term_hits = {term: (term.lower() in merged_text) for term in expected_terms}

        passed = len(chunks) > 0 and all(term_hits.values())
        results.append(
            CaseResult(
                name=case["name"],
                passed=passed,
                details={
                    "query": query,
                    "top_k": top_k,
                    "retrieved_count": len(chunks),
                    "expected_terms": expected_terms,
                    "term_hits": term_hits,
                    "sources": [chunk.source for chunk in chunks],
                },
            )
        )
    return results


def evaluate_investigate_cases(investigate_cases: list[dict[str, Any]]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in investigate_cases:
        question = case["question"]
        execute_sql = bool(case.get("execute_sql", False))
        max_rows = int(case.get("max_rows", 50))
        risk_threshold = float(case.get("risk_threshold", 0.5))
        rag_top_k = int(case.get("rag_top_k", 3))
        min_sql_rows = int(case.get("min_sql_rows", 0))
        min_rag_chunks = int(case.get("min_rag_chunks", 1))
        require_scoring_when_sql = bool(case.get("require_scoring_when_sql", False))

        try:
            with engine_scope() as engine:
                outcome = investigate(
                    engine=engine,
                    question=question,
                    execute_sql=execute_sql,
                    max_rows=max_rows,
                    threshold=risk_threshold,
                    rag_top_k=rag_top_k,
                )
            sql_error = None
        except SQLExecutionError as exc:
            outcome = None
            sql_error = str(exc)
        except Exception as exc:
            outcome = None
            sql_error = str(exc)

        if outcome is None:
            results.append(
                CaseResult(
                    name=case["name"],
                    passed=False,
                    details={
                        "question": question,
                        "execute_sql": execute_sql,
                        "error": sql_error,
                    },
                )
            )
            continue

        pass_checks = {
            "generated_sql_non_empty": bool(outcome.generated_sql.strip()),
            "rag_chunks_min": len(outcome.rag_chunks) >= min_rag_chunks,
            "rag_answer_non_empty": bool(outcome.rag_answer.strip()),
            "sql_rows_min": (len(outcome.sql_rows) >= min_sql_rows) if execute_sql else True,
            "scoring_present": (outcome.scoring_applied if require_scoring_when_sql and execute_sql else True),
        }
        passed = all(pass_checks.values())

        results.append(
            CaseResult(
                name=case["name"],
                passed=passed,
                details={
                    "question": question,
                    "execute_sql": execute_sql,
                    "sql_rows": len(outcome.sql_rows),
                    "scored_rows": len(outcome.scored_rows),
                    "high_risk_count": outcome.high_risk_count,
                    "avg_predicted_probability": outcome.avg_predicted_probability,
                    "rag_chunks": len(outcome.rag_chunks),
                    "checks": pass_checks,
                },
            )
        )
    return results


def summarize(results: list[CaseResult]) -> SectionSummary:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0
    return SectionSummary(total=total, passed=passed, failed=failed, pass_rate=pass_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 10 evaluation benchmark")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to benchmark cases JSON")
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH), help="Path to write evaluation report JSON")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="Optional database URL for execute_sql cases")
    args = parser.parse_args()

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url

    cases_path = Path(args.cases)
    output_path = Path(args.output)

    cases = _load_cases(cases_path)

    sql_results = evaluate_sql_cases(cases.get("sql_cases", []))
    rag_results = evaluate_rag_cases(cases.get("rag_cases", []))
    investigate_results = evaluate_investigate_cases(cases.get("investigate_cases", []))

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_file": str(cases_path),
        "summary": {
            "sql": asdict(summarize(sql_results)),
            "rag": asdict(summarize(rag_results)),
            "investigate": asdict(summarize(investigate_results)),
            "overall": asdict(summarize(sql_results + rag_results + investigate_results)),
        },
        "results": {
            "sql": [asdict(r) for r in sql_results],
            "rag": [asdict(r) for r in rag_results],
            "investigate": [asdict(r) for r in investigate_results],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
