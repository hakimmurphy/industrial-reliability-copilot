from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

import joblib
import pandas as pd
from sqlalchemy.engine import Engine

from app.agents.sql_agent import llm_sql
from app.rag.retriever import RetrievedChunk, answer_with_context
from app.tools.sql_tool import execute_query

MODEL_PATH = Path("app/models/baseline_failure_model.joblib")
REQUIRED_FEATURES = [
    "machine_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "delta_temp_k",
    "power_proxy",
]


@dataclass
class InvestigationResult:
    question: str
    generated_sql: str
    explanation: str
    sql_rows: list[dict[str, Any]]
    scored_rows: list[dict[str, Any]]
    model_available: bool
    scoring_applied: bool
    selected_threshold: float
    high_risk_count: int
    avg_predicted_probability: float | None
    rag_answer: str
    rag_chunks: list[RetrievedChunk]


@lru_cache(maxsize=1)
def _load_model(model_path: str = str(MODEL_PATH)) -> Any | None:
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def _can_score_rows(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    columns = set(rows[0].keys())
    return all(feature in columns for feature in REQUIRED_FEATURES)


def _score_rows(rows: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], int, float | None]:
    model = _load_model()
    if model is None or not _can_score_rows(rows):
        return rows, 0, None

    frame = pd.DataFrame(rows)
    probs = model.predict_proba(frame[REQUIRED_FEATURES])[:, 1]

    scored_rows: list[dict[str, Any]] = []
    high_risk_count = 0
    for row, prob in zip(rows, probs, strict=True):
        prob_f = float(prob)
        predicted_failure = prob_f >= threshold
        if predicted_failure:
            high_risk_count += 1

        out = dict(row)
        out["predicted_failure_probability"] = round(prob_f, 6)
        out["predicted_failure"] = bool(predicted_failure)
        scored_rows.append(out)

    return scored_rows, high_risk_count, float(mean([float(p) for p in probs]))


def investigate(
    engine: Engine,
    question: str,
    execute_sql: bool,
    max_rows: int,
    threshold: float,
    rag_top_k: int,
) -> InvestigationResult:
    sql_result = llm_sql(question)
    sql_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    model_available = _load_model() is not None
    scoring_applied = False
    high_risk_count = 0
    avg_prob: float | None = None

    if execute_sql:
        sql_rows = execute_query(engine=engine, sql=sql_result.sql, max_rows=max_rows)
        scored_rows, high_risk_count, avg_prob = _score_rows(sql_rows, threshold=threshold)
        scoring_applied = bool(sql_rows) and avg_prob is not None

    rag_prompt = question
    if scoring_applied:
        rag_prompt = (
            f"{question}\n"
            f"High risk rows above threshold {threshold}: {high_risk_count}. "
            f"Average predicted probability: {avg_prob:.4f}."
        )

    rag_answer, rag_chunks = answer_with_context(rag_prompt, top_k=rag_top_k)

    return InvestigationResult(
        question=question,
        generated_sql=sql_result.sql,
        explanation=sql_result.explanation,
        sql_rows=sql_rows,
        scored_rows=scored_rows,
        model_available=model_available,
        scoring_applied=scoring_applied,
        selected_threshold=threshold,
        high_risk_count=high_risk_count,
        avg_predicted_probability=avg_prob,
        rag_answer=rag_answer,
        rag_chunks=rag_chunks,
    )
