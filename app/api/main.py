from fastapi import FastAPI, HTTPException

from app.agents.investigation_agent import investigate
from app.agents.sql_agent import llm_sql
from app.config import get_settings
from app.db import engine_scope
from app.models.schemas import (
    AskRequest,
    AskResponse,
    InvestigateRequest,
    InvestigateResponse,
    RagAnswerRequest,
    RagAnswerResponse,
    RagChunk,
    RagRetrieveRequest,
    RagRetrieveResponse,
)
from app.rag.retriever import answer_with_context, retrieve
from app.tools.sql_tool import SQLExecutionError, execute_query

app = FastAPI(title="Industrial Reliability Copilot", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    agent_result = llm_sql(payload.question)

    if not payload.execute_sql:
        return AskResponse(
            question=payload.question,
            generated_sql=agent_result.sql,
            explanation=agent_result.explanation,
            rows=None,
        )

    try:
        with engine_scope() as engine:
            rows = execute_query(engine, agent_result.sql)
    except SQLExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {exc}") from exc

    return AskResponse(
        question=payload.question,
        generated_sql=agent_result.sql,
        explanation=agent_result.explanation,
        rows=rows,
    )


@app.post("/rag/retrieve", response_model=RagRetrieveResponse)
def rag_retrieve(payload: RagRetrieveRequest) -> RagRetrieveResponse:
    settings = get_settings()
    top_k = payload.top_k or settings.rag_default_top_k
    chunks = retrieve(query=payload.query, top_k=top_k)
    return RagRetrieveResponse(
        query=payload.query,
        count=len(chunks),
        chunks=[RagChunk(source=c.source, score=c.score, text=c.text) for c in chunks],
    )


@app.post("/rag/answer", response_model=RagAnswerResponse)
def rag_answer(payload: RagAnswerRequest) -> RagAnswerResponse:
    settings = get_settings()
    top_k = payload.top_k or settings.rag_default_top_k

    try:
        answer, context_chunks = answer_with_context(payload.question, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG answer failed: {exc}") from exc

    return RagAnswerResponse(
        question=payload.question,
        answer=answer,
        context_chunks=[RagChunk(source=c.source, score=c.score, text=c.text) for c in context_chunks],
    )


@app.post("/investigate", response_model=InvestigateResponse)
def investigate_issue(payload: InvestigateRequest) -> InvestigateResponse:
    try:
        with engine_scope() as engine:
            result = investigate(
                engine=engine,
                question=payload.question,
                execute_sql=payload.execute_sql,
                max_rows=payload.max_rows,
                threshold=payload.risk_threshold,
                rag_top_k=payload.rag_top_k,
            )
    except SQLExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {exc}") from exc

    return InvestigateResponse(
        question=result.question,
        generated_sql=result.generated_sql,
        explanation=result.explanation,
        sql_rows=result.sql_rows,
        scored_rows=result.scored_rows,
        model_available=result.model_available,
        scoring_applied=result.scoring_applied,
        selected_threshold=result.selected_threshold,
        high_risk_count=result.high_risk_count,
        avg_predicted_probability=result.avg_predicted_probability,
        rag_answer=result.rag_answer,
        rag_chunks=[RagChunk(source=c.source, score=c.score, text=c.text) for c in result.rag_chunks],
    )
