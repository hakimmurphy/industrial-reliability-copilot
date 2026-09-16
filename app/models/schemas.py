from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5)
    execute_sql: bool = False


class AskResponse(BaseModel):
    question: str
    generated_sql: str
    explanation: str
    rows: list[dict] | None = None


class RagRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class RagChunk(BaseModel):
    source: str
    score: float
    text: str


class RagRetrieveResponse(BaseModel):
    query: str
    count: int
    chunks: list[RagChunk]


class RagAnswerRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class RagAnswerResponse(BaseModel):
    question: str
    answer: str
    context_chunks: list[RagChunk]


class InvestigateRequest(BaseModel):
    question: str = Field(..., min_length=5)
    execute_sql: bool = True
    max_rows: int = Field(default=50, ge=1, le=500)
    risk_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    rag_top_k: int = Field(default=5, ge=1, le=20)


class InvestigateResponse(BaseModel):
    question: str
    generated_sql: str
    explanation: str
    sql_rows: list[dict]
    scored_rows: list[dict]
    model_available: bool
    scoring_applied: bool
    selected_threshold: float
    high_risk_count: int
    avg_predicted_probability: float | None = None
    rag_answer: str
    rag_chunks: list[RagChunk]
