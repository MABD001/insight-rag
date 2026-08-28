"""Request/response models. These drive the generated OpenAPI docs at /docs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class CitationOut(BaseModel):
    marker: int
    source: str
    title: str
    label: str
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    snippet: str
    score: float
    chunk_id: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    grounded: bool
    query_used: str
    grounding_coverage: float
    refusal_reason: str | None = None
    citations: list[CitationOut]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timings_ms: dict[str, float]


class IngestPathRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    force: bool = False


class IngestedDocument(BaseModel):
    document_id: str
    source: str
    chunks: int
    skipped: bool


class IngestResponse(BaseModel):
    documents: list[IngestedDocument]
    total_chunks: int


class SearchResult(BaseModel):
    marker: int
    label: str
    source: str
    page: int | None = None
    text: str
    fused_score: float
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    vector_store: str
    reranker: str
    documents: int
    chunks: int
