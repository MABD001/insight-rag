"""HTTP surface."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestedDocument,
    IngestPathRequest,
    IngestResponse,
    SearchResponse,
    SearchResult,
)
from app.ingest.loaders import SUPPORTED_SUFFIXES, load_bytes, load_path
from app.rag.pipeline import RagPipeline

router = APIRouter()


def _pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    pipeline = _pipeline(request)
    stats = await pipeline.store.stats()
    reranker = getattr(pipeline.reranker, "active_backend", pipeline.reranker.name)
    return HealthResponse(
        status="ok",
        llm_provider=pipeline.settings.llm_provider,
        embedding_provider=pipeline.settings.embedding_provider,
        vector_store=pipeline.settings.vector_store,
        reranker=reranker,
        documents=stats["documents"],
        chunks=stats["chunks"],
    )


@router.post("/ingest/files", response_model=IngestResponse, tags=["ingest"])
async def ingest_files(request: Request, files: list[UploadFile]) -> IngestResponse:
    pipeline = _pipeline(request)
    results = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type {suffix!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
            )
        try:
            document = load_bytes(await upload.read(), filename=upload.filename or "upload")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        results.append(await pipeline.ingest(document))

    return IngestResponse(
        documents=[IngestedDocument(**asdict(r)) for r in results],
        total_chunks=sum(r.chunks for r in results),
    )


@router.post("/ingest/paths", response_model=IngestResponse, tags=["ingest"])
async def ingest_paths(request: Request, body: IngestPathRequest) -> IngestResponse:
    pipeline = _pipeline(request)
    results = []
    for raw in body.paths:
        path = Path(raw)
        candidates = (
            sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            if path.is_dir()
            else [path]
        )
        if not candidates:
            raise HTTPException(status_code=404, detail=f"No ingestable files at {raw!r}")
        for candidate in candidates:
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"Not found: {candidate}")
            results.append(await pipeline.ingest(load_path(candidate), force=body.force))

    return IngestResponse(
        documents=[IngestedDocument(**asdict(r)) for r in results],
        total_chunks=sum(r.chunks for r in results),
    )


@router.delete("/documents/{document_id}", tags=["ingest"])
async def delete_document(request: Request, document_id: str) -> dict[str, int | str]:
    pipeline = _pipeline(request)
    removed = await pipeline.store.delete_document(document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Unknown document {document_id!r}")
    await pipeline.refresh_sparse_index()
    return {"document_id": document_id, "chunks_removed": removed}


@router.get("/search", response_model=SearchResponse, tags=["retrieval"])
async def search(request: Request, q: str) -> SearchResponse:
    """Retrieval without generation — the endpoint you use to debug relevance."""
    pipeline = _pipeline(request)
    retrieved = await pipeline.retrieve(q)
    return SearchResponse(
        query=q,
        results=[
            SearchResult(
                marker=position,
                label=item.citation_label,
                source=item.chunk.source,
                page=item.chunk.page,
                text=item.chunk.text[:600],
                fused_score=round(item.score, 6),
                rerank_score=item.rerank_score,
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
            )
            for position, item in enumerate(retrieved, start=1)
        ],
    )


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    pipeline = _pipeline(request)
    history = [(turn.role, turn.content) for turn in body.history]
    answer = await pipeline.answer(body.question, history)
    return ChatResponse(
        request_id=answer.request_id,
        answer=answer.text,
        grounded=answer.grounded,
        query_used=answer.query_used,
        grounding_coverage=answer.grounding_coverage,
        refusal_reason=answer.refusal_reason,
        citations=[asdict(c) for c in answer.citations],
        prompt_tokens=answer.usage.prompt_tokens,
        completion_tokens=answer.usage.completion_tokens,
        cost_usd=answer.cost_usd,
        timings_ms=answer.timings_ms,
    )


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    pipeline = _pipeline(request)
    history = [(turn.role, turn.content) for turn in body.history]

    async def event_source():
        async for event in pipeline.stream_answer(body.question, history):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
