"""The RAG pipeline.

Request flow:

    question
      -> (optional) query rewrite against conversation history
      -> dense search  ──┐
      -> BM25 search   ──┴─> reciprocal rank fusion
      -> rerank (precision pass)
      -> grounding guardrail  ── below threshold ─> refuse
      -> prompt assembly
      -> generate (buffered or streamed)
      -> attach citations + usage/cost/latency

Every stage records timing, so `/chat` returns a per-stage latency breakdown.
That is the difference between "the bot is slow" and "reranking is eating
600ms", and it is the first thing anyone debugging a RAG system needs.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import structlog

from app.config import Settings
from app.ingest.chunker import chunk_document
from app.ingest.loaders import LoadedDocument
from app.providers.base import ChatMessage, ChatProvider, EmbeddingProvider, Usage
from app.rag.grounding import GroundingChecker, GroundingVerdict
from app.rag.prompts import build_rewrite_prompt, build_system_prompt
from app.retrieval.base import RetrievedChunk, StoredChunk, VectorStore
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker, LexicalOverlapReranker

logger = structlog.get_logger(__name__)

REFUSAL = "I could not find an answer to that in the provided documents."
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class Citation:
    marker: int
    source: str
    title: str
    label: str
    page: int | None
    heading_path: list[str]
    snippet: str
    score: float
    chunk_id: str


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation]
    grounded: bool
    query_used: str
    grounding_coverage: float = 0.0
    refusal_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    request_id: str = ""


@dataclass(slots=True)
class IngestResult:
    document_id: str
    source: str
    chunks: int
    skipped: bool


class RagPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        store: VectorStore,
        embedder: EmbeddingProvider,
        chat: ChatProvider,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.chat = chat
        self.bm25 = BM25Index()
        self.reranker = (
            CrossEncoderReranker() if settings.rerank_enabled else LexicalOverlapReranker()
        )
        self.grounding = GroundingChecker(
            min_coverage=settings.min_grounding_coverage,
            min_score=settings.min_grounding_score,
        )
        self._bm25_ready = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    async def ingest(self, document: LoadedDocument, *, force: bool = False) -> IngestResult:
        document_id = f"doc_{document.content_hash[:16]}"

        if not force:
            existing = await self.store.document_hashes()
            if existing.get(document_id) == document.content_hash:
                # Identical content already indexed — skip the embedding spend.
                logger.info("ingest.skipped", source=document.source, document_id=document_id)
                return IngestResult(document_id, document.source, 0, skipped=True)

        await self.store.delete_document(document_id)

        chunks = chunk_document(
            document,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        stored = [
            StoredChunk(
                chunk_id=f"{document_id}:{chunk.index}",
                document_id=document_id,
                source=document.source,
                title=document.title,
                text=chunk.text,
                embed_text=chunk.embed_text,
                heading_path=chunk.heading_path,
                page=chunk.page,
                ordinal=chunk.index,
                metadata={"media_type": document.media_type},
            )
            for chunk in chunks
        ]

        if stored:
            embeddings = await self.embedder.embed([c.embed_text for c in stored])
            await self.store.upsert(stored, embeddings)

        await self.store.record_document(
            document_id, document.source, document.content_hash, len(stored)
        )
        await self.refresh_sparse_index()

        logger.info("ingest.completed", source=document.source, chunks=len(stored))
        return IngestResult(document_id, document.source, len(stored), skipped=False)

    async def refresh_sparse_index(self) -> None:
        """Rebuild every corpus-derived index. Called after any ingest or delete."""
        chunks = await self.store.all_chunks()
        self.bm25.build((c.chunk_id, c.embed_text) for c in chunks)
        # The grounding check is corpus-aware, so it has to be refitted too:
        # term specificity is meaningless against a stale vocabulary.
        self.grounding.fit(chunks)
        self._bm25_ready = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        if not self._bm25_ready:
            await self.refresh_sparse_index()

        embedding = (await self.embedder.embed([query]))[0]
        dense = await self.store.search(embedding, top_k=self.settings.dense_top_k)
        dense_ids = [chunk.chunk_id for chunk, _ in dense]
        sparse_ids = list(self.bm25.ranked_ids(query, top_k=self.settings.sparse_top_k))

        by_id: dict[str, StoredChunk] = {chunk.chunk_id: chunk for chunk, _ in dense}
        missing = [cid for cid in sparse_ids if cid not in by_id]
        if missing:
            # BM25 can surface chunks the dense pass never returned; fetch them.
            for chunk in await self.store.all_chunks():
                if chunk.chunk_id in missing:
                    by_id[chunk.chunk_id] = chunk

        fused = reciprocal_rank_fusion(
            {"dense": dense_ids, "sparse": sparse_ids}, k=self.settings.rrf_k
        )

        candidates = [
            RetrievedChunk(
                chunk=by_id[chunk_id],
                score=score,
                dense_rank=ranks.get("dense"),
                sparse_rank=ranks.get("sparse"),
            )
            for chunk_id, score, ranks in fused
            if chunk_id in by_id
        ]

        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, top_n=self.settings.rerank_top_n)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def _prepare(
        self, question: str, history: Sequence[tuple[str, str]]
    ) -> tuple[str, list[RetrievedChunk], dict[str, float]]:
        timings: dict[str, float] = {}

        started = time.perf_counter()
        query = await self._rewrite_query(question, history)
        timings["rewrite"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        chunks = await self.retrieve(query)
        timings["retrieve"] = round((time.perf_counter() - started) * 1000, 2)

        return query, chunks, timings

    async def _rewrite_query(
        self, question: str, history: Sequence[tuple[str, str]]
    ) -> str:
        if not history:
            return question
        result = await self.chat.complete(
            [ChatMessage(role="user", content=build_rewrite_prompt(question, history))],
            temperature=0.0,
        )
        rewritten = result.text.strip().strip('"')
        # A rewrite that collapses to nothing or explodes in length is a
        # model failure; fall back to the original rather than degrading search.
        if not rewritten or len(rewritten) > 4 * max(len(question), 80):
            return question
        return rewritten

    def _check_grounding(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> GroundingVerdict:
        return self.grounding.check(question, chunks)

    def _messages(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=build_system_prompt(
                    chunks, max_chars=self.settings.max_context_chars
                ),
            ),
            ChatMessage(role="user", content=question),
        ]

    def _citations(
        self, answer_text: str, chunks: Sequence[RetrievedChunk]
    ) -> list[Citation]:
        """Return only the passages the answer actually cited.

        Listing every retrieved chunk as a 'source' is the most common way RAG
        UIs mislead users: it implies the model used material it ignored.
        """
        markers = {int(m) for m in _CITATION_RE.findall(answer_text)}
        citations: list[Citation] = []
        for position, retrieved in enumerate(chunks, start=1):
            if position not in markers:
                continue
            chunk = retrieved.chunk
            citations.append(
                Citation(
                    marker=position,
                    source=chunk.source,
                    title=chunk.title,
                    label=retrieved.citation_label,
                    page=chunk.page,
                    heading_path=chunk.heading_path,
                    snippet=chunk.text[:400],
                    score=round(retrieved.rerank_score or retrieved.score, 6),
                    chunk_id=chunk.chunk_id,
                )
            )
        return citations

    async def answer(
        self, question: str, history: Sequence[tuple[str, str]] = ()
    ) -> Answer:
        request_id = uuid.uuid4().hex[:12]
        query, chunks, timings = await self._prepare(question, history)

        verdict = self._check_grounding(query, chunks)
        if not verdict.grounded:
            logger.info(
                "answer.refused",
                request_id=request_id,
                query=query,
                coverage=verdict.coverage,
                reason=verdict.reason,
                missing_terms=verdict.missing_terms[:8],
            )
            return Answer(
                text=REFUSAL,
                citations=[],
                grounded=False,
                query_used=query,
                grounding_coverage=verdict.coverage,
                refusal_reason=verdict.reason,
                timings_ms=timings,
                request_id=request_id,
            )

        started = time.perf_counter()
        result = await self.chat.complete(
            self._messages(question, chunks), temperature=self.settings.temperature
        )
        timings["generate"] = round((time.perf_counter() - started) * 1000, 2)
        timings["total"] = round(sum(timings.values()), 2)

        cost = result.usage.cost_usd(
            self.settings.input_token_cost_per_million,
            self.settings.output_token_cost_per_million,
        )
        logger.info(
            "answer.completed",
            request_id=request_id,
            tokens=result.usage.total_tokens,
            cost_usd=cost,
            latency_ms=timings["total"],
        )
        return Answer(
            text=result.text,
            citations=self._citations(result.text, chunks),
            grounded=True,
            query_used=query,
            grounding_coverage=verdict.coverage,
            usage=result.usage,
            cost_usd=cost,
            timings_ms=timings,
            request_id=request_id,
        )

    async def stream_answer(
        self, question: str, history: Sequence[tuple[str, str]] = ()
    ) -> AsyncIterator[dict]:
        """Yield SSE-shaped events: meta, token*, done — or meta, refusal."""
        request_id = uuid.uuid4().hex[:12]
        query, chunks, timings = await self._prepare(question, history)

        verdict = self._check_grounding(query, chunks)

        yield {
            "type": "meta",
            "request_id": request_id,
            "query_used": query,
            "grounding_coverage": verdict.coverage,
            "retrieved": [
                {
                    "marker": position,
                    "label": retrieved.citation_label,
                    "score": round(retrieved.rerank_score or retrieved.score, 6),
                }
                for position, retrieved in enumerate(chunks, start=1)
            ],
        }

        if not verdict.grounded:
            yield {"type": "token", "text": REFUSAL}
            yield {
                "type": "done",
                "grounded": False,
                "citations": [],
                "refusal_reason": verdict.reason,
                "timings_ms": timings,
                "request_id": request_id,
            }
            return

        started = time.perf_counter()
        buffer: list[str] = []
        async for token in self.chat.stream(
            self._messages(question, chunks), temperature=self.settings.temperature
        ):
            buffer.append(token)
            yield {"type": "token", "text": token}

        timings["generate"] = round((time.perf_counter() - started) * 1000, 2)
        timings["total"] = round(sum(timings.values()), 2)
        text = "".join(buffer)

        yield {
            "type": "done",
            "grounded": True,
            "citations": [
                {
                    "marker": c.marker,
                    "source": c.source,
                    "label": c.label,
                    "page": c.page,
                    "snippet": c.snippet,
                    "score": c.score,
                }
                for c in self._citations(text, chunks)
            ],
            "timings_ms": timings,
            "request_id": request_id,
        }
