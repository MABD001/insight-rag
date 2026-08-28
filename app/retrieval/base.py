"""Storage port. `memory` and `pgvector` are the two adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class StoredChunk:
    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    embed_text: str
    heading_path: list[str] = field(default_factory=list)
    page: int | None = None
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: StoredChunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    @property
    def citation_label(self) -> str:
        parts = [self.chunk.source]
        if self.chunk.page is not None:
            parts.append(f"p.{self.chunk.page}")
        if self.chunk.heading_path:
            parts.append(self.chunk.heading_path[-1])
        return " · ".join(parts)


class VectorStore(Protocol):
    async def upsert(
        self, chunks: Sequence[StoredChunk], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    async def search(
        self, embedding: Sequence[float], *, top_k: int
    ) -> list[tuple[StoredChunk, float]]: ...

    async def all_chunks(self) -> list[StoredChunk]: ...

    async def delete_document(self, document_id: str) -> int: ...

    async def document_hashes(self) -> dict[str, str]: ...

    async def record_document(
        self, document_id: str, source: str, content_hash: str, chunk_count: int
    ) -> None: ...

    async def stats(self) -> dict[str, int]: ...
