"""In-memory vector store.

The default adapter. Keeps a clean checkout runnable with one command and
makes the test suite fast and hermetic. Swapping to pgvector is a config
change, not a code change — that is the point of the `VectorStore` port.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np

from app.retrieval.base import StoredChunk


class MemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, StoredChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._documents: dict[str, dict[str, str | int]] = {}
        self._lock = asyncio.Lock()

    async def upsert(
        self, chunks: Sequence[StoredChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        async with self._lock:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                vector = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    vector = vector / norm
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = vector

    async def search(
        self, embedding: Sequence[float], *, top_k: int
    ) -> list[tuple[StoredChunk, float]]:
        async with self._lock:
            if not self._vectors:
                return []
            ids = list(self._vectors.keys())
            matrix = np.stack([self._vectors[i] for i in ids])
            query = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm
            # Both sides are unit-normalised, so the dot product is cosine.
            scores = matrix @ query
            order = np.argsort(-scores)[:top_k]
            return [(self._chunks[ids[i]], float(scores[i])) for i in order]

    async def all_chunks(self) -> list[StoredChunk]:
        async with self._lock:
            return list(self._chunks.values())

    async def delete_document(self, document_id: str) -> int:
        async with self._lock:
            doomed = [
                cid for cid, chunk in self._chunks.items() if chunk.document_id == document_id
            ]
            for chunk_id in doomed:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk_id, None)
            self._documents.pop(document_id, None)
            return len(doomed)

    async def document_hashes(self) -> dict[str, str]:
        async with self._lock:
            return {
                doc_id: str(row["content_hash"]) for doc_id, row in self._documents.items()
            }

    async def record_document(
        self, document_id: str, source: str, content_hash: str, chunk_count: int
    ) -> None:
        async with self._lock:
            self._documents[document_id] = {
                "source": source,
                "content_hash": content_hash,
                "chunk_count": chunk_count,
            }

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {"documents": len(self._documents), "chunks": len(self._chunks)}
