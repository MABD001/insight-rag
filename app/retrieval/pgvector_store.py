"""PostgreSQL + pgvector adapter.

Chosen over a dedicated vector database because most teams already run
Postgres: one backup story, one connection pool, transactional consistency
between chunks and their parent document rows, and metadata filtering in the
same query as the vector search.

Blocking psycopg calls are pushed to a worker thread so they never stall the
event loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from sqlalchemy import Engine, text

from app.retrieval.base import StoredChunk

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source       TEXT NOT NULL,
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    embed_text   TEXT NOT NULL,
    heading_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    page         INTEGER,
    ordinal      INTEGER NOT NULL DEFAULT 0,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding    VECTOR(%(dim)s)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- IVFFlat trades a little recall for a large speed win. `lists` should be
-- roughly sqrt(row_count); 100 suits corpora in the 10k-100k chunk range.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


class PgVectorStore:
    def __init__(self, engine: Engine, dimensions: int) -> None:
        self._engine = engine
        self._dimensions = dimensions

    def create_schema(self) -> None:
        with self._engine.begin() as connection:
            connection.exec_driver_sql(SCHEMA % {"dim": self._dimensions})

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _to_vector(embedding: Sequence[float]) -> str:
        return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"

    def _row_to_chunk(self, row) -> StoredChunk:  # noqa: ANN001 - sqlalchemy Row
        return StoredChunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            source=row.source,
            title=row.title,
            text=row.text,
            embed_text=row.embed_text,
            heading_path=list(row.heading_path or []),
            page=row.page,
            ordinal=row.ordinal,
            metadata=dict(row.metadata or {}),
        )

    # -- VectorStore -----------------------------------------------------
    async def upsert(
        self, chunks: Sequence[StoredChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        def _run() -> None:
            with self._engine.begin() as connection:
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    connection.execute(
                        text(
                            """
                            INSERT INTO chunks (chunk_id, document_id, source, title, text,
                                                embed_text, heading_path, page, ordinal,
                                                metadata, embedding)
                            VALUES (:chunk_id, :document_id, :source, :title, :text,
                                    :embed_text, CAST(:heading_path AS jsonb), :page,
                                    :ordinal, CAST(:metadata AS jsonb),
                                    CAST(:embedding AS vector))
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                text = EXCLUDED.text,
                                embed_text = EXCLUDED.embed_text,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata
                            """
                        ),
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "source": chunk.source,
                            "title": chunk.title,
                            "text": chunk.text,
                            "embed_text": chunk.embed_text,
                            "heading_path": json.dumps(chunk.heading_path),
                            "page": chunk.page,
                            "ordinal": chunk.ordinal,
                            "metadata": json.dumps(chunk.metadata),
                            "embedding": self._to_vector(embedding),
                        },
                    )

        await asyncio.to_thread(_run)

    async def search(
        self, embedding: Sequence[float], *, top_k: int
    ) -> list[tuple[StoredChunk, float]]:
        def _run() -> list[tuple[StoredChunk, float]]:
            with self._engine.connect() as connection:
                rows = connection.execute(
                    text(
                        """
                        SELECT *, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
                        FROM chunks
                        ORDER BY embedding <=> CAST(:embedding AS vector)
                        LIMIT :top_k
                        """
                    ),
                    {"embedding": self._to_vector(embedding), "top_k": top_k},
                ).fetchall()
            return [(self._row_to_chunk(row), float(row.score)) for row in rows]

        return await asyncio.to_thread(_run)

    async def all_chunks(self) -> list[StoredChunk]:
        def _run() -> list[StoredChunk]:
            with self._engine.connect() as connection:
                rows = connection.execute(text("SELECT * FROM chunks")).fetchall()
            return [self._row_to_chunk(row) for row in rows]

        return await asyncio.to_thread(_run)

    async def delete_document(self, document_id: str) -> int:
        def _run() -> int:
            with self._engine.begin() as connection:
                result = connection.execute(
                    text("DELETE FROM documents WHERE document_id = :document_id"),
                    {"document_id": document_id},
                )
                return result.rowcount or 0

        return await asyncio.to_thread(_run)

    async def document_hashes(self) -> dict[str, str]:
        def _run() -> dict[str, str]:
            with self._engine.connect() as connection:
                rows = connection.execute(
                    text("SELECT document_id, content_hash FROM documents")
                ).fetchall()
            return {row.document_id: row.content_hash for row in rows}

        return await asyncio.to_thread(_run)

    async def record_document(
        self, document_id: str, source: str, content_hash: str, chunk_count: int
    ) -> None:
        def _run() -> None:
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO documents (document_id, source, content_hash, chunk_count)
                        VALUES (:document_id, :source, :content_hash, :chunk_count)
                        ON CONFLICT (document_id) DO UPDATE SET
                            content_hash = EXCLUDED.content_hash,
                            chunk_count  = EXCLUDED.chunk_count,
                            ingested_at  = now()
                        """
                    ),
                    {
                        "document_id": document_id,
                        "source": source,
                        "content_hash": content_hash,
                        "chunk_count": chunk_count,
                    },
                )

        await asyncio.to_thread(_run)

    async def stats(self) -> dict[str, int]:
        def _run() -> dict[str, int]:
            with self._engine.connect() as connection:
                documents = connection.execute(
                    text("SELECT count(*) FROM documents")
                ).scalar_one()
                chunks = connection.execute(text("SELECT count(*) FROM chunks")).scalar_one()
            return {"documents": int(documents), "chunks": int(chunks)}

        return await asyncio.to_thread(_run)
