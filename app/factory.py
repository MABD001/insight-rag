"""Composition root — builds the store and pipeline from settings."""

from __future__ import annotations

from app.config import Settings
from app.providers import get_chat_provider, get_embedding_provider
from app.rag.pipeline import RagPipeline
from app.retrieval.base import VectorStore


def build_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "pgvector":
        from sqlalchemy import create_engine

        from app.retrieval.pgvector_store import PgVectorStore

        engine = create_engine(settings.database_url, pool_pre_ping=True)
        store = PgVectorStore(engine, dimensions=settings.embedding_dimensions)
        store.create_schema()
        return store

    from app.retrieval.memory_store import MemoryVectorStore

    return MemoryVectorStore()


def build_pipeline(settings: Settings) -> RagPipeline:
    return RagPipeline(
        settings=settings,
        store=build_store(settings),
        embedder=get_embedding_provider(),
        chat=get_chat_provider(),
    )
