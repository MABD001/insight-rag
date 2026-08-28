"""Shared fixtures. Everything here runs offline and deterministically."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.ingest.loaders import LoadedDocument, load_bytes
from app.providers.fake import FakeChatProvider, FakeEmbeddingProvider
from app.rag.pipeline import RagPipeline
from app.retrieval.memory_store import MemoryVectorStore

SAMPLE_MARKDOWN = """\
# Acme Support Handbook

## Refunds

Approved refunds are returned to the original payment method within 5 business
days. Original delivery charges are refunded only when the return is our fault.

## Shipping

Standard delivery takes 3 to 5 working days. Express delivery is next working
day if the order is placed before 14:00.

## Diagnostics

Error code AX-7781 indicates a failed temperature sensor on the mixer range.
Return the unit for service rather than resetting it.
"""


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store="memory",
        embedding_dimensions=256,
        chunk_size=400,
        chunk_overlap=80,
        dense_top_k=10,
        sparse_top_k=10,
        rerank_top_n=4,
    )


@pytest.fixture
def document() -> LoadedDocument:
    return load_bytes(SAMPLE_MARKDOWN.encode(), filename="handbook.md")


@pytest.fixture
async def pipeline(settings: Settings, document: LoadedDocument) -> RagPipeline:
    pipe = RagPipeline(
        settings=settings,
        store=MemoryVectorStore(),
        embedder=FakeEmbeddingProvider(dimensions=settings.embedding_dimensions),
        chat=FakeChatProvider(),
    )
    await pipe.ingest(document)
    return pipe
