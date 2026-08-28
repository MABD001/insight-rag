"""Runtime configuration.

Every knob that changes retrieval or generation behaviour lives here so that an
evaluation run can report the exact configuration that produced a score.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Providers -------------------------------------------------------
    # "fake" keeps the whole system runnable (and CI green) with no API key
    # and no network. "openai" and "ollama" are the real backends.
    llm_provider: Literal["fake", "openai", "ollama"] = "fake"
    embedding_provider: Literal["fake", "openai", "ollama"] = "fake"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    embedding_dimensions: int = 1536

    # --- Storage ---------------------------------------------------------
    # "memory" is the default so `uvicorn app.main:app` works on a clean
    # checkout. docker compose sets this to "pgvector".
    vector_store: Literal["memory", "pgvector"] = "memory"
    database_url: str = "postgresql+psycopg://insight:insight@localhost:5432/insight"

    # --- Startup seeding -------------------------------------------------
    # With the in-memory store the corpus dies with the process, so a fresh
    # `make run` would otherwise serve an empty index. Seeding only happens
    # when the store reports zero chunks, so a populated pgvector deployment
    # is never touched.
    seed_on_startup: bool = True
    seed_path: str = "sample_docs"

    # --- Chunking --------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 150

    # --- Retrieval -------------------------------------------------------
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 6
    rerank_enabled: bool = True

    # --- Grounding guardrail ---------------------------------------------
    # See app/rag/grounding.py for why a bare relevance threshold does not
    # work. Coverage is the primary signal; the score floor is a cheap
    # secondary guard against a corpus with no lexical overlap at all.
    min_grounding_coverage: float = 0.05
    min_grounding_score: float = 0.05

    # --- Generation ------------------------------------------------------
    max_context_chars: int = 8000
    temperature: float = 0.0
    request_timeout_seconds: float = 60.0

    # --- Cost accounting -------------------------------------------------
    # USD per 1M tokens, used to report per-request cost in the API response.
    input_token_cost_per_million: float = 0.15
    output_token_cost_per_million: float = 0.60


@lru_cache
def get_settings() -> Settings:
    return Settings()
