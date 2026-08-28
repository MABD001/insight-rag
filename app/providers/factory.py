"""Builds the configured providers. The only place backends are named."""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.providers.base import ChatProvider, EmbeddingProvider


def _require_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Either export it, or run with "
            "LLM_PROVIDER=fake / EMBEDDING_PROVIDER=fake for the offline stack."
        )
    return settings.openai_api_key


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        from app.providers.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=_require_key(settings),
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
    if settings.embedding_provider == "ollama":
        from app.providers.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            model=settings.ollama_embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout_seconds,
        )
    from app.providers.fake import FakeEmbeddingProvider

    return FakeEmbeddingProvider(dimensions=settings.embedding_dimensions)


@lru_cache
def get_chat_provider() -> ChatProvider:
    settings = get_settings()
    if settings.llm_provider == "openai":
        from app.providers.openai import OpenAIChatProvider

        return OpenAIChatProvider(
            api_key=_require_key(settings),
            model=settings.openai_chat_model,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
    if settings.llm_provider == "ollama":
        from app.providers.ollama import OllamaChatProvider

        return OllamaChatProvider(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout_seconds,
        )
    from app.providers.fake import FakeChatProvider

    return FakeChatProvider()
