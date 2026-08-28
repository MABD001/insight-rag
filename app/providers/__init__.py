"""Pluggable LLM and embedding backends.

The application depends only on the `EmbeddingProvider` / `ChatProvider`
protocols in `base.py`. That is what lets the same pipeline run against
OpenAI in production, Ollama on a laptop, and a deterministic fake in CI.
"""

from app.providers.base import ChatProvider, EmbeddingProvider, Usage
from app.providers.factory import get_chat_provider, get_embedding_provider

__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "Usage",
    "get_chat_provider",
    "get_embedding_provider",
]
