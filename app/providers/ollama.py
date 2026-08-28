"""Local Ollama provider — lets the whole stack run offline on a laptop."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from app.providers.base import ChatMessage, ChatResult, Usage


class OllamaEmbeddingProvider:
    def __init__(
        self,
        model: str,
        dimensions: int,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.dimensions = dimensions
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": list(texts)},
            )
            response.raise_for_status()
            return response.json()["embeddings"]


class OllamaChatProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _body(self, messages: Sequence[ChatMessage], temperature: float, stream: bool) -> dict:
        return {
            "model": self._model,
            "stream": stream,
            "options": {"temperature": temperature},
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

    async def complete(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> ChatResult:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=self._body(messages, temperature, stream=False),
            )
            response.raise_for_status()
            payload = response.json()
        return ChatResult(
            text=payload["message"]["content"],
            usage=Usage(
                prompt_tokens=payload.get("prompt_eval_count", 0),
                completion_tokens=payload.get("eval_count", 0),
            ),
        )

    async def stream(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as client, client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=self._body(messages, temperature, stream=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("done"):
                    break
                if content := payload.get("message", {}).get("content"):
                    yield content
