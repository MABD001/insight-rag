"""OpenAI-compatible provider.

Speaks the /v1 wire format, so it also works against Azure OpenAI gateways,
Together, Groq, vLLM and anything else that mirrors the OpenAI API — only
`openai_base_url` changes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.providers.base import ChatMessage, ChatResult, Usage

_RETRY = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=0.5, min=0.5, max=8),
    "retry": retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    "reraise": True,
}


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self.dimensions = dimensions
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    @retry(**_RETRY)
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers,
                json={"model": self._model, "input": list(texts)},
            )
            response.raise_for_status()
            payload = response.json()
        # The API does not guarantee ordering, so sort by the echoed index.
        rows = sorted(payload["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]


class OpenAIChatProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    def _body(self, messages: Sequence[ChatMessage], temperature: float) -> dict:
        return {
            "model": self._model,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

    @retry(**_RETRY)
    async def complete(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> ChatResult:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=self._body(messages, temperature),
            )
            response.raise_for_status()
            payload = response.json()
        usage_payload = payload.get("usage") or {}
        return ChatResult(
            text=payload["choices"][0]["message"]["content"],
            usage=Usage(
                prompt_tokens=usage_payload.get("prompt_tokens", 0),
                completion_tokens=usage_payload.get("completion_tokens", 0),
            ),
        )

    async def stream(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        body = self._body(messages, temperature) | {"stream": True}
        async with httpx.AsyncClient(timeout=self._timeout) as client, client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if content := delta.get("content"):
                    yield content
