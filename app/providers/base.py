"""Provider protocols shared by every backend."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class Usage:
    """Token accounting for a single generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, input_per_million: float, output_per_million: float) -> float:
        return round(
            self.prompt_tokens / 1_000_000 * input_per_million
            + self.completion_tokens / 1_000_000 * output_per_million,
            6,
        )


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatResult:
    text: str
    usage: Usage = field(default_factory=Usage)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors. Must be deterministic for a given input."""

    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class ChatProvider(Protocol):
    """Generates an answer, either buffered or streamed token by token."""

    name: str

    async def complete(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> ChatResult: ...

    def stream(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> AsyncIterator[str]: ...
