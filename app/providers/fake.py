"""Deterministic offline providers.

These exist so the full pipeline — ingestion, hybrid retrieval, reranking,
grounding checks, streaming and the eval harness — can be exercised in CI
with no API key, no network and no GPU, and produce byte-identical results
on every run. Retrieval quality tests are therefore measuring *retrieval*,
not model variance.

The embedder is a hashed bag-of-words projection: same text always maps to
the same unit vector, and texts sharing vocabulary land near each other, so
cosine similarity stays a meaningful signal.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence

from app.providers.base import ChatMessage, ChatResult, Usage
from app.rag.prompts import CONTEXT_MARKER

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbeddingProvider:
    """Hashing vectoriser with L2 normalisation."""

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = _tokenize(text)
        if not tokens:
            # A zero vector would make cosine similarity undefined; anchor
            # empty input to a fixed basis vector instead.
            vec[0] = 1.0
            return vec
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class FakeChatProvider:
    """Extractive stand-in for a real LLM.

    It does not attempt to sound like a language model. It answers from the
    retrieved context by selecting the sentences that overlap most with the
    question, then appends the citation markers the real prompt asks for.
    That keeps citation plumbing, streaming and grounding logic under test
    without pretending to evaluate generation quality.
    """

    name = "fake"

    async def complete(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> ChatResult:
        question = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        context = next((m.content for m in messages if m.role == "system"), "")
        answer = self._extract(question, context)
        usage = Usage(
            prompt_tokens=sum(len(_tokenize(m.content)) for m in messages),
            completion_tokens=len(_tokenize(answer)),
        )
        return ChatResult(text=answer, usage=usage)

    async def stream(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        result = await self.complete(messages, temperature=temperature)
        for word in result.text.split(" "):
            yield word + " "

    @staticmethod
    def _extract(question: str, context: str) -> str:
        # The instruction block contains bracket markers of its own ("e.g. [2]"),
        # so extraction must start after the context sentinel, not at the first
        # bracket in the prompt.
        _, _, context = context.partition(CONTEXT_MARKER)
        q_terms = set(_tokenize(question))
        # Context arrives as "[1] chunk text" blocks built by the prompt
        # builder; recover the citation index alongside each sentence.
        blocks = re.findall(r"\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", context, re.S)
        scored: list[tuple[float, str, str]] = []
        for marker, body in blocks:
            # The prompt builder prefixes each passage with a parenthesised
            # location line, e.g. "(policy.md - Refunds)". It is metadata for
            # the model, not answer text, so it must not be extractable.
            body = re.sub(r"^\([^)]*\)\s*", "", body.strip(), flags=re.S)
            for sentence in re.split(r"(?<=[.!?])\s+", body.strip()):
                terms = set(_tokenize(sentence))
                if not terms:
                    continue
                overlap = len(q_terms & terms) / math.sqrt(len(terms))
                if overlap > 0:
                    scored.append((overlap, sentence.strip(), marker))
        if not scored:
            return "I could not find an answer to that in the provided documents."
        scored.sort(key=lambda item: item[0], reverse=True)
        # Up to three sentences, but only those scoring close to the best one.
        # Without the relative floor a crude extractor pads every answer with
        # the next-best sentence in the corpus, however irrelevant.
        best = scored[0][0]
        chosen = [row for row in scored[:3] if row[0] >= 0.45 * best]
        return " ".join(f"{text} [{marker}]" for _, text, marker in chosen)
