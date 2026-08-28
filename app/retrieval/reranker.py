"""Reranking stage.

Fusion optimises recall — it gathers everything plausibly relevant. Reranking
optimises precision over that smaller candidate set, which is what keeps the
prompt short and the answer grounded.

Two implementations share one interface:

* `CrossEncoderReranker` loads a sentence-transformers cross-encoder when the
  optional extra is installed. Best quality, ~400MB of weights.
* `LexicalOverlapReranker` is the default. It scores query/chunk term overlap
  with IDF weighting plus a proximity bonus for adjacent query terms. It is
  dependency-free, deterministic and fast enough to run per request, which
  keeps CI honest and the container small.

The scores are also what the grounding guardrail thresholds against, so the
default path must produce a calibrated, comparable number — not just an order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.retrieval.base import RetrievedChunk
from app.retrieval.bm25 import tokenize


class LexicalOverlapReranker:
    name = "lexical-overlap"

    def score(self, query: str, text: str) -> float:
        query_terms = tokenize(query)
        chunk_terms = tokenize(text)
        if not query_terms or not chunk_terms:
            return 0.0

        chunk_set = set(chunk_terms)
        matched = [t for t in query_terms if t in chunk_set]
        if not matched:
            return 0.0

        coverage = len(set(matched)) / len(set(query_terms))

        # Rarer query terms matching is stronger evidence than common ones.
        weights = sum(1.0 / (1.0 + math.log(1 + chunk_terms.count(t))) for t in set(matched))
        weight_score = weights / len(set(query_terms))

        # Proximity: reward chunks where matched query terms appear close
        # together, which separates a real answer from an incidental mention.
        positions = [i for i, t in enumerate(chunk_terms) if t in set(matched)]
        if len(positions) > 1:
            span = positions[-1] - positions[0] + 1
            proximity = len(positions) / span
        else:
            proximity = 1.0 / (1.0 + math.log(1 + len(chunk_terms)))

        # Length normalisation stops a 900-char chunk beating a tight answer
        # purely by containing more words.
        brevity = 1.0 / (1.0 + math.log(1 + len(chunk_terms) / 50))

        return round(0.5 * coverage + 0.2 * weight_score + 0.2 * proximity + 0.1 * brevity, 6)

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        for candidate in candidates:
            candidate.rerank_score = self.score(query, candidate.chunk.embed_text)
        ordered = sorted(
            candidates,
            key=lambda c: (-(c.rerank_score or 0.0), c.chunk.chunk_id),
        )
        return ordered[:top_n]


class CrossEncoderReranker:
    """sentence-transformers cross-encoder, with a graceful fallback.

    Instantiating this when the optional dependency is absent does not raise —
    it transparently degrades to `LexicalOverlapReranker`, so a deployment
    that forgets the extra keeps working with slightly lower precision rather
    than failing at request time.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._fallback = LexicalOverlapReranker()
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        except Exception:  # noqa: BLE001 - optional dependency, any failure degrades
            self._model = None

    @property
    def active_backend(self) -> str:
        return self.name if self._model is not None else self._fallback.name

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        if self._model is None:
            return self._fallback.rerank(query, candidates, top_n=top_n)

        pairs = [(query, c.chunk.embed_text) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores, strict=True):
            # Cross-encoder logits are unbounded; squash to 0..1 so the
            # grounding threshold means the same thing for both backends.
            candidate.rerank_score = float(1.0 / (1.0 + math.exp(-float(score))))
        ordered = sorted(
            candidates, key=lambda c: (-(c.rerank_score or 0.0), c.chunk.chunk_id)
        )
        return ordered[:top_n]
