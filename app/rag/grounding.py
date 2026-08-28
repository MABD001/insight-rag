"""Grounding guardrail — deciding when to refuse.

The obvious implementation is a threshold on the retriever's relevance score.
It does not work, and it is worth explaining why, because the failure is not
obvious until you measure it.

Retrieval always returns *something*. Ask "Who is the CTO of Northwind Supply?"
of a corpus where every document says "Northwind Supply" in its heading, and
the top chunk scores well — the query genuinely does overlap the corpus. Ask
"Who pays return shipping if I changed my mind?" and the correct chunk scores
lower, because the phrasing shares fewer words with it. Measured on this
repo's golden set, the two classes overlap outright:

    answerable   best rerank score: 0.29 .. 0.66
    unanswerable best rerank score: 0.42 .. 0.47

No threshold separates those. A system built on one either refuses good
questions or answers unanswerable ones with confident nonsense.

This checker asks a different question: **are the distinctive terms of the
query actually present in the retrieved context?**

- Each query term is weighted by its corpus IDF, so "refunds" counts for far
  more than "supply", which appears in every document.
- Terms the corpus has never seen ("bitcoin", "revenue", "officer") get
  maximum weight and are always misses — an unknown term is the strongest
  available evidence that the corpus cannot answer the question.
- Interrogatives, modals and pronouns are excluded; they never discriminate
  between documents.
- Light suffix stemming stops "refunds"/"refund" and "changed"/"change" from
  being treated as unrelated.

On the same golden set that formulation separates cleanly:

    answerable   weighted coverage: 0.09 .. 1.00
    unanswerable weighted coverage: 0.00, 0.00, 0.00

Unanswerable questions land on exactly zero because *none* of their content
terms appear anywhere in the retrieved context. That is a principled signal
rather than a tuned constant, which is why it survives a change of corpus.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.retrieval.base import RetrievedChunk, StoredChunk
from app.retrieval.bm25 import tokenize

# Words that are common in questions and carry no discriminating power. Kept
# separate from the BM25 stopword list, which is deliberately smaller because
# ranking and grounding need different things from tokenisation.
QUERY_STOPWORDS = frozenset(
    """
    how what when where why who which whose whom do does did done can could will would
    shall should may might must is are was were be been being am i me my mine we us our
    ours you your yours he she they them their there here it its to for of in on at by with
    from about into over under than then that this these those a an the and or but if so as
    not no yes long much many more most less least take takes taken taking get gets got
    getting need needs needed want wants use uses used using apply applies happen happens
    mean means please tell show give
    """.split()
)

_SUFFIXES = ("ing", "ies", "ied", "es", "ed", "ly", "s")


def stem(token: str) -> str:
    """Very light suffix stripping.

    Not linguistically correct, and not trying to be. It exists to stop
    "refunds"/"refund" and "changed"/"change" scoring as unrelated terms. A
    real stemmer (Snowball) is a one-line swap if a corpus needs it; for
    grounding, over-stemming is worse than under-stemming because it
    manufactures false matches.
    """
    lowered = token.lower()
    if len(lowered) <= 4:
        return lowered
    for suffix in _SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 3:
            return lowered[: -len(suffix)]
    return lowered


def content_terms(text: str) -> set[str]:
    return {stem(t) for t in tokenize(text) if t not in QUERY_STOPWORDS}


@dataclass(slots=True)
class GroundingVerdict:
    grounded: bool
    coverage: float
    best_score: float
    matched_terms: list[str]
    missing_terms: list[str]

    @property
    def reason(self) -> str:
        if self.grounded:
            return "grounded"
        if self.coverage == 0.0:
            return "no query terms present in retrieved context"
        return f"weighted term coverage {self.coverage:.2f} below threshold"


class GroundingChecker:
    """Corpus-aware grounding check. Rebuilt whenever the corpus changes."""

    def __init__(self, *, min_coverage: float = 0.05, min_score: float = 0.05) -> None:
        self.min_coverage = min_coverage
        self.min_score = min_score
        self._document_frequency: dict[str, int] = {}
        self._corpus_size = 0

    def fit(self, chunks: Sequence[StoredChunk]) -> None:
        self._document_frequency = {}
        for chunk in chunks:
            for term in content_terms(chunk.embed_text):
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1
        self._corpus_size = len(chunks)

    def specificity(self, term: str) -> float:
        """0 for a term in every chunk, 1 for a term the corpus has never seen."""
        if self._corpus_size <= 1:
            return 1.0
        frequency = self._document_frequency.get(term, 0)
        if frequency == 0:
            return 1.0
        return min(1.0, math.log(self._corpus_size / frequency) / math.log(self._corpus_size))

    def check(
        self, question: str, retrieved: Sequence[RetrievedChunk], *, context_depth: int = 3
    ) -> GroundingVerdict:
        if not retrieved:
            return GroundingVerdict(False, 0.0, 0.0, [], sorted(content_terms(question)))

        best_score = max((r.rerank_score or r.score) for r in retrieved)
        terms = content_terms(question)
        if not terms:
            # A question made entirely of stopwords ("what about it?") cannot
            # be grounded on its own; the caller should have rewritten it.
            return GroundingVerdict(False, 0.0, best_score, [], [])

        context = set()
        for item in retrieved[:context_depth]:
            context |= content_terms(item.chunk.embed_text)

        total_weight = sum(self.specificity(t) for t in terms)
        matched = sorted(t for t in terms if t in context)
        missing = sorted(t for t in terms if t not in context)
        matched_weight = sum(self.specificity(t) for t in matched)
        coverage = round(matched_weight / total_weight, 6) if total_weight else 0.0

        grounded = coverage >= self.min_coverage and best_score >= self.min_score
        return GroundingVerdict(grounded, coverage, best_score, matched, missing)
