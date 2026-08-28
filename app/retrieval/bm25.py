"""Okapi BM25 sparse retrieval.

Implemented directly rather than pulled from a dependency: it is ~60 lines,
it removes a package that regularly breaks on new Python releases, and it
keeps tokenisation identical to the rest of the pipeline. It also means the
index can be rebuilt in-process during ingestion without a second service.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small: aggressive stopword lists hurt on short technical
# queries where words like "not" and "no" carry the meaning.
_STOPWORDS = frozenset(
    """
    a an the of to in for on at by is are was were be been and or as it its this that these
    those with from
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """In-memory BM25. Rebuilt on ingest; corpora up to ~10^5 chunks are fine."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, documents: Iterable[tuple[str, str]]) -> None:
        self._ids, self._term_frequencies, self._lengths = [], [], []
        self._document_frequency = Counter()

        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._ids.append(doc_id)
            self._term_frequencies.append(counts)
            self._lengths.append(len(tokens))
            self._document_frequency.update(counts.keys())

        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def _idf(self, term: str) -> float:
        n = len(self._ids)
        df = self._document_frequency.get(term, 0)
        if df == 0:
            return 0.0
        # Robertson/Sparck-Jones idf with the +1 smoothing that keeps it
        # non-negative for terms appearing in more than half the corpus.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[str, float]] = []
        for position, doc_id in enumerate(self._ids):
            counts = self._term_frequencies[position]
            length = self._lengths[position]
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._average_length or 1.0)
                )
                score += self._idf(term) * frequency * (self.k1 + 1) / denominator
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda row: (-row[1], row[0]))
        return scored[:top_k]

    def ranked_ids(self, query: str, *, top_k: int = 20) -> Sequence[str]:
        return [doc_id for doc_id, _ in self.search(query, top_k=top_k)]
