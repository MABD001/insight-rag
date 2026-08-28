"""Reciprocal rank fusion.

Dense and sparse retrieval fail in opposite directions: embeddings miss exact
identifiers ("error SB-4021", a part number, a surname), and BM25 misses
paraphrase. RRF merges the two ranked lists using rank position only, so the
two scoring scales never have to be normalised against each other — which is
what makes it robust when one retriever returns wildly different magnitudes.

    score(d) = sum over lists of 1 / (k + rank(d))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[str]], *, k: int = 60
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse named ranked lists of ids.

    Returns `(id, fused_score, {list_name: rank})`, highest score first, with
    ranks 1-indexed so they can be surfaced in the API for debugging.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for list_name, ids in ranked_lists.items():
        for position, item_id in enumerate(ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(item_id, {})[list_name] = position

    fused = [(item_id, score, ranks[item_id]) for item_id, score in scores.items()]
    # Tie-break on id so results are stable across runs.
    fused.sort(key=lambda row: (-row[1], row[0]))
    return fused
