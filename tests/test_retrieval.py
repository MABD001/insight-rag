from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion


def test_bm25_finds_exact_identifiers():
    # The case dense retrieval is worst at: an error code carries almost no
    # semantic signal, so embeddings cannot place it usefully.
    index = BM25Index()
    index.build(
        [
            ("a", "Error code AX-7781 indicates a failed temperature sensor."),
            ("b", "Standard delivery takes three to five working days."),
            ("c", "Refunds are returned to the original payment method."),
        ]
    )
    results = index.search("AX-7781", top_k=3)
    assert results[0][0] == "a"


def test_bm25_prefers_rarer_terms():
    index = BM25Index()
    index.build(
        [
            ("common", "delivery delivery delivery delivery"),
            ("rare", "delivery hydraulic coupling"),
        ]
    )
    results = dict(index.search("hydraulic", top_k=2))
    assert "rare" in results
    assert "common" not in results


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index()
    index.build([("a", "refunds and shipping")])
    assert index.search("thermodynamics", top_k=5) == []


def test_bm25_handles_an_empty_index():
    assert BM25Index().search("anything", top_k=5) == []


def test_rrf_rewards_agreement_between_retrievers():
    # "b" is mid-ranked in both lists; "a" tops one and is absent from the
    # other. Agreement should win, which is the entire point of fusion.
    fused = reciprocal_rank_fusion(
        {"dense": ["a", "b", "c"], "sparse": ["c", "b", "d"]}, k=1
    )
    ordering = [item_id for item_id, _, _ in fused]
    assert ordering[0] in {"b", "c"}
    assert set(ordering) == {"a", "b", "c", "d"}


def test_rrf_reports_the_contributing_ranks():
    fused = reciprocal_rank_fusion({"dense": ["x", "y"], "sparse": ["y"]}, k=60)
    ranks = {item_id: rank for item_id, _, rank in fused}
    assert ranks["y"] == {"dense": 2, "sparse": 1}
    assert ranks["x"] == {"dense": 1}


def test_rrf_is_deterministic_for_tied_scores():
    lists = {"a": ["p", "q"], "b": ["q", "p"]}
    first = reciprocal_rank_fusion(lists, k=60)
    second = reciprocal_rank_fusion(lists, k=60)
    assert first == second


def test_rrf_rejects_a_non_positive_k():
    try:
        reciprocal_rank_fusion({"a": ["x"]}, k=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
