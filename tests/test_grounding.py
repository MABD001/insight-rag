"""The guardrail is the difference between a usable system and a liability."""

from app.rag.grounding import GroundingChecker, content_terms, stem
from app.retrieval.base import RetrievedChunk, StoredChunk


def _chunk(chunk_id: str, text: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        document_id="doc",
        source="policy.md",
        title="Policy",
        text=text,
        embed_text=text,
    )


CORPUS = [
    _chunk(
        "1",
        "Approved refunds are returned to the original payment method within 5 business days.",
    ),
    _chunk("2", "Standard delivery takes 3 to 5 working days and costs 4.95 pounds."),
    _chunk("3", "Error code AX-7781 indicates a failed temperature sensor on the mixer."),
]


def _checker() -> GroundingChecker:
    checker = GroundingChecker()
    checker.fit(CORPUS)
    return checker


def _retrieved(*chunks: StoredChunk) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=c, score=0.5, rerank_score=0.5) for c in chunks]


def test_stemming_unifies_simple_inflections():
    assert stem("refunds") == stem("refund")
    assert stem("delivered") == stem("deliver")


def test_stemming_leaves_short_tokens_alone():
    # Over-stemming manufactures false matches, which is worse for a guardrail
    # than missing a match.
    assert stem("as") == "as"
    assert stem("gas") == "gas"


def test_interrogatives_are_not_content_terms():
    assert content_terms("How long does it take?") == {"take"} or "how" not in content_terms(
        "How long does it take?"
    )


def test_answerable_question_is_grounded():
    verdict = _checker().check("How long do refunds take?", _retrieved(CORPUS[0]))
    assert verdict.grounded
    assert verdict.coverage > 0


def test_unanswerable_question_is_refused():
    # Every content term is absent from the corpus, so coverage is zero.
    verdict = _checker().check(
        "Who is the chief technology officer?", _retrieved(CORPUS[0])
    )
    assert not verdict.grounded
    assert verdict.coverage == 0.0
    assert "no query terms" in verdict.reason


def test_terms_unknown_to_the_corpus_are_maximally_specific():
    checker = _checker()
    assert checker.specificity("bitcoin") == 1.0
    # "days" appears in two of three chunks, so it discriminates poorly.
    assert checker.specificity(stem("days")) < checker.specificity("bitcoin")


def test_empty_retrieval_is_never_grounded():
    assert not _checker().check("anything at all", []).grounded


def test_question_of_only_stopwords_is_refused():
    verdict = _checker().check("what about it?", _retrieved(CORPUS[0]))
    assert not verdict.grounded


def test_verdict_reports_which_terms_were_missing():
    verdict = _checker().check("refunds for bitcoin purchases", _retrieved(CORPUS[0]))
    assert "bitcoin" in verdict.missing_terms
