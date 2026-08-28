import pytest

from app.rag.pipeline import REFUSAL, RagPipeline


async def test_ingest_produces_chunks(pipeline: RagPipeline):
    stats = await pipeline.store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] > 1


async def test_reingesting_identical_content_is_skipped(pipeline: RagPipeline, document):
    before = await pipeline.store.stats()
    result = await pipeline.ingest(document)
    after = await pipeline.store.stats()

    assert result.skipped is True
    assert result.chunks == 0
    assert after == before


async def test_force_reingest_is_not_skipped(pipeline: RagPipeline, document):
    result = await pipeline.ingest(document, force=True)
    assert result.skipped is False
    assert result.chunks > 0


async def test_hybrid_retrieval_finds_an_exact_error_code(pipeline: RagPipeline):
    results = await pipeline.retrieve("AX-7781")
    assert results
    assert "AX-7781" in results[0].chunk.text


async def test_answer_is_grounded_and_cited(pipeline: RagPipeline):
    answer = await pipeline.answer("How long do refunds take?")
    assert answer.grounded
    assert answer.citations
    assert all(c.marker > 0 for c in answer.citations)
    assert answer.timings_ms["total"] > 0


async def test_citations_only_include_passages_the_answer_used(pipeline: RagPipeline):
    answer = await pipeline.answer("How long do refunds take?")
    cited_markers = {c.marker for c in answer.citations}
    for marker in cited_markers:
        assert f"[{marker}]" in answer.text


async def test_unanswerable_question_is_refused(pipeline: RagPipeline):
    answer = await pipeline.answer("What is the CEO's home address?")
    assert not answer.grounded
    assert answer.text == REFUSAL
    assert answer.citations == []
    assert answer.refusal_reason


async def test_deleting_a_document_removes_it_from_retrieval(pipeline: RagPipeline):
    hashes = await pipeline.store.document_hashes()
    document_id = next(iter(hashes))

    removed = await pipeline.store.delete_document(document_id)
    await pipeline.refresh_sparse_index()

    assert removed > 0
    assert await pipeline.retrieve("AX-7781") == []


async def test_streaming_emits_meta_tokens_then_done(pipeline: RagPipeline):
    events = [e async for e in pipeline.stream_answer("How long do refunds take?")]

    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["grounded"] is True


async def test_streamed_and_buffered_answers_agree(pipeline: RagPipeline):
    buffered = await pipeline.answer("How long do refunds take?")
    tokens = [
        event["text"]
        async for event in pipeline.stream_answer("How long do refunds take?")
        if event["type"] == "token"
    ]
    assert "".join(tokens).strip() == buffered.text.strip()


async def test_streaming_refusal_still_closes_cleanly(pipeline: RagPipeline):
    events = [e async for e in pipeline.stream_answer("Who won the 1998 world cup?")]
    assert events[-1]["type"] == "done"
    assert events[-1]["grounded"] is False


@pytest.mark.parametrize(
    "question",
    ["How long do refunds take?", "When does express delivery arrive?"],
)
async def test_answers_are_deterministic(pipeline: RagPipeline, question: str):
    # Determinism is what makes the eval harness a regression gate rather
    # than a coin flip.
    first = await pipeline.answer(question)
    second = await pipeline.answer(question)
    assert first.text == second.text
