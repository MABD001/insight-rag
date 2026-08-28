from app.ingest.chunker import chunk_document
from app.ingest.loaders import load_bytes

NESTED = """\
# Manual

## Installation

Mount the bracket using the supplied screws. Torque to 12 Nm.

### Wiring

Connect the live conductor to terminal L before applying power.

## Maintenance

Replace the filter every 6 months.
"""


def test_chunks_carry_their_heading_trail():
    document = load_bytes(NESTED.encode(), filename="manual.md")
    chunks = chunk_document(document, chunk_size=400, chunk_overlap=50)

    wiring = next(c for c in chunks if "terminal L" in c.text)
    assert wiring.heading_path == ["Manual", "Installation", "Wiring"]


def test_heading_trail_is_prepended_to_embedded_text():
    # This is what lets a chunk reading "Replace the filter" match a question
    # about maintenance even though it never uses the word.
    document = load_bytes(NESTED.encode(), filename="manual.md")
    chunks = chunk_document(document, chunk_size=400, chunk_overlap=50)

    filter_chunk = next(c for c in chunks if "filter" in c.text)
    assert "Maintenance" in filter_chunk.embed_text
    assert "Maintenance" not in filter_chunk.text


def test_no_chunk_exceeds_the_size_budget():
    body = " ".join(f"Sentence number {i} about hydraulic couplings." for i in range(400))
    document = load_bytes(f"# Big\n\n{body}".encode(), filename="big.md")
    chunks = chunk_document(document, chunk_size=500, chunk_overlap=100)

    assert chunks
    assert all(len(c.text) <= 500 for c in chunks)


def test_oversized_single_paragraph_is_split_on_sentences():
    paragraph = " ".join(f"Fact {i} is recorded here." for i in range(200))
    document = load_bytes(f"# T\n\n{paragraph}".encode(), filename="p.md")
    chunks = chunk_document(document, chunk_size=300, chunk_overlap=60)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_overlap_never_starts_mid_word():
    body = " ".join(f"token{i}" for i in range(300))
    document = load_bytes(f"# T\n\n{body}".encode(), filename="o.md")
    chunks = chunk_document(document, chunk_size=200, chunk_overlap=50)

    for chunk in chunks[1:]:
        assert not chunk.text.startswith("oken")


def test_overlap_must_be_smaller_than_chunk_size():
    document = load_bytes(b"# T\n\nbody", filename="t.md")
    try:
        chunk_document(document, chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")
