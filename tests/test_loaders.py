import pytest

from app.ingest.loaders import load_bytes


def test_markdown_title_comes_from_the_first_heading():
    document = load_bytes(b"# Warranty Terms\n\nBody text here.", filename="w.md")
    assert document.title == "Warranty Terms"


def test_html_scripts_and_navigation_are_stripped():
    html = b"""<html><head><style>p{color:red}</style></head>
    <body><nav>Menu</nav><p>Refunds take 5 days.</p>
    <script>alert(1)</script><footer>(c) 2026</footer></body></html>"""
    document = load_bytes(html, filename="page.html")

    assert "Refunds take 5 days." in document.text
    assert "alert(1)" not in document.text
    assert "Menu" not in document.text


def test_identical_content_hashes_identically():
    first = load_bytes(b"# A\n\nSame body.", filename="one.md")
    second = load_bytes(b"# A\n\nSame body.", filename="two.md")
    # Equal hashes are what make re-ingestion skippable across renames.
    assert first.content_hash == second.content_hash


def test_differing_content_hashes_differently():
    first = load_bytes(b"# A\n\nBody one.", filename="a.md")
    second = load_bytes(b"# A\n\nBody two.", filename="a.md")
    assert first.content_hash != second.content_hash


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_bytes(b"data", filename="archive.zip")


def test_empty_document_raises():
    with pytest.raises(ValueError, match="No extractable text"):
        load_bytes(b"   \n  ", filename="blank.md")
