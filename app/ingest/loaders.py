"""Document loaders.

Each loader returns plain text plus a content hash. The hash is what makes
re-ingestion incremental: unchanged documents are skipped instead of being
re-embedded, which is the difference between a $0.02 and a $20 re-index on a
large corpus.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


@dataclass(slots=True)
class LoadedDocument:
    source: str
    title: str
    text: str
    content_hash: str
    media_type: str


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            # Page markers survive chunking, so a citation can point a reader
            # at the actual page of the source PDF.
            pages.append(f"[page {number}]\n{text}")
    return "\n\n".join(pages)


def _load_html(data: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


def load_bytes(data: bytes, *, filename: str) -> LoadedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text, media_type = _load_pdf(data), "application/pdf"
    elif suffix in {".html", ".htm"}:
        text, media_type = _load_html(data), "text/html"
    elif suffix in {".md", ".markdown"}:
        text, media_type = data.decode("utf-8", errors="replace"), "text/markdown"
    elif suffix == ".txt":
        text, media_type = data.decode("utf-8", errors="replace"), "text/plain"
    else:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    text = text.strip()
    if not text:
        raise ValueError(f"No extractable text in {filename!r}")

    return LoadedDocument(
        source=filename,
        title=_derive_title(text, fallback=Path(filename).stem),
        text=text,
        content_hash=_hash(text),
        media_type=media_type,
    )


def load_path(path: str | Path) -> LoadedDocument:
    path = Path(path)
    return load_bytes(path.read_bytes(), filename=path.name)


def _derive_title(text: str, *, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and not stripped.startswith("[page "):
            return stripped[:200]
    return fallback
