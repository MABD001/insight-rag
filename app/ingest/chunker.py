"""Structure-aware chunking.

Naive fixed-width splitting cuts sentences in half, which shows up later as
citations that quote fragments and as retrieval that matches the wrong half
of an idea. This splitter respects document structure in decreasing order of
preference — markdown headings, then paragraphs, then sentences — and only
falls back to a hard character cut when a single sentence exceeds the budget.

Each chunk carries the heading trail it was found under. That trail is
prepended to the embedded text, so a chunk reading "Refunds are processed
within 5 days" still matches a question about "billing policy" when it sits
under a `## Billing` heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingest.loaders import LoadedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PAGE_RE = re.compile(r"^\[page (\d+)\]$")


@dataclass(slots=True)
class Chunk:
    text: str
    index: int
    heading_path: list[str] = field(default_factory=list)
    page: int | None = None
    char_start: int = 0
    char_end: int = 0

    @property
    def embed_text(self) -> str:
        """Text actually sent to the embedder, enriched with its heading trail."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


@dataclass(slots=True)
class _Section:
    heading_path: list[str]
    page: int | None
    lines: list[str] = field(default_factory=list)
    char_start: int = 0

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def _split_into_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    heading_path: list[str] = []
    page: int | None = None
    current = _Section(heading_path=[], page=None, char_start=0)
    offset = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()

        if page_match := _PAGE_RE.match(stripped):
            page = int(page_match.group(1))
            if current.body:
                sections.append(current)
            current = _Section(heading_path=list(heading_path), page=page, char_start=offset)
            offset += len(line)
            continue

        if heading_match := _HEADING_RE.match(stripped):
            if current.body:
                sections.append(current)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_path = heading_path[: level - 1] + [title]
            current = _Section(heading_path=list(heading_path), page=page, char_start=offset)
            offset += len(line)
            continue

        current.lines.append(line.rstrip("\n"))
        offset += len(line)

    if current.body:
        sections.append(current)
    return sections


def _pack(units: list[str], size: int, overlap: int, joiner: str) -> list[str]:
    """Greedily fill chunks up to `size`, carrying `overlap` characters forward."""
    chunks: list[str] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer}{joiner}{unit}" if buffer else unit
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
            tail = buffer[-overlap:] if overlap else ""
            # Resume at a word boundary so the overlap never starts mid-word.
            if tail and (space := tail.find(" ")) != -1:
                tail = tail[space + 1 :]
            buffer = f"{tail}{joiner}{unit}" if tail else unit
        else:
            buffer = unit
        while len(buffer) > size:
            chunks.append(buffer[:size])
            buffer = buffer[size - overlap :] if overlap else buffer[size:]
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def chunk_document(
    document: LoadedDocument, *, chunk_size: int = 900, chunk_overlap: int = 150
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for section in _split_into_sections(document.text):
        body = section.body
        if not body:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        pieces = _pack(paragraphs, chunk_size, chunk_overlap, joiner="\n\n")

        # A single oversized paragraph still needs sentence-level splitting.
        refined: list[str] = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                refined.append(piece)
                continue
            sentences = [s.strip() for s in _SENTENCE_RE.split(piece) if s.strip()]
            refined.extend(_pack(sentences, chunk_size, chunk_overlap, joiner=" "))

        cursor = section.char_start
        for piece in refined:
            chunks.append(
                Chunk(
                    text=piece,
                    index=len(chunks),
                    heading_path=list(section.heading_path),
                    page=section.page,
                    char_start=cursor,
                    char_end=cursor + len(piece),
                )
            )
            cursor += max(len(piece) - chunk_overlap, 1)

    return chunks
