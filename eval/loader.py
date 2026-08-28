"""Minimal YAML subset reader for the golden set.

Deliberately not a PyYAML dependency: the golden set uses a fixed, flat shape
(list of mappings with scalar and inline-list values), and parsing it here
keeps the eval harness runnable with zero extra install for anyone auditing
the repo. If the schema ever grows, swap this for PyYAML — `load_golden_set`
is the only caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class GoldenCase:
    id: str
    question: str
    answerable: bool = True
    expect_source: str | None = None
    expect_terms: list[str] = field(default_factory=list)


def _parse_scalar(raw: str) -> object:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in _split_inline(inner)]
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    return raw.strip("\"'")


def _split_inline(inner: str) -> list[str]:
    """Split on commas that are not inside quotes."""
    parts, buffer, quote = [], "", ""
    for char in inner:
        if quote:
            if char == quote:
                quote = ""
            buffer += char
        elif char in "\"'":
            quote = char
            buffer += char
        elif char == ",":
            parts.append(buffer)
            buffer = ""
        else:
            buffer += char
    if buffer.strip():
        parts.append(buffer)
    return parts


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    current: dict[str, object] = {}

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split(" #")[0].rstrip() if " #" in raw_line else raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if line.startswith("- "):
            if current:
                cases.append(_build(current))
            current = {}
            line = "  " + line[2:]

        key, _, value = line.strip().partition(":")
        if not _:
            continue
        current[key.strip()] = _parse_scalar(value)

    if current:
        cases.append(_build(current))
    return cases


def _build(row: dict[str, object]) -> GoldenCase:
    return GoldenCase(
        id=str(row["id"]),
        question=str(row["question"]),
        answerable=bool(row.get("answerable", True)),
        expect_source=str(row["expect_source"]) if row.get("expect_source") else None,
        expect_terms=[str(t) for t in (row.get("expect_terms") or [])],
    )
