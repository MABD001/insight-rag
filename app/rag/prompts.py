"""Prompt construction.

Kept in one module so the exact wording is reviewable and versionable — the
prompt is as much a part of retrieval quality as the retriever, and silently
editing it inside a request handler is how RAG systems become unreproducible.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.base import RetrievedChunk

# Separates the instruction block from the retrieved passages. Anything that
# parses the assembled prompt must split on this rather than scanning for
# bracket markers, which also appear inside the instructions themselves.
CONTEXT_MARKER = "Context passages:"

SYSTEM_PROMPT_HEADER = """\
You answer questions strictly from the numbered context passages below.

Rules:
1. Use ONLY the context. Never rely on prior knowledge, even if you are confident.
2. Cite every factual claim with its passage number in square brackets, e.g. [2].
   A sentence drawing on two passages gets both, e.g. [1][3].
3. If the context does not contain the answer, reply exactly:
   "I could not find an answer to that in the provided documents."
   Do not guess, and do not offer a partial answer built on assumptions.
4. Quote figures, dates, names and identifiers exactly as they appear.
5. Be concise. Two or three sentences unless the question needs a list.

Context passages:
"""

QUERY_REWRITE_PROMPT = """\
Rewrite the user's latest message into a standalone search query that will
work without the conversation history. Resolve pronouns and implicit
references using the history. Keep any exact identifiers, error codes,
product names or figures verbatim. Return ONLY the rewritten query.

Conversation history:
{history}

Latest message: {question}

Standalone query:"""


def build_context_block(chunks: Sequence[RetrievedChunk], *, max_chars: int) -> str:
    """Render retrieved chunks as numbered passages, respecting a char budget.

    Truncation is by whole passage: a half-included passage produces citations
    that point at text the model never actually saw.
    """
    parts: list[str] = []
    used = 0
    for position, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        location = f"{chunk.source}"
        if chunk.page is not None:
            location += f", page {chunk.page}"
        if chunk.heading_path:
            location += f" — {' > '.join(chunk.heading_path)}"
        block = f"[{position}] ({location})\n{chunk.text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_system_prompt(chunks: Sequence[RetrievedChunk], *, max_chars: int) -> str:
    return SYSTEM_PROMPT_HEADER + "\n" + build_context_block(chunks, max_chars=max_chars)


def build_rewrite_prompt(question: str, history: Sequence[tuple[str, str]]) -> str:
    rendered = (
        "\n".join(f"{role}: {content}" for role, content in history[-6:])
        if history
        else "(none)"
    )
    return QUERY_REWRITE_PROMPT.format(history=rendered, question=question)
