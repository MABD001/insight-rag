"""CLI ingestion: `python -m scripts.ingest <path> [path...]`.

Useful for indexing a corpus before the API starts, and for re-indexing from
a cron job or a deployment hook.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.config import get_settings
from app.factory import build_pipeline
from app.ingest.loaders import SUPPORTED_SUFFIXES, load_path


async def main(paths: list[str]) -> int:
    pipeline = build_pipeline(get_settings())

    targets: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            targets.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            )
        elif path.exists():
            targets.append(path)
        else:
            print(f"  skip   {raw} (not found)", file=sys.stderr)

    if not targets:
        print("Nothing to ingest.", file=sys.stderr)
        return 1

    total = 0
    for target in targets:
        try:
            result = await pipeline.ingest(load_path(target))
        except ValueError as exc:
            print(f"  skip   {target} ({exc})", file=sys.stderr)
            continue
        total += result.chunks
        status = "skip" if result.skipped else "index"
        print(f"  {status:<6} {target}  ({result.chunks} chunks)")

    stats = await pipeline.store.stats()
    print(
        f"\nIndexed {total} new chunks. "
        f"Corpus: {stats['documents']} docs, {stats['chunks']} chunks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:] or ["sample_docs"])))
