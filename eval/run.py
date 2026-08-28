"""Evaluation harness.

Scores retrieval and answer quality against `golden_set.yaml` and writes a
JSON report so runs can be diffed. Metrics are chosen to be objective and
reproducible rather than to flatter the system:

  context_recall     Did retrieval surface the document that holds the answer?
  mrr                How highly was the correct document ranked? (1/rank)
  answer_match       Does the answer contain the expected verbatim terms?
  citation_precision Of the passages cited, how many came from the right source?
  refusal_accuracy   On unanswerable questions, did it correctly decline?
  hallucination_rate On unanswerable questions, did it answer anyway?

`hallucination_rate` is the number to watch. A system can look excellent on the
answerable half and still be unusable if it invents answers for the rest.

    python -m eval.run
    python -m eval.run --compare        # diff against the previous run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import get_settings
from app.factory import build_pipeline
from app.ingest.loaders import load_path
from app.rag.pipeline import REFUSAL, RagPipeline
from eval.loader import GoldenCase, load_golden_set

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "eval" / "report.json"


@dataclass(slots=True)
class CaseResult:
    id: str
    question: str
    answerable: bool
    retrieved_sources: list[str] = field(default_factory=list)
    context_recall: float = 0.0
    reciprocal_rank: float = 0.0
    answer_match: float = 0.0
    citation_precision: float = 0.0
    refused: bool = False
    passed: bool = False
    answer: str = ""
    latency_ms: float = 0.0


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("£", "").replace(",", "").split())


async def _score_case(pipeline: RagPipeline, case: GoldenCase) -> CaseResult:
    started = time.perf_counter()
    answer = await pipeline.answer(case.question)
    latency = round((time.perf_counter() - started) * 1000, 2)

    retrieved = await pipeline.retrieve(case.question)
    sources = [item.chunk.source for item in retrieved]
    result = CaseResult(
        id=case.id,
        question=case.question,
        answerable=case.answerable,
        retrieved_sources=sources,
        answer=answer.text,
        refused=answer.text.strip() == REFUSAL or not answer.grounded,
        latency_ms=latency,
    )

    if not case.answerable:
        # The only correct behaviour is refusal.
        result.passed = result.refused
        return result

    if case.expect_source and case.expect_source in sources:
        result.context_recall = 1.0
        result.reciprocal_rank = 1.0 / (sources.index(case.expect_source) + 1)

    haystack = _normalise(answer.text)
    if case.expect_terms:
        hits = sum(1 for term in case.expect_terms if _normalise(term) in haystack)
        result.answer_match = hits / len(case.expect_terms)

    if answer.citations:
        correct = sum(1 for c in answer.citations if c.source == case.expect_source)
        result.citation_precision = correct / len(answer.citations)

    # A case passes only if it retrieved the right document AND the answer
    # actually contains the expected facts. Retrieval alone is not success.
    result.passed = result.context_recall == 1.0 and result.answer_match == 1.0
    return result


def _aggregate(results: list[CaseResult]) -> dict[str, float | int]:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "cases": len(results),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "context_recall": mean([r.context_recall for r in answerable]),
        "mrr": mean([r.reciprocal_rank for r in answerable]),
        "answer_match": mean([r.answer_match for r in answerable]),
        "citation_precision": mean([r.citation_precision for r in answerable]),
        "refusal_accuracy": mean([1.0 if r.refused else 0.0 for r in unanswerable]),
        "hallucination_rate": mean([0.0 if r.refused else 1.0 for r in unanswerable]),
        "pass_rate": mean([1.0 if r.passed else 0.0 for r in results]),
        "p50_latency_ms": round(
            sorted(r.latency_ms for r in results)[len(results) // 2], 2
        )
        if results
        else 0.0,
    }


def _bar(value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def _print_report(metrics: dict, results: list[CaseResult], config: dict) -> None:
    print("\n\033[1minsight-rag — evaluation\033[0m")
    print(
        f"  providers: llm={config['llm_provider']} "
        f"embeddings={config['embedding_provider']} store={config['vector_store']}"
    )
    print(
        f"  retrieval: dense_k={config['dense_top_k']} sparse_k={config['sparse_top_k']} "
        f"rrf_k={config['rrf_k']} rerank_n={config['rerank_top_n']}"
    )
    print(
        f"  grounding: min_coverage={config['min_grounding_coverage']} "
        f"min_score={config['min_grounding_score']}"
    )
    print(f"  corpus:    {metrics['cases']} cases "
          f"({metrics['answerable_cases']} answerable, "
          f"{metrics['unanswerable_cases']} unanswerable)\n")

    groups = [
        (
            "retrieval & safety — provider independent",
            [
                ("Context recall", metrics["context_recall"], True),
                ("MRR", metrics["mrr"], True),
                ("Refusal accuracy", metrics["refusal_accuracy"], True),
                ("Hallucination rate", metrics["hallucination_rate"], False),
            ],
        ),
        (
            f"answer quality — depends on llm={config['llm_provider']}",
            [
                ("Answer match", metrics["answer_match"], True),
                ("Citation precision", metrics["citation_precision"], True),
                ("Overall pass rate", metrics["pass_rate"], True),
            ],
        ),
    ]
    for heading, rows in groups:
        print(f"  \033[90m{heading}\033[0m")
        for label, value, higher_better in rows:
            good = value >= 0.8 if higher_better else value <= 0.2
            colour = "\033[32m" if good else "\033[33m"
            print(f"  {label:<20} {colour}{_bar(value)}\033[0m {value:.2%}")
        print()

    print(f"\n  p50 latency: {metrics['p50_latency_ms']} ms")

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  \033[33m{len(failures)} case(s) needing attention:\033[0m")
        for result in failures:
            reason = (
                "answered an unanswerable question"
                if not result.answerable
                else f"recall={result.context_recall:.0f} match={result.answer_match:.2f}"
            )
            print(f"    · {result.id:<24} {reason}")
    print()


def _compare(previous: dict, current: dict) -> None:
    print("  \033[1mvs previous run\033[0m")
    for key in (
        "context_recall",
        "mrr",
        "answer_match",
        "citation_precision",
        "refusal_accuracy",
        "hallucination_rate",
        "pass_rate",
    ):
        before, after = previous.get(key, 0.0), current.get(key, 0.0)
        delta = after - before
        arrow = "→" if abs(delta) < 1e-9 else ("↑" if delta > 0 else "↓")
        colour = "\033[90m" if abs(delta) < 1e-9 else ("\033[32m" if delta > 0 else "\033[31m")
        print(f"    {key:<20} {before:.2%} {colour}{arrow} {after:.2%} "
              f"({delta:+.2%})\033[0m")
    print()


async def main_async(compare: bool) -> int:
    settings = get_settings()
    pipeline = build_pipeline(settings)

    for path in sorted((ROOT / "sample_docs").glob("*.md")):
        await pipeline.ingest(load_path(path))

    cases = load_golden_set(ROOT / "eval" / "golden_set.yaml")
    results = [await _score_case(pipeline, case) for case in cases]
    metrics = _aggregate(results)

    config = {
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "vector_store": settings.vector_store,
        "dense_top_k": settings.dense_top_k,
        "sparse_top_k": settings.sparse_top_k,
        "rrf_k": settings.rrf_k,
        "rerank_top_n": settings.rerank_top_n,
        "min_grounding_coverage": settings.min_grounding_coverage,
        "min_grounding_score": settings.min_grounding_score,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }

    previous = None
    if compare and REPORT_PATH.exists():
        previous = json.loads(REPORT_PATH.read_text())

    _print_report(metrics, results, config)
    if previous:
        _compare(previous["metrics"], metrics)

    REPORT_PATH.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": config,
                "metrics": metrics,
                "cases": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"  report written to {REPORT_PATH.relative_to(ROOT)}\n")

    # The CI gate asserts only what an offline run can honestly prove:
    # retrieval quality and refusal behaviour. Answer-match and citation
    # precision are reported but not gated, because with LLM_PROVIDER=fake
    # they measure a deliberately simple extractive stand-in rather than the
    # model a deployment would actually use.
    failed = (
        metrics["context_recall"] < 0.9
        or metrics["mrr"] < 0.8
        or metrics["hallucination_rate"] > 0.0
        or metrics["refusal_accuracy"] < 1.0
    )
    if failed:
        print("  \033[31mFAIL\033[0m retrieval or refusal thresholds not met\n")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the insight-rag evaluation suite.")
    parser.add_argument(
        "--compare", action="store_true", help="diff against the previous report"
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args.compare))


if __name__ == "__main__":
    raise SystemExit(main())
