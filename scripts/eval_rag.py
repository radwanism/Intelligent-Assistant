"""Retrieval evaluation — brief B.10.

Measures the property the design is built around: **can a question phrased the
way an Egyptian customer actually phrases it find the page that answers it,
including when that page is in MSA or English?** — and, separately, **does the
assistant decline when nothing on te.eg answers the question?**

Retrieval metrics (in-scope questions only):
  recall@k        did any expected source appear in the top k
  MRR             how high the first correct source ranked
  latency         per-query, so the B.10 budget is measured not asserted

Refusal metrics (the retrieval gate, no model needed):
  false refusal   in-scope questions the gate would turn away
  refusal         out-of-scope questions the gate catches
  separation      whether any threshold cleanly divides the two populations

The golden set is split by query language so the cross-lingual case is reported
separately rather than averaged away — an overall recall@5 that hides "English
queries retrieve nothing" would be a misleading number.

The ordering evaluated is the serving ordering (`select_context`), not a copy of
it. On a profile with a reranker the report also gives the fusion-only numbers,
which is what shows whether the reranker earns its latency.

What this does NOT measure: whether the *model* then declines or answers
faithfully, and whether a cited source supports the claim. Those need the LLM
and a judged answer set; the refusal figures here describe the retrieval gate
only.

Usage:
    python scripts/eval_rag.py [--k 5] [--out data/eval/results/rag.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from te_assistant.config import Settings, get_settings
from te_assistant.retrieval.context import select_context
from te_assistant.retrieval.hybrid import HybridRetriever, confidence_of
from te_assistant.retrieval.normalize_ar import detect_language
from te_assistant.retrieval.rerank import get_reranker

DEFAULT_GOLDEN = Path("data/eval/golden_set.jsonl")


def load_golden(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"no golden set at {path}. It is committed with the repository — "
            f"restore it with `git checkout -- {path}`."
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _matches(url: str, expected: list[str]) -> bool:
    """A hit is a retrieved chunk whose URL contains an expected path fragment.

    Substring rather than equality because te.eg serves the same content under
    several URL shapes (/personal/x and /web/guest/personal/x), and scoring
    those as misses would measure URL routing, not retrieval.
    """
    return any(fragment in url for fragment in expected)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (see insights.records for why not interpolated)."""
    ordered = sorted(values)
    index = min(max(int(round(pct / 100 * len(ordered) + 0.5)) - 1, 0), len(ordered) - 1)
    return ordered[index]


def summarise_confidence(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "min": round(min(values), 3),
        "p5": round(_percentile(values, 5), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 95), 3),
        "max": round(max(values), 3),
    }


def refusal_report(in_scope: list[float], out_of_scope: list[float], threshold: float) -> dict:
    """What the retrieval gate does at `threshold`, and where it *could* sit.

    `confidence` is what generation compares against the threshold: below it the
    model is never called. The two populations only separate cleanly if the best
    out-of-scope score is below the worst in-scope one; when they overlap there
    is no threshold that is both safe and useful, and the report says so rather
    than offering a number.
    """
    report: dict = {
        "threshold": threshold,
        "in_scope": {
            "n": len(in_scope),
            "false_refusal_rate": (
                round(sum(c < threshold for c in in_scope) / len(in_scope), 3) if in_scope else None
            ),
            "confidence": summarise_confidence(in_scope),
        },
        "out_of_scope": {
            "n": len(out_of_scope),
            "refusal_rate": (
                round(sum(c < threshold for c in out_of_scope) / len(out_of_scope), 3)
                if out_of_scope else None
            ),
            "confidence": summarise_confidence(out_of_scope),
        },
        "note": (
            "retrieval gate only — whether the model then declines or answers "
            "faithfully is not measured here"
        ),
    }
    if in_scope and out_of_scope:
        worst_in, best_out = min(in_scope), max(out_of_scope)
        report["separable"] = best_out < worst_in
        report["suggested_threshold"] = (
            round((best_out + worst_in) / 2, 3) if best_out < worst_in else None
        )
        report["overlap"] = (
            None if best_out < worst_in
            else f"best out-of-scope {best_out:.3f} >= worst in-scope {worst_in:.3f}: "
                 "no threshold separates them; refusal must come from the prompt"
        )
    return report


def evaluate(
    retriever: HybridRetriever,
    golden: list[dict],
    k: int,
    *,
    settings: Settings | None = None,
    use_reranker: bool = True,
) -> dict:
    settings = settings or retriever.settings
    in_scope = [item for item in golden if not item.get("expect_refusal")]
    out_of_scope = [item for item in golden if item.get("expect_refusal")]

    per_language: dict[str, list[dict]] = defaultdict(list)
    rows: list[dict] = []
    in_scope_confidence: list[float] = []

    for item in in_scope:
        question = item["question"]
        expected = item["expected_url_contains"]
        language = item.get("language") or detect_language(question).value

        started = time.perf_counter()
        result = retriever.retrieve(question)
        selection = select_context(question, result.chunks, settings, use_reranker=use_reranker)
        elapsed_ms = (time.perf_counter() - started) * 1000

        top = selection.chunks[:k]
        urls = [scored.chunk.url for scored in top]
        in_scope_confidence.append(confidence_of(selection.chunks))

        rank = next(
            (i + 1 for i, url in enumerate(urls) if _matches(url, expected)), None
        )
        row = {
            "question": question,
            "language": language,
            "expected": expected,
            "hit": rank is not None,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "latency_ms": round(elapsed_ms, 1),
            "confidence": round(in_scope_confidence[-1], 3),
            "top_urls": urls[:3],
        }
        rows.append(row)
        per_language[language].append(row)

    out_of_scope_confidence: list[float] = []
    for item in out_of_scope:
        result = retriever.retrieve(item["question"])
        selection = select_context(
            item["question"], result.chunks, settings, use_reranker=use_reranker
        )
        out_of_scope_confidence.append(confidence_of(selection.chunks))

    def summarise(items: list[dict]) -> dict:
        if not items:
            return {}
        return {
            "n": len(items),
            f"recall@{k}": round(sum(r["hit"] for r in items) / len(items), 3),
            "mrr": round(statistics.mean(r["reciprocal_rank"] for r in items), 3),
            "median_latency_ms": round(
                statistics.median(r["latency_ms"] for r in items), 1
            ),
        }

    return {
        "k": k,
        "reranker_in_use": use_reranker and get_reranker(settings) is not None,
        "overall": summarise(rows),
        "by_language": {lang: summarise(items) for lang, items in per_language.items()},
        "refusal": refusal_report(
            in_scope_confidence, out_of_scope_confidence, settings.min_retrieval_confidence
        ),
        "misses": [
            {"question": r["question"], "language": r["language"], "top": r["top_urls"]}
            for r in rows
            if not r["hit"]
        ],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--out", type=Path, default=Path("data/eval/results/rag.json"))
    args = parser.parse_args()

    settings = get_settings()
    golden = load_golden(args.golden)

    retriever = HybridRetriever(settings=settings)
    retriever.warmup()

    report = evaluate(retriever, golden, args.k, settings=settings)
    if report["reranker_in_use"]:
        # Same questions, fusion ordering only: the difference is the reranker's worth.
        report["without_reranker"] = evaluate(
            retriever, golden, args.k, settings=settings, use_reranker=False
        )["overall"]
    report["profile"] = settings.profile.value
    report["embedder"] = settings.slots.embedder
    report["reranker"] = settings.slots.reranker
    report["kb_chunks"] = retriever.store.kb_size()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = report["overall"]
    print(f"\nprofile   : {report['profile']}  ({report['embedder']})")
    print(f"reranker  : {'in use' if report['reranker_in_use'] else 'none'}")
    print(f"kb chunks : {report['kb_chunks']}")
    print(f"questions : {overall['n']} in scope\n")
    print(f"  recall@{args.k} : {overall[f'recall@{args.k}']:.1%}")
    print(f"  MRR        : {overall['mrr']:.3f}")
    print(f"  latency    : {overall['median_latency_ms']:.0f} ms (median)\n")

    if "without_reranker" in report:
        plain = report["without_reranker"]
        print(f"  without reranker: recall@{args.k} {plain[f'recall@{args.k}']:.1%}, "
              f"MRR {plain['mrr']:.3f}, {plain['median_latency_ms']:.0f} ms\n")

    print("  by query language:")
    for language, stats in sorted(report["by_language"].items()):
        print(
            f"    {language:6} n={stats['n']:<3} "
            f"recall@{args.k}={stats[f'recall@{args.k}']:.1%}  mrr={stats['mrr']:.3f}"
        )

    refusal = report["refusal"]
    if refusal["out_of_scope"]["n"]:
        print(f"\n  refusal gate (threshold {refusal['threshold']}):")
        print(f"    in scope      n={refusal['in_scope']['n']:<3} "
              f"false refusals {refusal['in_scope']['false_refusal_rate']:.1%}  "
              f"confidence {refusal['in_scope']['confidence']}")
        print(f"    out of scope  n={refusal['out_of_scope']['n']:<3} "
              f"refused {refusal['out_of_scope']['refusal_rate']:.1%}  "
              f"confidence {refusal['out_of_scope']['confidence']}")
        if refusal.get("separable"):
            print(f"    separable; a threshold of {refusal['suggested_threshold']} "
                  "would split them (set TE_MIN_RETRIEVAL_CONFIDENCE)")
        elif refusal.get("overlap"):
            print(f"    {refusal['overlap']}")

    if report["misses"]:
        print(f"\n  {len(report['misses'])} miss(es):")
        for miss in report["misses"]:
            print(f"    [{miss['language']}] {miss['question']}")

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
