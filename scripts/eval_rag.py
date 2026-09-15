"""Retrieval evaluation — brief B.10.

Measures the property the design is built around: **can a question phrased the
way an Egyptian customer actually phrases it find the page that answers it,
including when that page is in MSA or English?**

Metrics:
  recall@k        did any expected source appear in the top k
  MRR             how high the first correct source ranked
  latency         per-query, so the B.10 budget is measured not asserted

The golden set is split by query language so the cross-lingual case is reported
separately rather than averaged away — an overall recall@5 that hides "English
queries retrieve nothing" would be a misleading number.

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

from te_assistant.config import get_settings
from te_assistant.retrieval.hybrid import HybridRetriever, diversify
from te_assistant.retrieval.normalize_ar import detect_language

DEFAULT_GOLDEN = Path("data/eval/golden_set.jsonl")


def load_golden(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"no golden set at {path}. Run scripts/make_golden_set.py first."
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


def evaluate(retriever: HybridRetriever, golden: list[dict], k: int) -> dict:
    per_language: dict[str, list[dict]] = defaultdict(list)
    rows: list[dict] = []

    for item in golden:
        question = item["question"]
        expected = item["expected_url_contains"]
        language = item.get("language") or detect_language(question).value

        started = time.perf_counter()
        result = retriever.retrieve(question)
        elapsed_ms = (time.perf_counter() - started) * 1000

        top = diversify(result.chunks)[:k]
        urls = [scored.chunk.url for scored in top]

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
            "top_urls": urls[:3],
        }
        rows.append(row)
        per_language[language].append(row)

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
        "overall": summarise(rows),
        "by_language": {lang: summarise(items) for lang, items in per_language.items()},
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

    report = evaluate(retriever, golden, args.k)
    report["profile"] = settings.profile.value
    report["embedder"] = settings.slots.embedder
    report["reranker"] = settings.slots.reranker
    report["kb_chunks"] = retriever.store.kb_size()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = report["overall"]
    print(f"\nprofile   : {report['profile']}  ({report['embedder']})")
    print(f"kb chunks : {report['kb_chunks']}")
    print(f"questions : {overall['n']}\n")
    print(f"  recall@{args.k} : {overall[f'recall@{args.k}']:.1%}")
    print(f"  MRR        : {overall['mrr']:.3f}")
    print(f"  latency    : {overall['median_latency_ms']:.0f} ms (median)\n")

    print("  by query language:")
    for language, stats in sorted(report["by_language"].items()):
        print(
            f"    {language:6} n={stats['n']:<3} "
            f"recall@{args.k}={stats[f'recall@{args.k}']:.1%}  mrr={stats['mrr']:.3f}"
        )

    if report["misses"]:
        print(f"\n  {len(report['misses'])} miss(es):")
        for miss in report["misses"]:
            print(f"    [{miss['language']}] {miss['question']}")

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
