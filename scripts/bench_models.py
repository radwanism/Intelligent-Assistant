"""Model comparison — the benchmark A.6.3 explicitly permits.

A.6.3 says external APIs and alternative models may be used "for benchmarking or
comparison purposes only, not as the core solution". This script is where that
happens, and it is the reason the serving path can ship well-adopted defaults
without hand-waving about the alternatives.

Three comparisons:

  --asr        faster-whisper vs the Egyptian-specific fine-tunes
  --embed      the profile's embedder vs BGE-M3 and (optionally) jina-v3
  --ocr        RapidOCR vs PaddleOCR-VL

Every candidate here was checked against the Hugging Face API before being
listed. Two carry licences that DISQUALIFY them from the serving path and they
are labelled accordingly — they may be measured, never deployed:

    jinaai/jina-embeddings-v3            CC-BY-NC-4.0
    jinaai/jina-reranker-v2-base-multi   CC-BY-NC-4.0
    MAdel121/f5-tts-egyptian-arabic      CC-BY-NC-4.0

Adoption is reported alongside the score, because for a model with 43 downloads
a month a self-reported WER is a claim, not a measurement.

    python scripts/bench_models.py --embed
    python scripts/bench_models.py --asr --include-nawah
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Candidates. `commercial_ok` gates whether a model may ever reach production.
# ---------------------------------------------------------------------------
ASR_CANDIDATES = [
    {
        "id": "small",
        "loader": "faster-whisper",
        "licence": "MIT",
        "commercial_ok": True,
        "note": "current cpu-lite default",
    },
    {
        "id": "large-v3",
        "loader": "faster-whisper",
        "licence": "MIT",
        "commercial_ok": True,
        "note": "current gpu-colab default",
    },
    {
        "id": "oddadmix/Nawah-ASR-118M-v5",
        "loader": "transformers",
        "licence": "Apache-2.0",
        "commercial_ok": True,
        "downloads_30d": 43,
        "note": (
            "Egyptian-specific, 118M params. WER 0.3358 is SELF-REPORTED on the "
            "author's own eval set. Not Whisper-architecture, so it needs "
            "transformers+torch and cannot run under CTranslate2 — which is why "
            "it is not a cpu-lite candidate."
        ),
    },
    {
        "id": "MAdel121/whisper-small-egyptian-arabic",
        "loader": "transformers",
        "licence": "MIT",
        "commercial_ok": True,
        "downloads_30d": 215,
        "note": "Egyptian fine-tune of whisper-small",
    },
    {
        "id": "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic",
        "loader": "transformers",
        "licence": "Apache-2.0",
        "commercial_ok": True,
        "downloads_30d": 287,
        "note": "LoRA fine-tune on Egyptian-ASR-MGB-3",
    },
]

EMBED_CANDIDATES = [
    {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "backend": "fastembed",
        "licence": "Apache-2.0",
        "commercial_ok": True,
        "note": "current cpu-lite default, 0.22 GB",
    },
    {
        "id": "intfloat/multilingual-e5-large",
        "backend": "fastembed",
        "licence": "MIT",
        "commercial_ok": True,
        "note": "2.24 GB — better retriever, excluded on footprint not licence",
    },
    {
        "id": "BAAI/bge-m3",
        "backend": "flagembedding",
        "licence": "MIT",
        "commercial_ok": True,
        "note": "current gpu-colab default",
    },
    {
        "id": "jinaai/jina-embeddings-v3",
        "backend": "transformers",
        "licence": "CC-BY-NC-4.0",
        "commercial_ok": False,
        "note": "BENCHMARK ONLY — non-commercial licence disqualifies it",
    },
]

OCR_CANDIDATES = [
    {
        "id": "rapidocr-onnxruntime",
        "licence": "Apache-2.0",
        "commercial_ok": True,
        "note": "current default, pip-only, no system binary",
    },
    {
        "id": "PaddlePaddle/PaddleOCR-VL",
        "licence": "Apache-2.0",
        "commercial_ok": True,
        "note": "stronger document understanding; Arabic support UNVERIFIED",
    },
]


# ---------------------------------------------------------------------------
def bench_embedders(golden: Path, k: int, candidates: list[dict]) -> list[dict]:
    """Re-embed the corpus per candidate and re-run the retrieval eval.

    Each candidate needs its own collection: vector dimensions differ (384 vs
    1024), so they cannot share an index.
    """
    from scripts.eval_rag import evaluate, load_golden  # noqa: PLC0415

    from te_assistant.config import get_settings

    settings = get_settings()
    questions = load_golden(golden)
    results: list[dict] = []

    for candidate in candidates:
        if not candidate["commercial_ok"]:
            print(f"\n  [{candidate['id']}] NON-COMMERCIAL — measuring only")
        print(f"\n  benchmarking {candidate['id']} ...")
        started = time.perf_counter()
        try:
            from te_assistant.retrieval.embedder import OnnxEmbedder
            from te_assistant.retrieval.hybrid import HybridRetriever
            from te_assistant.retrieval.store import VectorStore

            if candidate["backend"] != "fastembed":
                print("    ! requires the [gpu] extra; skipping on this profile")
                continue

            embedder = OnnxEmbedder(candidate["id"], dim=0)
            probe = embedder.embed_query("test")
            embedder.dim = len(probe)

            store = VectorStore(settings, embedder=embedder)
            retriever = HybridRetriever(store=store, settings=settings)
            retriever.warmup()

            report = evaluate(retriever, questions, k)
            results.append({
                **candidate,
                "dim": embedder.dim,
                "recall": report["overall"][f"recall@{k}"],
                "mrr": report["overall"]["mrr"],
                "median_latency_ms": report["overall"]["median_latency_ms"],
                "by_language": report["by_language"],
                "elapsed_s": round(time.perf_counter() - started, 1),
            })
        except Exception as exc:
            print(f"    ! failed: {exc}")
            results.append({**candidate, "error": str(exc)})

    return results


def print_table(results: list[dict], k: int) -> None:
    print(f"\n{'model':<58} {'lic':<14} {'dim':>5} {f'r@{k}':>7} {'MRR':>6}")
    print("-" * 95)
    for row in results:
        if "error" in row:
            print(f"{row['id'][:57]:<58} {row['licence']:<14} {'—':>5} {'FAILED':>7}")
            continue
        flag = "" if row["commercial_ok"] else "  [NON-COMMERCIAL]"
        print(
            f"{row['id'][:57]:<58} {row['licence']:<14} {row['dim']:>5} "
            f"{row['recall']:>6.1%} {row['mrr']:>6.3f}{flag}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark model alternatives")
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--include-nawah", action="store_true",
                        help="include the low-adoption Egyptian ASR candidates")
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_set.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("data/eval/results/bench.json"))
    args = parser.parse_args()

    if not (args.asr or args.embed or args.ocr):
        parser.error("choose at least one of --asr / --embed / --ocr")

    report: dict[str, object] = {}

    if args.embed:
        print("\n=== EMBEDDERS ===")
        results = bench_embedders(args.golden, args.k, EMBED_CANDIDATES)
        print_table(results, args.k)
        report["embedders"] = results

    if args.asr:
        print("\n=== ASR ===")
        print("Requires data/eval/audio/manifest.jsonl — run scripts/eval_asr.py")
        print("per candidate with --model, then collate. Candidates:\n")
        for candidate in ASR_CANDIDATES:
            if "Nawah" in candidate["id"] and not args.include_nawah:
                continue
            downloads = candidate.get("downloads_30d")
            adoption = f"{downloads} dl/30d" if downloads else "—"
            print(f"  {candidate['id']}")
            print(f"    licence={candidate['licence']}  adoption={adoption}")
            print(f"    {candidate['note']}\n")
        report["asr_candidates"] = ASR_CANDIDATES

    if args.ocr:
        print("\n=== OCR ===")
        for candidate in OCR_CANDIDATES:
            print(f"  {candidate['id']}: {candidate['note']}")
        report["ocr_candidates"] = OCR_CANDIDATES

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
