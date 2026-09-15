"""ASR evaluation — WER/CER on Egyptian-dialect and noisy audio (brief B.10).

Reports WER **normalised for Arabic orthography**, which matters more than it
sounds: raw WER counts "إنترنت" against "انترنت" as an error, and on Arabic that
single convention can shift the number by several points without any difference
in what the user actually said. Both figures are reported so the normalisation
is visible rather than a thumb on the scale.

Also reports how often the hallucination filter fired, because on noisy input
that is the metric that separates "transcribed nothing" from "confidently
transcribed something that was never said".

Expects `data/eval/audio/manifest.jsonl`:
    {"audio": "clip01.wav", "reference": "عايز أعرف أسعار باقات الإنترنت",
     "dialect": "arz", "condition": "clean"}

    python scripts/eval_asr.py
    python scripts/eval_asr.py --model small --compute-type int8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from te_assistant.retrieval.normalize_ar import normalize_arabic

AUDIO_DIR = Path("data/eval/audio")
MANIFEST = AUDIO_DIR / "manifest.jsonl"


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"no manifest at {path}.\n"
            "Record a few clips (Egyptian dialect, MSA, English, plus at least\n"
            "one noisy and one silent clip) and list them with their reference\n"
            "transcripts, one JSON object per line."
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_pair(reference: str, hypothesis: str) -> dict[str, float]:
    import jiwer

    raw_wer = jiwer.wer(reference, hypothesis) if reference.strip() else 0.0
    raw_cer = jiwer.cer(reference, hypothesis) if reference.strip() else 0.0

    # Normalised: collapses hamza/alef/ta-marbuta/tatweel/diacritic variants, so
    # the score reflects recognition rather than orthographic convention.
    norm_ref = normalize_arabic(reference.lower())
    norm_hyp = normalize_arabic(hypothesis.lower())
    norm_wer = jiwer.wer(norm_ref, norm_hyp) if norm_ref.strip() else 0.0
    norm_cer = jiwer.cer(norm_ref, norm_hyp) if norm_ref.strip() else 0.0

    return {
        "wer": round(raw_wer, 4),
        "cer": round(raw_cer, 4),
        "wer_normalised": round(norm_wer, 4),
        "cer_normalised": round(norm_cer, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ASR")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model", default=None, help="override the profile's ASR model")
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--out", type=Path, default=Path("data/eval/results/asr.json"))
    args = parser.parse_args()

    from te_assistant.config import get_settings
    from te_assistant.speech import asr as asr_module

    settings = get_settings()
    if args.model:
        # Rebuild the cached model with the override so a benchmark run does not
        # silently measure whatever the profile happened to load first.
        from faster_whisper import WhisperModel

        asr_module._MODEL = WhisperModel(  # noqa: SLF001 - benchmark override
            args.model,
            device="cuda" if settings.is_gpu else "cpu",
            compute_type=args.compute_type or settings.slots.asr_compute_type,
        )
        model_name = args.model
    else:
        model_name = settings.slots.asr

    items = load_manifest(args.manifest)
    rows: list[dict] = []

    for item in items:
        audio_path = args.manifest.parent / item["audio"]
        if not audio_path.exists():
            print(f"  ! missing audio: {audio_path}")
            continue

        started = time.perf_counter()
        result = asr_module.transcribe(audio_path)
        elapsed = time.perf_counter() - started

        reference = item.get("reference", "")
        scores = score_pair(reference, result.text)
        rtf = elapsed / result.duration_seconds if result.duration_seconds else None

        rows.append({
            "audio": item["audio"],
            "dialect": item.get("dialect", "unknown"),
            "condition": item.get("condition", "clean"),
            "reference": reference,
            "hypothesis": result.text,
            "confidence": result.confidence,
            "filtered_hallucination": result.filtered_hallucination,
            "needed_clarification": asr_module.needs_clarification(result),
            "latency_s": round(elapsed, 2),
            # Real-time factor: <1.0 means faster than the audio is long, which
            # is the threshold for a conversational system.
            "rtf": round(rtf, 2) if rtf else None,
            **scores,
        })

    if not rows:
        raise SystemExit("no clips were evaluated")

    def mean(key: str, subset: list[dict]) -> float:
        values = [r[key] for r in subset if r.get(key) is not None]
        return round(statistics.mean(values), 4) if values else 0.0

    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_dialect: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_dialect[row["dialect"]].append(row)

    report = {
        "model": model_name,
        "profile": settings.profile.value,
        "n": len(rows),
        "overall": {
            "wer": mean("wer", rows),
            "wer_normalised": mean("wer_normalised", rows),
            "cer_normalised": mean("cer_normalised", rows),
            "mean_rtf": mean("rtf", rows),
            "hallucinations_filtered": sum(r["filtered_hallucination"] for r in rows),
            "clarifications_requested": sum(r["needed_clarification"] for r in rows),
        },
        "by_condition": {
            name: {"n": len(subset), "wer_normalised": mean("wer_normalised", subset)}
            for name, subset in by_condition.items()
        },
        "by_dialect": {
            name: {"n": len(subset), "wer_normalised": mean("wer_normalised", subset)}
            for name, subset in by_dialect.items()
        },
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = report["overall"]
    print(f"\nmodel : {model_name}  ({settings.profile.value})")
    print(f"clips : {report['n']}\n")
    print(f"  WER (raw)        : {overall['wer']:.1%}")
    print(f"  WER (normalised) : {overall['wer_normalised']:.1%}")
    print(f"  CER (normalised) : {overall['cer_normalised']:.1%}")
    print(f"  mean RTF         : {overall['mean_rtf']:.2f}")
    print(f"  hallucinations   : {overall['hallucinations_filtered']} filtered")
    print(f"  clarifications   : {overall['clarifications_requested']} requested\n")

    for name, stats in sorted(report["by_condition"].items()):
        print(f"    {name:10} n={stats['n']:<3} WER={stats['wer_normalised']:.1%}")

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
