"""Pre-fetch every model the active profile needs.

Two uses:

1. **Warming a machine before a demo** so the first request is not a 2 GB
   download. On Colab this is the step that takes a minute or two instead of the
   hour it takes on a slow domestic connection.
2. **Building an offline bundle** for an air-gapped on-premises install. Run with
   `--dest models/`, copy the directory across, and set `TE_MODELS_DIR` on the
   target. Nothing in the serving path then touches the network.

Downloads are resumable — huggingface_hub caches by content hash, so a killed
run continues rather than restarting.

    python scripts/fetch_models.py                 # active profile, HF cache
    python scripts/fetch_models.py --dest models/  # flat dir for offline use
    python scripts/fetch_models.py --profile gpu-colab
    python scripts/fetch_models.py --skip llm      # everything except the LLM
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from te_assistant.config import CPU_SLOTS, GPU_SLOTS, ModelSlots, Profile
from te_assistant.speech.tts import PIPER_VOICES_REPO, VOICE_PATHS


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def fetch_llm(slots: ModelSlots, dest: Path | None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    if slots.llm_file:
        print(f"  LLM  {slots.llm_repo} :: {slots.llm_file}")
        path = Path(hf_hub_download(slots.llm_repo, slots.llm_file))
        print(f"       {_human(path.stat().st_size)}")
        if dest:
            shutil.copy2(path, dest / path.name)
    else:
        # AWQ/transformers repos are many files; snapshot_download handles them.
        print(f"  LLM  {slots.llm_repo} (full snapshot)")
        snapshot_download(slots.llm_repo)


def fetch_asr(slots: ModelSlots) -> None:
    """faster-whisper resolves and caches the CTranslate2 conversion itself."""
    print(f"  ASR  faster-whisper {slots.asr} ({slots.asr_compute_type})")
    from faster_whisper import WhisperModel

    WhisperModel(slots.asr, device="cpu", compute_type="int8")


def fetch_embedder(slots: ModelSlots, profile: Profile) -> None:
    print(f"  EMB  {slots.embedder}")
    if profile is Profile.GPU_COLAB:
        from huggingface_hub import snapshot_download

        snapshot_download(slots.embedder)
    else:
        from fastembed import TextEmbedding

        TextEmbedding(model_name=slots.embedder)


def fetch_reranker(slots: ModelSlots) -> None:
    if not slots.reranker:
        print("  RNK  (none on this profile)")
        return
    print(f"  RNK  {slots.reranker}")
    from huggingface_hub import snapshot_download

    snapshot_download(slots.reranker)


def fetch_tts(slots: ModelSlots, dest: Path | None) -> None:
    from huggingface_hub import hf_hub_download

    for voice in (slots.tts_voice_ar, slots.tts_voice_en):
        relative = VOICE_PATHS.get(voice)
        if relative is None:
            print(f"  TTS  ! unknown voice {voice}, skipping")
            continue
        print(f"  TTS  {voice}")
        model = Path(hf_hub_download(PIPER_VOICES_REPO, relative))
        config = Path(hf_hub_download(PIPER_VOICES_REPO, relative + ".json"))
        if dest:
            shutil.copy2(model, dest / f"{voice}.onnx")
            shutil.copy2(config, dest / f"{voice}.onnx.json")


def fetch_ocr() -> None:
    """RapidOCR ships its ONNX weights inside the wheel; loading verifies them."""
    print("  OCR  rapidocr-onnxruntime (bundled)")
    from rapidocr_onnxruntime import RapidOCR

    RapidOCR()


STEPS = ("llm", "asr", "embedder", "reranker", "tts", "ocr")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-fetch models")
    parser.add_argument("--profile", choices=[p.value for p in Profile], default=None)
    parser.add_argument("--dest", type=Path, default=None, help="also copy into this dir")
    parser.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    parser.add_argument("--only", nargs="*", default=None, choices=STEPS)
    args = parser.parse_args()

    profile = Profile(args.profile) if args.profile else None
    if profile is None:
        from te_assistant.config import detect_profile

        profile = detect_profile()
    slots = GPU_SLOTS if profile is Profile.GPU_COLAB else CPU_SLOTS

    wanted = set(args.only or STEPS) - set(args.skip)

    if args.dest:
        args.dest.mkdir(parents=True, exist_ok=True)

    print(f"\nprofile: {profile.value}")
    print(f"target : {args.dest or 'huggingface cache'}\n")

    started = time.time()
    actions = {
        "llm": lambda: fetch_llm(slots, args.dest),
        "asr": lambda: fetch_asr(slots),
        "embedder": lambda: fetch_embedder(slots, profile),
        "reranker": lambda: fetch_reranker(slots),
        "tts": lambda: fetch_tts(slots, args.dest),
        "ocr": fetch_ocr,
    }

    failures: list[str] = []
    for name in STEPS:
        if name not in wanted:
            continue
        try:
            actions[name]()
        except Exception as exc:
            # Keep going: a missing reranker should not stop the LLM downloading.
            print(f"       ! failed: {exc}")
            failures.append(name)

    print(f"\ndone in {time.time() - started:.0f}s")
    if failures:
        print(f"failed: {', '.join(failures)}")
        return 1
    if args.dest:
        print(f"\nOffline install: set TE_MODELS_DIR={args.dest.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
