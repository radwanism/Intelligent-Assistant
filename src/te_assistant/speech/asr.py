"""Automatic speech recognition, tuned for noisy Egyptian-dialect audio (A.1).

The case study calls out **noisy recordings** and **Egyptian dialect** as the
real-world conditions to handle, so this module is not a thin Whisper wrapper.
Four things happen around the model:

1. **VAD** trims silence and non-speech before transcription. This is the single
   most effective anti-hallucination measure available: Whisper's failure mode
   on silence is not to return nothing, it is to confidently emit text it
   learned from training-data captions.

2. **A dialect-biased prompt.** Whisper drifts toward MSA on Egyptian input
   because MSA dominates its Arabic training data. An `initial_prompt` written
   in Egyptian Arabic biases decoding toward preserving the dialect, which is
   what the user actually said and what retrieval should see.

3. **Hallucination filtering** on the output (see `halluc_filter`).

4. **A confidence gate.** A low-confidence transcript triggers a clarification
   turn rather than a confidently wrong answer to a misheard question.

Model choice is per profile: `small` int8 on CPU, `large-v3` fp16 on GPU. The
Egyptian-specific fine-tunes (Nawah-ASR-118M, whisper-large-v3-egyptian) are
benchmark candidates in `scripts/bench_models.py` rather than defaults — at 43
and 287 monthly downloads they are unverified, and the brief's own evaluation
requirement is the right way to decide, not a coin flip on the critical path.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..retrieval.normalize_ar import detect_language
from ..schemas import Language, TranscribeResponse
from .halluc_filter import filter_transcript

log = logging.getLogger(__name__)

# Written in Egyptian Arabic on purpose: it is a decoding hint, and telecom
# vocabulary in the user's own register is what we want the model to favour.
EGYPTIAN_PROMPT = "ازيك، عايز أسأل عن باقات الإنترنت والموبايل من WE المصرية للاتصالات."

# Below this, ask the user to repeat rather than answer a guess.
MIN_CONFIDENCE = 0.45

_MODEL: Any | None = None
_LOCK = threading.Lock()


def get_model(settings: Settings | None = None) -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    settings = settings or get_settings()
    with _LOCK:
        if _MODEL is None:
            from faster_whisper import WhisperModel

            slots = settings.slots
            device = "cuda" if settings.is_gpu else "cpu"
            _MODEL = WhisperModel(
                slots.asr,
                device=device,
                compute_type=slots.asr_compute_type,
                cpu_threads=0,  # 0 lets CTranslate2 pick, which it does well
            )
            log.info(
                "ASR loaded: %s on %s (%s)", slots.asr, device, slots.asr_compute_type
            )
    return _MODEL


def _confidence_from(segments: list[Any]) -> float:
    """Turn per-segment average log-probability into a 0..1 confidence.

    Weighted by segment duration so a long clear utterance is not dragged down
    by a short uncertain fragment at the end.
    """
    if not segments:
        return 0.0
    total_weight = 0.0
    total = 0.0
    for segment in segments:
        duration = max(float(segment.end) - float(segment.start), 0.01)
        logprob = float(getattr(segment, "avg_logprob", -1.0))
        total += math.exp(logprob) * duration
        total_weight += duration
    return round(min(max(total / max(total_weight, 0.01), 0.0), 1.0), 3)


def transcribe(
    audio_path: Path,
    *,
    settings: Settings | None = None,
    language_hint: str | None = None,
) -> TranscribeResponse:
    """Transcribe one audio file."""
    settings = settings or get_settings()
    model = get_model(settings)

    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language_hint,          # None => autodetect (handles AR/EN)
            task="transcribe",               # never "translate": no English pivot
            beam_size=5 if settings.is_gpu else 1,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            initial_prompt=EGYPTIAN_PROMPT,
            condition_on_previous_text=False,  # stops repetition loops on noise
            temperature=[0.0, 0.2, 0.4],       # fallback ladder on decode failure
            no_speech_threshold=0.6,
        )
        segments = list(segments_iter)
    except Exception:
        log.exception("transcription failed for %s", audio_path.name)
        return TranscribeResponse(text="", language=Language.UNKNOWN, confidence=0.0)

    raw_text = " ".join(segment.text.strip() for segment in segments).strip()
    confidence = _confidence_from(segments)

    text, was_hallucination = filter_transcript(raw_text)
    if was_hallucination:
        log.info("hallucination filter removed: %r", raw_text[:80])
        confidence = min(confidence, 0.2)

    # Whisper reports ar/en; our dialect distinction comes from the text itself.
    language = detect_language(text) if text else Language.UNKNOWN
    if language is Language.UNKNOWN and getattr(info, "language", None) == "en":
        language = Language.EN

    return TranscribeResponse(
        text=text,
        language=language,
        confidence=confidence,
        duration_seconds=round(float(getattr(info, "duration", 0.0)), 2),
        filtered_hallucination=was_hallucination,
    )


def needs_clarification(result: TranscribeResponse) -> bool:
    return (not result.text.strip()) or result.confidence < MIN_CONFIDENCE


CLARIFY_AR = "معلش، الصوت مش واضح أوي. ممكن تعيد السؤال تاني؟"
CLARIFY_EN = "Sorry, I couldn't hear that clearly. Could you say it again?"


def clarification_message(language: Language) -> str:
    return CLARIFY_AR if (language.is_arabic or language is Language.UNKNOWN) else CLARIFY_EN
