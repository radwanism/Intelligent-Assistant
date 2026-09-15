"""Text-to-speech via Piper (ONNX).

A.2 is explicit that a spoken answer must **always** be accompanied by text —
never voice-only. That rule is enforced in the API layer, not here; this module
only produces audio when asked.

Piper is the right default for this system rather than a compromise:
  * it is ONNX, so it runs on CPU without pulling PyTorch, which is what keeps
    the on-premises install at ~1.5 GB;
  * it synthesises far faster than real time on four cores, so TTS is a rounding
    error in the latency budget rather than a second bottleneck after the LLM;
  * voices are ~60 MB each, so shipping both Arabic and English is cheap.

The Egyptian-dialect Qwen3-TTS fine-tune would sound better, but it is a 1.7B
model — on the CPU profile it would dominate the latency budget, and its 68
monthly downloads make it unverified. It is a benchmark candidate, not a default.

Citation markers are stripped before synthesis. Reading "[1]" aloud is noise; the
citations belong in the text panel, which is always shown anyway.
"""

from __future__ import annotations

import logging
import re
import threading
import wave
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..schemas import Language

log = logging.getLogger(__name__)

PIPER_VOICES_REPO = "rhasspy/piper-voices"

# Where each voice lives inside the Hub repo.
VOICE_PATHS: dict[str, str] = {
    "ar_JO-kareem-medium": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
}

CITATION_MARKER = re.compile(r"\[\d{1,2}\]")
MARKDOWN_NOISE = re.compile(r"[*_`#|]+")
URL_RE = re.compile(r"https?://\S+")

MAX_TTS_CHARS = 1200

_VOICES: dict[str, Any] = {}
_LOCK = threading.RLock()


def speakable(text: str) -> str:
    """Strip anything that should be read with the eyes, not the ears."""
    cleaned = CITATION_MARKER.sub("", text)
    cleaned = URL_RE.sub("", cleaned)
    cleaned = MARKDOWN_NOISE.sub(" ", cleaned)
    # Drop a trailing "Sources:" block if it survived.
    cleaned = re.split(r"\n\s*(?:المصادر|Sources)\s*:", cleaned)[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_TTS_CHARS]


def voice_for(language: Language, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if language is Language.EN:
        return settings.slots.tts_voice_en
    # Arabic, Egyptian, mixed and unknown all get the Arabic voice: for
    # code-switched text the Arabic voice handles embedded Latin words far
    # better than the English voice handles Arabic script.
    return settings.slots.tts_voice_ar


def _resolve_voice_files(voice_name: str) -> tuple[Path, Path]:
    """Locate the .onnx and .onnx.json for a voice, fetching once if needed."""
    import os

    local_dir = os.getenv("TE_MODELS_DIR")
    if local_dir:
        model = Path(local_dir) / f"{voice_name}.onnx"
        config = Path(local_dir) / f"{voice_name}.onnx.json"
        if model.exists() and config.exists():
            return model, config

    from huggingface_hub import hf_hub_download

    relative = VOICE_PATHS.get(voice_name)
    if relative is None:
        raise ValueError(f"unknown Piper voice: {voice_name}")

    model = Path(hf_hub_download(PIPER_VOICES_REPO, relative))
    config = Path(hf_hub_download(PIPER_VOICES_REPO, relative + ".json"))
    return model, config


def get_voice(voice_name: str) -> Any:
    if voice_name in _VOICES:
        return _VOICES[voice_name]
    with _LOCK:
        if voice_name not in _VOICES:
            from piper import PiperVoice

            model, config = _resolve_voice_files(voice_name)
            _VOICES[voice_name] = PiperVoice.load(str(model), config_path=str(config))
            log.info("Piper voice loaded: %s", voice_name)
    return _VOICES[voice_name]


def release_voices() -> None:
    """Drop loaded voices (GPU profile lazy slot, see config.ModelSlots)."""
    with _LOCK:
        _VOICES.clear()


def synthesize(
    text: str,
    output_path: Path,
    *,
    language: Language = Language.AR,
    settings: Settings | None = None,
) -> Path | None:
    """Render `text` to a WAV file. Returns None if there was nothing to say."""
    settings = settings or get_settings()
    content = speakable(text)
    if not content:
        return None

    voice_name = voice_for(language, settings)
    try:
        voice = get_voice(voice_name)
    except Exception:
        # TTS is additive — the text answer is always shown — so a voice that
        # fails to load degrades the response, it does not fail the request.
        log.exception("could not load voice %s; skipping synthesis", voice_name)
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(output_path), "wb") as wav:
            voice.synthesize_wav(content, wav)
    except Exception:
        log.exception("synthesis failed for voice %s", voice_name)
        return None

    return output_path if output_path.exists() else None
