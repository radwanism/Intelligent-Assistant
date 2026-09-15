"""Speech service — ASR + TTS on its own port.

This is the one component that genuinely earns a separate process (brief B.1,
D.4). The justification is specific rather than "microservices are good":

  * It holds the heaviest models in the system and has the slowest cold start,
    so restarting it must not restart retrieval and the LLM.
  * It is the most likely thing to run out of memory — long audio, a large
    Whisper model, and on the GPU profile it shares VRAM with everything else.
  * Speech is optional to the product. Text chat is not. Putting them in one
    process would mean an OOM while transcribing a two-minute voice note takes
    down text chat for every other concurrent user.

So the contract is: **if this service is down, the assistant still works.** The
core service treats speech as optional and the UI hides the microphone. That
degradation is asserted in `tests/test_degradation.py`.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import get_settings
from ..schemas import Language, TranscribeResponse
from . import asr, tts

log = logging.getLogger(__name__)

MAX_AUDIO_MB = 25

app = FastAPI(title="TE Assistant — Speech Service", version="0.1.0")

_audio_dir = Path(tempfile.gettempdir()) / "te_assistant_audio"
_audio_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness only — deliberately does not load the models.

    A health check that triggers a 30-second model load would make the service
    look unhealthy precisely while it is starting up correctly.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "profile": settings.profile.value,
        "asr_model": settings.slots.asr,
        "asr_loaded": asr._MODEL is not None,  # noqa: SLF001 - diagnostic
        "voices_loaded": list(tts._VOICES),    # noqa: SLF001 - diagnostic
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language_hint: str | None = Form(default=None),
) -> TranscribeResponse:
    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(payload) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"audio exceeds {MAX_AUDIO_MB} MB")

    suffix = Path(audio.filename or "clip.wav").suffix or ".wav"
    temp_path = _audio_dir / f"in-{uuid.uuid4().hex}{suffix}"
    temp_path.write_bytes(payload)

    started = time.perf_counter()
    try:
        result = asr.transcribe(temp_path, language_hint=language_hint or None)
    finally:
        temp_path.unlink(missing_ok=True)

    log.info(
        "transcribed %.1fs of audio in %.2fs (conf=%.2f, lang=%s)",
        result.duration_seconds,
        time.perf_counter() - started,
        result.confidence,
        result.language.value,
    )
    return result


@app.post("/synthesize")
async def synthesize(text: str = Form(...), language: str = Form(default="ar")):
    """Render text to speech. 204 when there is nothing speakable.

    204 rather than an error: "no audio" is a normal outcome (an answer that is
    only a citation list, for instance) and the caller always has the text.
    """
    try:
        lang = Language(language)
    except ValueError:
        lang = Language.AR

    output_path = _audio_dir / f"out-{uuid.uuid4().hex}.wav"
    started = time.perf_counter()
    produced = tts.synthesize(text, output_path, language=lang)
    if produced is None:
        from fastapi import Response

        return Response(status_code=204)

    log.info("synthesised %d chars in %.2fs", len(text), time.perf_counter() - started)
    return FileResponse(
        produced, media_type="audio/wav", filename="response.wav",
        background=_cleanup_task(produced),
    )


def _cleanup_task(path: Path):
    from starlette.background import BackgroundTask

    def _remove() -> None:
        path.unlink(missing_ok=True)

    return BackgroundTask(_remove)


@app.post("/warmup")
def warmup() -> dict[str, str]:
    """Preload models so the first real request is not the slow one."""
    asr.get_model()
    settings = get_settings()
    tts.get_voice(settings.slots.tts_voice_ar)
    return {"status": "warm"}


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = get_settings()
    uvicorn.run(
        app, host=settings.speech_host, port=settings.speech_port, log_level="info"
    )


if __name__ == "__main__":
    main()
