"""UI behaviour around session expiry and spoken answers.

The handlers are driven for real; only the HTTP calls to core and speech are
stubbed, so these assert what the *user* ends up with, not how it is wired.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from te_assistant.ui import gradio_app as ui


def _chat_result(session_id: str, answer: str = "An answer [1].") -> dict:
    return {
        "answer": answer, "citations": [], "language": "en",
        "record": {"session_id": session_id, "timings": {}},
    }


# --------------------------------------------------------------------------
# adopt_session
# --------------------------------------------------------------------------
def test_the_servers_session_id_is_adopted() -> None:
    history: list[dict] = []
    assert ui.adopt_session(history, "old", _chat_result("new")) == "new"


def test_a_changed_session_tells_the_user_once() -> None:
    history: list[dict] = []
    ui.adopt_session(history, "old", _chat_result("new"))
    assert len(history) == 1
    assert "expired" in history[0]["content"]
    assert "انتهت" in history[0]["content"], "the notice must be bilingual"


def test_an_unchanged_session_says_nothing() -> None:
    history: list[dict] = []
    assert ui.adopt_session(history, "same", _chat_result("same")) == "same"
    assert history == []


def test_the_first_turn_of_a_conversation_is_not_reported_as_a_renewal() -> None:
    history: list[dict] = []
    ui.adopt_session(history, "", _chat_result("fresh"))
    assert history == []


def test_a_response_without_a_record_keeps_the_current_id() -> None:
    history: list[dict] = []
    assert ui.adopt_session(history, "keep", {"answer": "x"}) == "keep"
    assert history == []


# --------------------------------------------------------------------------
# on_text / on_voice use it
# --------------------------------------------------------------------------
def test_text_turn_after_expiry_switches_to_the_new_session(monkeypatch) -> None:
    monkeypatch.setattr(ui, "send_chat", lambda *a, **k: _chat_result("new-session"))
    history, _, _, _, _, session_id = ui.on_text("hello", [], "expired-session")
    assert session_id == "new-session"
    assert any("expired" in m["content"] for m in history)
    assert history[-1]["content"] == "An answer [1]."


def test_text_turn_in_a_live_session_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(ui, "send_chat", lambda *a, **k: _chat_result("live"))
    history, _, _, _, _, session_id = ui.on_text("hello", [], "live")
    assert session_id == "live"
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_voice_turn_after_expiry_switches_to_the_new_session(monkeypatch) -> None:
    heard = {"text": "hi", "confidence": 0.9, "language": "en"}
    monkeypatch.setattr(ui, "transcribe", lambda path: heard)
    monkeypatch.setattr(ui, "send_chat", lambda *a, **k: _chat_result("new-session"))
    monkeypatch.setattr(ui, "synthesize", lambda text, language: None)
    history, _, _, _, session_id = ui.on_voice("clip.wav", [], "expired-session")
    assert session_id == "new-session"
    assert any("expired" in m["content"] for m in history)


# --------------------------------------------------------------------------
# on_upload
# --------------------------------------------------------------------------
class _Response:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}
        self.text = str(self._payload)

    def json(self) -> dict:
        return self._payload


_OK = {"doc_name": "bill.pdf", "pages": 2, "chunks": 5, "used_ocr": False}


def test_upload_to_an_expired_session_renews_and_retries(monkeypatch, tmp_path) -> None:
    sent: list[str] = []
    replies = iter([_Response(404, {"detail": "unknown or expired session"}), _Response(200, _OK)])

    def fake_post(path: str, session_id: str):
        sent.append(session_id)
        return next(replies)

    monkeypatch.setattr(ui, "_post_document", fake_post)
    monkeypatch.setattr(ui, "new_session", lambda: "renewed")
    status, session_id = ui.on_upload([str(tmp_path / "bill.pdf")], "expired")

    assert sent == ["expired", "renewed"]
    assert session_id == "renewed"
    assert "bill.pdf" in status and "expired" in status


def test_upload_to_a_live_session_is_unchanged(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ui, "_post_document", lambda path, sid: _Response(200, _OK))
    monkeypatch.setattr(ui, "new_session", lambda: pytest.fail("must not renew a live session"))
    status, session_id = ui.on_upload([str(tmp_path / "bill.pdf")], "live")
    assert session_id == "live"
    assert "bill.pdf" in status and "expired" not in status


def test_a_real_upload_error_is_still_reported(monkeypatch, tmp_path) -> None:
    rejected = _Response(415, {"detail": "unsupported file type"})
    monkeypatch.setattr(ui, "_post_document", lambda p, s: rejected)
    status, _ = ui.on_upload([str(tmp_path / "x.exe")], "live")
    assert "unsupported file type" in status


def test_the_uploader_offers_every_format_the_backend_accepts() -> None:
    from te_assistant.documents.parse import SUPPORTED_SUFFIXES

    demo = ui.build_ui()
    offered: set[str] = set()
    for block in demo.blocks.values():
        if type(block).__name__ == "File":
            offered = set(block.file_types or [])
    assert SUPPORTED_SUFFIXES <= offered, f"missing in UI: {SUPPORTED_SUFFIXES - offered}"


# --------------------------------------------------------------------------
# synthesize — in memory, no files
# --------------------------------------------------------------------------
def _wav_bytes(rate: int = 22050, seconds: float = 0.1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        tone = np.sin(np.linspace(0, 440, int(rate * seconds))) * 8000
        wav.writeframes(tone.astype("<i2").tobytes())
    return buffer.getvalue()


class _FakeClient:
    def __init__(self, status: int = 200, content: bytes = b"") -> None:
        self._status, self._content = status, content

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def post(self, *a, **k):
        status, content = self._status, self._content

        class _R:
            status_code = status

            def __init__(self) -> None:
                self.content = content

            def raise_for_status(self) -> None:
                if status >= 400:
                    raise RuntimeError(f"HTTP {status}")

        return _R()


def test_synthesis_returns_audio_in_memory_and_writes_no_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui, "_client", lambda: _FakeClient(200, _wav_bytes()))
    result = ui.synthesize("hello", "en")

    assert result is not None
    rate, samples = result
    assert rate == 22050
    assert samples.shape[0] == int(22050 * 0.1)
    assert samples.dtype == np.int16
    assert list(tmp_path.iterdir()) == [], "no audio file may be left in the working directory"


def test_nothing_to_speak_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(ui, "_client", lambda: _FakeClient(204))
    assert ui.synthesize("[1]", "ar") is None


def test_a_speech_failure_degrades_to_no_audio(monkeypatch) -> None:
    """TTS is additive: the text answer is already on screen."""
    monkeypatch.setattr(ui, "_client", lambda: _FakeClient(500))
    assert ui.synthesize("hello", "en") is None


def test_corrupt_audio_degrades_to_no_audio(monkeypatch) -> None:
    monkeypatch.setattr(ui, "_client", lambda: _FakeClient(200, b"not a wav file"))
    assert ui.synthesize("hello", "en") is None


# --------------------------------------------------------------------------
# The degradation contract: speech down => microphone hidden, text chat intact
# --------------------------------------------------------------------------
def _audio_blocks(demo) -> list:
    """Audio components the user can actually see.

    When speech is down the page keeps a hidden placeholder Audio so the output
    wiring stays valid; Gradio gives every Audio the microphone as a default
    source, so only *visible* ones count as "offering" a microphone.
    """
    return [b for b in demo.blocks.values()
            if type(b).__name__ == "Audio" and getattr(b, "visible", True)]


def _markdown_text(demo) -> str:
    return " ".join(str(getattr(b, "value", "")) for b in demo.blocks.values()
                    if type(b).__name__ == "Markdown")


def test_speech_down_hides_the_microphone_and_says_so(monkeypatch) -> None:
    monkeypatch.setattr(ui, "core_health", lambda: {"status": "ok", "speech_available": False})
    demo = ui.build_ui()
    microphones = [a for a in _audio_blocks(demo) if "microphone" in (a.sources or [])]
    assert microphones == [], "no microphone may be offered when speech is down"
    assert "Speech service is offline" in _markdown_text(demo)


def test_speech_down_leaves_text_chat_wired(monkeypatch) -> None:
    monkeypatch.setattr(ui, "core_health", lambda: {"status": "ok", "speech_available": False})
    demo = ui.build_ui()
    names = {getattr(dep, "name", "") or str(dep) for dep in demo.fns.values()}
    assert any("on_text" in n for n in names), "text chat must still be wired"
    assert any("on_upload" in n for n in names), "uploads must still be wired"


def test_speech_up_offers_the_microphone(monkeypatch) -> None:
    monkeypatch.setattr(ui, "core_health", lambda: {"status": "ok", "speech_available": True})
    demo = ui.build_ui()
    assert [a for a in _audio_blocks(demo) if "microphone" in (a.sources or [])]
    assert "Speech service is offline" not in _markdown_text(demo)


def test_core_down_degrades_instead_of_crashing_the_page(monkeypatch) -> None:
    def refuse(*a, **k):
        raise ConnectionError("core unreachable")

    monkeypatch.setattr(ui, "_client", refuse)
    health = ui.core_health()
    assert health["status"] == "down" and health["speech_available"] is False
