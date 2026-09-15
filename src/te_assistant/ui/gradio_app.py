"""Gradio interface (deliverable A.5).

Requirements this screen implements literally:

  * voice input and text chat in one place (A.2, A.4);
  * **text is always displayed**, including for voice turns — the audio player is
    additive and never the only output (A.2);
  * source citations shown for every answer (A.2);
  * document upload, scoped to the session (A.2, B.3);
  * chat history (A.2);
  * audio playback for voice responses (A.2).

It is a thin client on purpose: it holds no models and talks to `core` over HTTP,
so a backend restart does not take the page down, and a missing `speech` service
just hides the microphone.

Gradio rather than Streamlit because the demo runs on Colab: `launch(share=True)`
produces a public URL with no tunnel setup, and the microphone and audio
components are first-class. A.5 permits either.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from ..config import get_settings

log = logging.getLogger(__name__)

settings = get_settings()
CORE_URL = os.getenv("TE_CORE_URL", settings.core_url)
SPEECH_URL = os.getenv("TE_SPEECH_URL", settings.speech_url)

REQUEST_TIMEOUT = 180.0  # CPU generation is slow; a short timeout would kill valid turns


# --------------------------------------------------------------------------
# Backend calls
# --------------------------------------------------------------------------
def _client() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT)


def core_health() -> dict[str, Any]:
    try:
        with _client() as client:
            return client.get(f"{CORE_URL}/health").json()
    except Exception as exc:
        log.warning("core health check failed: %s", exc)
        return {"status": "down", "speech_available": False}


def new_session() -> str:
    with _client() as client:
        return client.post(f"{CORE_URL}/session").json()["session_id"]


def transcribe(audio_path: str) -> dict[str, Any]:
    with _client() as client, open(audio_path, "rb") as handle:
        response = client.post(
            f"{SPEECH_URL}/transcribe", files={"audio": (Path(audio_path).name, handle)}
        )
        response.raise_for_status()
        return response.json()


def synthesize(text: str, language: str) -> str | None:
    """Returns a path to a WAV, or None when there is nothing to speak."""
    try:
        with _client() as client:
            response = client.post(
                f"{SPEECH_URL}/synthesize", data={"text": text, "language": language}
            )
            if response.status_code == 204:
                return None
            response.raise_for_status()
            out = Path(gr.utils.abspath("."), f"tts_{abs(hash(text)) % 10**8}.wav")
            out.write_bytes(response.content)
            return str(out)
    except Exception as exc:
        # TTS is additive: the text answer is already on screen.
        log.warning("synthesis unavailable: %s", exc)
        return None


def send_chat(message: str, session_id: str, input_mode: str, confidence: float | None
              ) -> dict[str, Any]:
    with _client() as client:
        response = client.post(
            f"{CORE_URL}/chat",
            json={
                "message": message,
                "session_id": session_id,
                "input_mode": input_mode,
                "transcript_confidence": confidence,
            },
        )
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def format_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "_No sources cited for this answer._"
    lines = ["### Sources"]
    for citation in citations:
        where = citation.get("url") or citation.get("doc_name") or ""
        page = f" — page {citation['page']}" if citation.get("page") else ""
        title = citation.get("title") or "Source"
        link = f"[{title}]({where})" if where.startswith("http") else f"**{title}**"
        lines.append(f"{citation['marker']} {link}{page}")
        if snippet := citation.get("snippet"):
            lines.append(f"> {snippet}")
    return "\n\n".join(lines)


def format_trace(record: dict[str, Any]) -> str:
    """Show the B.4 chain actually running — the demo's evidence panel."""
    timings = record.get("timings", {})
    rows = [
        "### Turn trace",
        f"- **Language detected:** `{record.get('language')}`",
        f"- **Intent:** `{record.get('intent')}`",
        f"- **Guardrail:** {'🛑 blocked' if record.get('blocked_by_guardrail') else '✅ passed'}",
        f"- **PII masked:** {record.get('pii_types_masked') or 'none'}",
        f"- **Grounded:** {'yes' if record.get('grounded') else 'no'}",
        f"- **Resolution:** `{record.get('resolution')}`",
    ]
    if action := record.get("action"):
        status = "executed" if action.get("executed") else f"refused — {action.get('reason')}"
        rows.append(f"- **Action:** `{action.get('intent')}` → {status}")
    if frustration := record.get("frustration"):
        if frustration.get("score"):
            rows.append(
                f"- **Frustration:** {frustration['score']} "
                f"({', '.join(frustration.get('reasons', []))})"
            )
    stages = ", ".join(
        f"{key.removesuffix('_ms')} {value:.0f}ms"
        for key, value in timings.items()
        if value and key != "total_ms"
    )
    rows.append(f"- **Latency:** {timings.get('total_ms', 0):.0f} ms total ({stages})")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------
def on_text(message: str, history: list[dict[str, str]], session_id: str):
    if not message or not message.strip():
        return history, "", "", "", None, session_id

    if not session_id:
        session_id = new_session()

    history = (history or []) + [{"role": "user", "content": message}]
    try:
        result = send_chat(message, session_id, "text", None)
    except Exception as exc:
        history.append(
            {"role": "assistant", "content": f"⚠️ The assistant is unavailable: {exc}"}
        )
        return history, "", "", "", None, session_id

    history.append({"role": "assistant", "content": result["answer"]})
    # Text turns produce no audio (A.4: no needless TTS).
    return (
        history,
        "",
        format_citations(result.get("citations", [])),
        format_trace(result.get("record", {})),
        None,
        session_id,
    )


def on_voice(audio_path: str | None, history: list[dict[str, str]], session_id: str):
    if not audio_path:
        return history, "", "", None, session_id

    if not session_id:
        session_id = new_session()

    try:
        asr = transcribe(audio_path)
    except Exception as exc:
        history = (history or []) + [
            {"role": "assistant", "content": f"⚠️ Speech service unavailable: {exc}"}
        ]
        return history, "", "", None, session_id

    transcript = asr.get("text", "").strip()
    if not transcript:
        message = (
            "معلش، الصوت مش واضح أوي. ممكن تعيد السؤال تاني؟"
            if asr.get("language", "ar").startswith("ar")
            else "Sorry, I couldn't hear that clearly. Could you say it again?"
        )
        history = (history or []) + [{"role": "assistant", "content": message}]
        return history, "", "", None, session_id

    # The transcript is shown so the user can see what was heard — essential on
    # dialect audio, where a misrecognition otherwise looks like a bad answer.
    history = (history or []) + [{"role": "user", "content": f"🎤 {transcript}"}]

    try:
        result = send_chat(transcript, session_id, "voice", asr.get("confidence"))
    except Exception as exc:
        history.append(
            {"role": "assistant", "content": f"⚠️ The assistant is unavailable: {exc}"}
        )
        return history, "", "", None, session_id

    answer = result["answer"]
    history.append({"role": "assistant", "content": answer})

    # A.2: voice is additive. The text above is already displayed; if synthesis
    # fails the turn still succeeded.
    audio_out = synthesize(answer, result.get("language", "ar"))

    return (
        history,
        format_citations(result.get("citations", [])),
        format_trace(result.get("record", {})),
        audio_out,
        session_id,
    )


def on_upload(files: list[str] | None, session_id: str):
    if not files:
        return "No file selected.", session_id
    if not session_id:
        session_id = new_session()

    messages: list[str] = []
    for path in files:
        try:
            with _client() as client, open(path, "rb") as handle:
                response = client.post(
                    f"{CORE_URL}/documents",
                    data={"session_id": session_id},
                    files={"file": (Path(path).name, handle)},
                )
            if response.status_code != 200:
                detail = response.json().get("detail", response.text)
                messages.append(f"❌ **{Path(path).name}** — {detail}")
                continue
            result = response.json()
            ocr = " (OCR)" if result.get("used_ocr") else ""
            messages.append(
                f"✅ **{result['doc_name']}** — {result['pages']} page(s), "
                f"{result['chunks']} chunk(s){ocr}. You can now ask about it."
            )
        except Exception as exc:
            messages.append(f"❌ **{Path(path).name}** — {exc}")
    return "\n\n".join(messages), session_id


def on_insights(session_id: str) -> str:
    if not session_id:
        return "_No session yet._"
    try:
        with _client() as client:
            return client.get(f"{CORE_URL}/insights/{session_id}").json()["markdown"]
    except Exception as exc:
        return f"_Insights unavailable: {exc}_"


def on_end_session(session_id: str):
    """Ends the session and evicts its documents (B.8)."""
    if session_id:
        try:
            with _client() as client:
                client.delete(f"{CORE_URL}/session/{session_id}")
        except Exception as exc:
            log.warning("could not end session: %s", exc)
    return [], "", "", None, "", "Session ended — uploaded documents were deleted."


# --------------------------------------------------------------------------
MIN_GRADIO = (4, 44)


def _gradio_version() -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in gr.__version__.split(".")[:2])
    except Exception:
        return (99, 0)  # unparseable: assume new rather than block startup


# Chatbot arguments we would *like*, filtered against what this Gradio accepts.
OPTIONAL_CHATBOT_KWARGS: dict[str, object] = {
    # 4.44-5.x: opt in to {'role','content'} history. Removed in 6, where that
    # format is the only one.
    "type": "messages",
    # Present through 5.x, removed in 6 (copy is built into the message UI).
    "show_copy_button": True,
}


def _chatbot_kwargs() -> dict[str, object]:
    """Build Chatbot arguments valid for the installed Gradio.

    `type="messages"` has a narrow lifetime:

        < 4.44   absent — history is [user, bot] tuples, incompatible with us
        4.44-5.x present, and must be passed to opt in to message dicts
        >= 6     REMOVED — message dicts are the only format, and passing it
                 raises `unexpected keyword argument 'type'`

    So the identical TypeError means "too old" or "too new" depending which side
    you are on — which is why this filters by signature rather than comparing
    version numbers. Colab ships its own Gradio and satisfies the dependency at
    import time, so the mismatch only ever appears at launch, and a version
    range pinned in pyproject.toml would not have prevented it.
    """
    params = inspect.signature(gr.Chatbot.__init__).parameters
    kwargs = {k: v for k, v in OPTIONAL_CHATBOT_KWARGS.items() if k in params}

    # `type` missing can mean either era; only the old one is unusable.
    if "type" not in params and _gradio_version() < MIN_GRADIO:
        raise RuntimeError(
            f"Gradio {gr.__version__} is too old — this UI needs "
            f">= {'.'.join(map(str, MIN_GRADIO))}, where chat history is a list of "
            "{'role', 'content'} dicts rather than tuples.\n"
            "    pip install -U 'gradio>=5.9'\n"
            "then RESTART the runtime — an already-imported Gradio is not replaced."
        )

    return kwargs


def build_ui() -> gr.Blocks:
    health = core_health()
    speech_available = bool(health.get("speech_available"))

    with gr.Blocks(title="Telecom Egypt — Intelligent Assistant", fill_height=True) as demo:
        session_state = gr.State("")

        gr.Markdown(
            "# 🇪🇬 Telecom Egypt — Intelligent Assistant\n"
            "Ask in **Arabic, English or Egyptian dialect**, by voice or text. "
            "Every answer is grounded in te.eg content or your uploaded documents, "
            "with sources shown."
        )

        if not speech_available:
            gr.Markdown(
                "> ⚠️ **Speech service is offline** — voice input is disabled. "
                "Text chat works normally."
            )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation", height=460, **_chatbot_kwargs()
                )
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="اسأل عن باقات الإنترنت… / Ask about internet plans…",
                        show_label=False, scale=8, lines=1,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)

                if speech_available:
                    microphone = gr.Audio(
                        sources=["microphone"], type="filepath",
                        label="🎤 Ask by voice", format="wav",
                    )
                    # Always rendered alongside the text answer, never instead of it.
                    audio_output = gr.Audio(
                        label="🔊 Spoken answer", autoplay=True, interactive=False
                    )
                else:
                    microphone = None
                    audio_output = gr.Audio(visible=False)

            with gr.Column(scale=2):
                with gr.Tab("Sources"):
                    citations_panel = gr.Markdown("_Ask something to see its sources._")
                with gr.Tab("Your documents"):
                    uploader = gr.File(
                        label="Upload PDF, DOCX, TXT or an image",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"],
                        type="filepath",
                    )
                    upload_status = gr.Markdown(
                        "_Uploads are private to this session and deleted when it ends._"
                    )
                with gr.Tab("Trace"):
                    trace_panel = gr.Markdown(
                        "_Shows the guardrail → PII → intent → permission chain per turn._"
                    )
                with gr.Tab("Insights"):
                    insights_panel = gr.Markdown("_No turns yet._")
                    refresh_insights = gr.Button("Refresh")
                    end_button = gr.Button("End session & delete my documents", variant="stop")

        text_outputs = [
            chatbot, textbox, citations_panel, trace_panel, audio_output, session_state
        ]
        textbox.submit(on_text, [textbox, chatbot, session_state], text_outputs)
        send_button.click(on_text, [textbox, chatbot, session_state], text_outputs)

        if microphone is not None:
            microphone.stop_recording(
                on_voice,
                [microphone, chatbot, session_state],
                [chatbot, citations_panel, trace_panel, audio_output, session_state],
            )

        uploader.upload(
            on_upload, [uploader, session_state], [upload_status, session_state]
        )
        refresh_insights.click(on_insights, [session_state], [insights_panel])
        end_button.click(
            on_end_session,
            [session_state],          # it needs the id to evict the right session
            [chatbot, citations_panel, trace_panel, audio_output, session_state,
             upload_status],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Telecom Egypt Assistant UI")
    parser.add_argument(
        "--share", action="store_true", help="public URL (used on Colab)"
    )
    parser.add_argument("--port", type=int, default=settings.ui_port)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_ui().launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, show_api=False
    )


if __name__ == "__main__":
    main()
