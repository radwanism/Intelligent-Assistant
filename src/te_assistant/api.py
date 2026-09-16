"""Core service — retrieval, security, actions, sessions.

Everything that is not speech and not UI lives here, deliberately (brief B.1:
don't split by default). These components share the session store and the vector
index; separating them would buy no failure isolation and would add a network
hop to every retrieval.

The one thing this service treats as optional is `speech`. If that service is
down, `/health` reports it and the UI hides the microphone — text chat is
unaffected. That is the degradation contract from the plan.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import get_settings
from .documents.parse import UnsupportedDocument
from .documents.session_docs import UploadTooLarge, ingest_upload
from .insights.records import Insights, render_markdown, summarize
from .pipeline import Services, handle_turn
from .schemas import ChatRequest, ChatResponse, UploadResponse

log = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 300

_services: Services | None = None


def get_services() -> Services:
    if _services is None:  # pragma: no cover - startup ordering guard
        raise RuntimeError("services not initialised")
    return _services


def _sweep_loop(services: Services, stop: threading.Event) -> None:
    """Evict expired sessions periodically (B.8).

    Without this, uploaded documents and their embeddings would live as long as
    the process — which for a POC left running during a demo day is exactly the
    data-retention problem this case study is supposed to be careful about.
    """
    while not stop.wait(SWEEP_INTERVAL_SECONDS):
        try:
            services.sessions.sweep()
        except Exception:
            log.exception("session sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _services
    settings = get_settings()
    _services = Services.build(settings)
    _services.warmup()

    stop = threading.Event()
    thread = threading.Thread(
        target=_sweep_loop, args=(_services, stop), daemon=True, name="session-sweeper"
    )
    thread.start()

    log.info(
        "core ready: profile=%s, kb_chunks=%d",
        settings.profile.value, _services.retriever.store.kb_size(),
    )
    yield
    stop.set()


app = FastAPI(title="TE Assistant — Core", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Return JSON for unhandled errors, not Starlette's plain-text 500.

    A plain-text body makes `response.json()` raise `JSONDecodeError: Expecting
    value: line 1 column 1`, which tells the caller nothing and hides the real
    exception inside the server log. Every client of this API — the Gradio UI,
    the notebook, curl — then reports the JSON parse failure instead of the
    cause.

    The message is included deliberately. This is an on-premises POC with no
    untrusted callers, and being able to read the failure in the response is
    worth more here than withholding it. A public deployment should log the
    detail and return only the error id.
    """
    error_id = uuid.uuid4().hex[:8]
    log.exception("unhandled error %s on %s %s", error_id, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "error_id": error_id,
            "path": request.url.path,
        },
    )


@app.get("/health")
async def health(services: Services = Depends(get_services)) -> dict[str, object]:
    settings = services.settings
    speech_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.speech_url}/health")
            speech_ok = response.status_code == 200
    except Exception:
        speech_ok = False

    from .gpu import memory_report

    payload: dict[str, object] = {
        "status": "ok",
        "profile": settings.profile.value,
        "kb_chunks": services.retriever.store.kb_size(),
        "bm25_ready": services.retriever.bm25.ready,
        "active_sessions": services.sessions.active_count,
        # The UI reads this to decide whether to show the microphone.
        "speech_available": speech_ok,
    }

    # Surface VRAM here so an impending OOM is visible before a request fails.
    # `other_processes_gib` is the number that matters: PyTorch reports only its
    # own allocations, so a notebook kernel holding 11 GiB is invisible in the
    # traceback and the error reads as though the model is simply too large.
    if gpu := memory_report():
        payload["gpu"] = gpu
        if gpu["free_gib"] < 1.0 and gpu["other_processes_gib"] > 1.0:
            payload["warning"] = (
                f"only {gpu['free_gib']:.2f} GiB VRAM free; "
                f"{gpu['other_processes_gib']:.2f} GiB is held by another process"
            )

    return payload


@app.post("/session")
def create_session(services: Services = Depends(get_services)) -> dict[str, str]:
    session = services.sessions.create()
    return {"session_id": session.session_id}


@app.delete("/session/{session_id}")
def end_session(session_id: str, services: Services = Depends(get_services)) -> dict[str, str]:
    """End a session and evict its documents and embeddings."""
    services.sessions.end(session_id)
    return {"status": "ended"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, services: Services = Depends(get_services)
) -> ChatResponse:
    started = time.perf_counter()
    response = handle_turn(services, request)
    log.info(
        "turn %s: intent=%s grounded=%s %.0fms",
        response.record.turn_id, response.record.intent.value,
        response.record.grounded, (time.perf_counter() - started) * 1000,
    )
    return response


@app.post("/documents", response_model=UploadResponse)
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    services: Services = Depends(get_services),
) -> UploadResponse:
    session = services.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown or expired session")

    payload = await file.read()
    try:
        return ingest_upload(
            session=session,
            filename=file.filename or "upload.bin",
            payload=payload,
            store=services.retriever.store,
            settings=services.settings,
        )
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedDocument as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/insights/{session_id}")
def session_insights(
    session_id: str, services: Services = Depends(get_services)
) -> dict[str, object]:
    records = services.records.for_session(session_id)
    insights: Insights = summarize(records)
    return {
        "session_id": session_id,
        "insights": insights.__dict__,
        "markdown": render_markdown(insights),
    }


@app.get("/metrics")
def metrics(services: Services = Depends(get_services)) -> dict[str, object]:
    """Live latency / quality counters (B.10).

    The figures quoted in the presentation come from here, so they are measured
    on the running system rather than transcribed by hand.
    """
    from .monitoring.metrics import METRICS

    snapshot = METRICS.snapshot()
    snapshot["profile"] = services.settings.profile.value
    snapshot["models"] = {
        "llm": services.settings.slots.llm_repo,
        "embedder": services.settings.slots.embedder,
        "asr": services.settings.slots.asr,
        "reranker": services.settings.slots.reranker,
    }
    return snapshot


@app.get("/metrics/{session_id}")
def session_metrics(
    session_id: str, services: Services = Depends(get_services)
) -> dict[str, object]:
    from .monitoring.metrics import METRICS

    return {"session_id": session_id, "metrics": METRICS.session(session_id)}


@app.get("/handoffs")
def pending_handoffs(services: Services = Depends(get_services)) -> dict[str, object]:
    """What a human agent's queue would read."""
    tickets = services.handoffs.pending()
    return {"count": len(tickets), "tickets": [t.model_dump(mode="json") for t in tickets]}


@app.post("/demo/grant")
def demo_grant(
    session_id: str = Form(...),
    customer_id: str = Form(default="cust-1001"),
    scope: str = Form(default="billing:read"),
    services: Services = Depends(get_services),
) -> dict[str, str]:
    """Bind a demo account to a session.

    Stands in for the authenticated login that a production system would have.
    It exists so the permission check in B.4 is exercised against a real grant
    rather than mocked away — and so the demo can show the *refused* case by
    simply not calling this.
    """
    services.db.grant(session_id, customer_id, scope)
    return {"status": "granted", "customer_id": customer_id, "scope": scope}


@app.get("/audit/{session_id}")
def audit_trail(
    session_id: str, services: Services = Depends(get_services)
) -> dict[str, object]:
    return {"session_id": session_id, "entries": services.db.audit_trail(session_id)}


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = get_settings()
    uvicorn.run(app, host=settings.core_host, port=settings.core_port, log_level="info")


if __name__ == "__main__":
    main()
