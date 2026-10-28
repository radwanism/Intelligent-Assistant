"""The HTTP surface, end to end, with a scripted model.

Everything below goes through FastAPI exactly as the UI and the notebook do —
session, upload, chat, insights — so a problem at the API boundary (a field that
no longer serialises, a dependency that is not wired) shows up here and not in a
demo.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from te_assistant import api
from te_assistant.config import Settings, get_settings
from te_assistant.monitoring.metrics import METRICS
from te_assistant.pipeline import Services

TOKEN = "operator-token"
PHONE = "01198765432"
BILL = (
    "Account statement for the customer. For any billing question about this "
    f"statement please call the registered line {PHONE}. The monthly charge for "
    "the WE Nitro plan is 300 EGP and includes 200 GB of data every month."
)


class StubEmbedder:
    dim = 16

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dim)]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class ScriptedLLM:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate(self, messages, *, max_tokens, temperature, stop=None) -> str:
        text = "\n".join(m["content"] for m in messages)
        self.seen.append(text)
        if messages[0]["content"].startswith("You classify"):
            if "bill" in messages[-1]["content"].lower():
                return '{"intent": "check_bill_balance", "confidence": 0.96, "slots": {}}'
            return '{"intent": "answer_question", "confidence": 0.9, "slots": {}}'
        return "The monthly charge is 300 EGP [1]."

    def stream(self, *a, **k):
        yield ""


@pytest.fixture()
def env(tmp_path, monkeypatch):
    base = get_settings()
    settings = Settings(
        profile=base.profile, slots=base.slots, data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma", corpus_path=tmp_path / "c.jsonl",
        sqlite_path=tmp_path / "t.db", session_files_dir=tmp_path / "sessions",
        traces_path=tmp_path / "traces.jsonl",
    )
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TE_AGENT_TOKEN", TOKEN)
    services = Services.build(settings)
    services.retriever.store._embedder = StubEmbedder()  # noqa: SLF001
    llm = ScriptedLLM()
    monkeypatch.setattr(api, "_services", services)
    monkeypatch.setattr("te_assistant.pipeline.get_llm", lambda _s: llm)
    monkeypatch.setattr("te_assistant.retrieval.context.get_reranker", lambda _s: None)
    METRICS.reset()
    return TestClient(api.app), services, llm


def _chat(client: TestClient, session_id: str, message: str) -> dict:
    response = client.post("/chat", json={"message": message, "session_id": session_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_then_ask_returns_a_cited_answer_with_the_document_restored(env) -> None:
    client, _, llm = env
    session_id = client.post("/session").json()["session_id"]

    upload = client.post(
        "/documents", data={"session_id": session_id},
        files={"file": ("statement.txt", BILL.encode(), "text/plain")},
    )
    assert upload.status_code == 200, upload.text
    assert upload.json()["chunks"] >= 1

    body = _chat(client, session_id, "What is the monthly charge on my statement?")

    assert body["record"]["grounded"] is True
    assert body["citations"], "the answer must cite the upload"
    assert any(PHONE in c["snippet"] for c in body["citations"]), "the user sees their own document"
    assert all(PHONE not in prompt for prompt in llm.seen), "the model never does"


def test_the_trace_file_written_through_the_api_holds_no_pii(env) -> None:
    client, services, _ = env
    session_id = client.post("/session").json()["session_id"]
    client.post("/documents", data={"session_id": session_id},
                files={"file": ("statement.txt", BILL.encode(), "text/plain")})
    _chat(client, session_id, f"my number is {PHONE}, what is the monthly charge?")

    persisted = services.settings.traces_path.read_text(encoding="utf-8")
    assert persisted.strip(), "the turn should have been recorded"
    assert PHONE not in persisted


def test_insights_and_metrics_reflect_an_ordinary_answered_turn(env) -> None:
    client, _, _ = env
    session_id = client.post("/session").json()["session_id"]
    client.post("/documents", data={"session_id": session_id},
                files={"file": ("statement.txt", BILL.encode(), "text/plain")})
    _chat(client, session_id, "What is the monthly charge on my statement?")

    insights = client.get(f"/insights/{session_id}").json()["insights"]
    assert insights["turns"] == 1 and insights["grounded_rate"] == 1.0
    assert client.get("/metrics").json()["turns"] == 1, "the main exit path reaches /metrics"
    assert client.get(f"/metrics/{session_id}").json()["metrics"]["turns"] == 1


def test_the_demo_flow_refused_then_granted_works_over_http(env) -> None:
    client, _, _ = env
    session_id = client.post("/session").json()["session_id"]

    refused = _chat(client, session_id, "what is my bill")
    assert refused["record"]["resolution"] == "refused"

    assert client.post("/demo/grant", data={"session_id": session_id},
                       headers={"X-Agent-Token": TOKEN}).status_code == 200
    granted = _chat(client, session_id, "what is my bill")
    assert granted["record"]["action"]["executed"] is True
    result = granted["record"]["action"]["result"]
    assert result["plan"] == "WE Nitro 200", "the caller keeps their own data"

    trail = client.get(f"/audit/{session_id}").json()["entries"]
    assert [bool(e["allowed"]) for e in trail] == [False, True], "both decisions audited"


def test_escalation_reaches_the_operator_queue_masked(env) -> None:
    client, _, _ = env
    session_id = client.post("/session").json()["session_id"]
    _chat(client, session_id, f"my line {PHONE} is down")
    body = _chat(client, session_id, "I want to speak to a human agent")
    assert body["handoff"] is not None

    assert client.get("/handoffs").status_code == 401
    queue = client.get("/handoffs", headers={"X-Agent-Token": TOKEN})
    assert queue.status_code == 200 and queue.json()["count"] == 1
    assert PHONE not in queue.text


def test_ending_a_session_removes_its_documents_and_cached_tokens(env) -> None:
    client, services, _ = env
    session_id = client.post("/session").json()["session_id"]
    client.post("/documents", data={"session_id": session_id},
                files={"file": ("statement.txt", BILL.encode(), "text/plain")})
    _chat(client, session_id, "monthly charge")
    assert services.retriever.store.session_size(session_id) > 0

    assert client.delete(f"/session/{session_id}").status_code == 200

    assert services.retriever.store.session_size(session_id) == 0
    assert session_id not in services.retriever._session_tokens  # noqa: SLF001
    # An expired id on /documents is a 404 — the UI relies on this to renew.
    gone = client.post("/documents", data={"session_id": session_id},
                       files={"file": ("a.txt", BILL.encode(), "text/plain")})
    assert gone.status_code == 404


def test_a_chat_on_an_expired_session_answers_under_a_new_id(env) -> None:
    """The contract the UI's `adopt_session` is built on."""
    client, _, _ = env
    body = _chat(client, "no-such-session", "hello there")
    assert body["record"]["session_id"] != "no-such-session"


def test_the_chat_response_round_trips_as_json(env) -> None:
    client, _, _ = env
    session_id = client.post("/session").json()["session_id"]
    client.post("/documents", data={"session_id": session_id},
                files={"file": ("statement.txt", BILL.encode(), "text/plain")})
    response = client.post("/chat", json={"message": "monthly charge", "session_id": session_id})
    parsed = json.loads(response.text)
    assert set(parsed) >= {"answer", "citations", "language", "record"}
