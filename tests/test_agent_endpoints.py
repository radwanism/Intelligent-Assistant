"""Operator-only endpoints.

A session token is a capability over one session's own data. `/handoffs` spans
sessions and `/demo/grant` mints permissions, so holding a session — even a
valid one — must never be enough to reach them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from te_assistant import api
from te_assistant.config import Settings, get_settings
from te_assistant.pipeline import Services
from te_assistant.schemas import IntentName, Language

TOKEN = "operator-secret-for-tests"
PHONE = "01012345678"


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
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
    monkeypatch.setattr(api, "_services", services)
    # No `with`: skipping the lifespan keeps real models out of the test.
    return TestClient(api.app)


def _services() -> Services:
    return api.get_services()


def _open_ticket(services: Services, session_id: str = "victim-session") -> None:
    services.handoffs.open(
        session_id=session_id, reason="test",
        transcript=[{"role": "user", "content": f"my line is {PHONE}"}],
        language=Language.EN, last_intent=IntentName.HUMAN_HANDOFF,
    )


# --------------------------------------------------------------------------
# /handoffs
# --------------------------------------------------------------------------
def test_handoffs_refuses_a_caller_with_no_token(client) -> None:
    _open_ticket(_services())
    response = client.get("/handoffs")
    assert response.status_code == 401
    assert "transcript" not in response.text


def test_handoffs_refuses_a_wrong_token(client) -> None:
    assert client.get("/handoffs", headers={"X-Agent-Token": "guess"}).status_code == 401


def test_a_valid_session_token_does_not_open_the_agent_queue(client) -> None:
    """Holding a real session is exactly what an ordinary user has."""
    _open_ticket(_services())
    session_id = client.post("/session").json()["session_id"]
    assert client.get("/handoffs", headers={"X-Agent-Token": session_id}).status_code == 401


def test_handoffs_opens_for_the_operator_and_is_masked(client) -> None:
    _open_ticket(_services())
    response = client.get("/handoffs", headers={"X-Agent-Token": TOKEN})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert PHONE not in response.text, "the queue must not hold raw PII"


# --------------------------------------------------------------------------
# /demo/grant
# --------------------------------------------------------------------------
def test_grant_refuses_a_caller_with_no_token_and_grants_nothing(client) -> None:
    response = client.post("/demo/grant", data={"session_id": "s1", "customer_id": "cust-1001"})
    assert response.status_code == 401
    assert _services().db.customers_for_session("s1") == []


def test_grant_works_for_the_operator(client) -> None:
    response = client.post(
        "/demo/grant",
        data={"session_id": "s1", "customer_id": "cust-1001", "scope": "billing:read"},
        headers={"X-Agent-Token": TOKEN},
    )
    assert response.status_code == 200
    assert _services().db.has_permission("s1", "cust-1001", "billing:read")


# --------------------------------------------------------------------------
# Nothing else changed
# --------------------------------------------------------------------------
def test_session_scoped_endpoints_stay_open(client) -> None:
    """Sessions are anonymous capabilities; these must keep working unauthenticated."""
    session_id = client.post("/session").json()["session_id"]
    assert client.get(f"/insights/{session_id}").status_code == 200
    assert client.get(f"/audit/{session_id}").status_code == 200
    assert client.get(f"/metrics/{session_id}").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.delete(f"/session/{session_id}").status_code == 200


# --------------------------------------------------------------------------
# Token provenance
# --------------------------------------------------------------------------
def test_a_token_is_generated_when_none_is_configured(tmp_path, monkeypatch) -> None:
    base = get_settings()
    settings = Settings(
        profile=base.profile, slots=base.slots, data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma", corpus_path=tmp_path / "c.jsonl",
        sqlite_path=tmp_path / "t.db", session_files_dir=tmp_path / "sessions",
        traces_path=tmp_path / "traces.jsonl",
    )
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("TE_AGENT_TOKEN", raising=False)

    first = Services.build(settings)
    assert first.agent_token_generated
    assert len(first.agent_token) >= 24, "a generated token must be unguessable"
    assert Services.build(settings).agent_token != first.agent_token


def test_a_configured_token_is_used_verbatim(tmp_path, monkeypatch) -> None:
    base = get_settings()
    settings = Settings(
        profile=base.profile, slots=base.slots, data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma", corpus_path=tmp_path / "c.jsonl",
        sqlite_path=tmp_path / "t.db", session_files_dir=tmp_path / "sessions",
        traces_path=tmp_path / "traces.jsonl",
    )
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TE_AGENT_TOKEN", "  chosen  ")
    services = Services.build(settings)
    assert services.agent_token == "chosen"
    assert not services.agent_token_generated
