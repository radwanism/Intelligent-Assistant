"""Session isolation — brief B.8.

The requirement: with concurrent users, one session must never retrieve another
session's uploaded documents. These tests assert that directly, including the
case the design is actually built against — a caller who *forgets* the filter.

A deterministic stub embedder is injected so the suite runs in milliseconds and
tests the isolation logic rather than a model's similarity scores.
"""

from __future__ import annotations

import hashlib

import pytest

from te_assistant.config import Settings, get_settings
from te_assistant.retrieval.store import SessionScopeError, VectorStore
from te_assistant.schemas import Chunk, SourceKind


class StubEmbedder:
    """Deterministic hash-based vectors. Same text always maps to same point."""

    dim = 16

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dim)]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


@pytest.fixture()
def store(tmp_path) -> VectorStore:
    base = get_settings()
    settings = Settings(
        profile=base.profile,
        slots=base.slots,
        data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        corpus_path=tmp_path / "corpus.jsonl",
        sqlite_path=tmp_path / "test.db",
        session_files_dir=tmp_path / "sessions",
        traces_path=tmp_path / "traces.jsonl",
    )
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return VectorStore(settings, embedder=StubEmbedder())


def _session_chunk(session_id: str, text: str, chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        title="bill.pdf",
        source_kind=SourceKind.USER_DOCUMENT,
        session_id=session_id,
        doc_name="bill.pdf",
    )


@pytest.fixture()
def two_sessions(store: VectorStore) -> VectorStore:
    store.add_session_chunks(
        [_session_chunk("alice", "Alice bill total 500 EGP", "a1")], "alice"
    )
    store.add_session_chunks(
        [_session_chunk("bob", "Bob bill total 900 EGP", "b1")], "bob"
    )
    return store


# --------------------------------------------------------------------------
# The core property
# --------------------------------------------------------------------------
def test_session_cannot_see_another_sessions_document(two_sessions: VectorStore) -> None:
    results = two_sessions.search_session_docs("bill total", "alice", k=10)
    assert results, "alice should find her own document"
    for scored in results:
        assert scored.chunk.session_id == "alice"
        assert "Bob" not in scored.chunk.text


def test_searching_for_the_others_exact_text_still_isolates(
    two_sessions: VectorStore,
) -> None:
    """Even querying Bob's exact wording must not surface Bob's chunk."""
    results = two_sessions.search_session_docs("Bob bill total 900 EGP", "alice", k=10)
    assert all(scored.chunk.session_id == "alice" for scored in results)


def test_unknown_session_retrieves_nothing(two_sessions: VectorStore) -> None:
    assert two_sessions.search_session_docs("bill", "mallory", k=10) == []


# --------------------------------------------------------------------------
# The failure mode the design removes: forgetting the filter
# --------------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["", None])
def test_unscoped_session_search_is_refused(two_sessions: VectorStore, missing) -> None:
    """There is no 'search all sessions' path — omitting the id raises."""
    with pytest.raises(SessionScopeError):
        two_sessions.search_session_docs("bill", missing, k=10)


def test_writing_without_a_session_is_refused(store: VectorStore) -> None:
    with pytest.raises(SessionScopeError):
        store.add_session_chunks([_session_chunk("alice", "x", "x1")], "")


def test_chunk_session_must_match_the_writing_session(store: VectorStore) -> None:
    """A mislabelled chunk is rejected rather than silently written."""
    with pytest.raises(SessionScopeError):
        store.add_session_chunks([_session_chunk("alice", "x", "x1")], "bob")


def test_user_documents_cannot_enter_the_shared_kb(store: VectorStore) -> None:
    """An upload must never leak into the corpus every session can read."""
    with pytest.raises(SessionScopeError):
        store.add_kb_chunks([_session_chunk("alice", "private", "p1")])


def test_kb_chunks_cannot_be_written_into_the_session_collection(
    store: VectorStore,
) -> None:
    kb_chunk = Chunk(
        chunk_id="k1", text="public page", source_kind=SourceKind.TE_EG, session_id="alice"
    )
    with pytest.raises(SessionScopeError):
        store.add_session_chunks([kb_chunk], "alice")


# --------------------------------------------------------------------------
# Eviction
# --------------------------------------------------------------------------
def test_ending_a_session_evicts_only_its_own_chunks(two_sessions: VectorStore) -> None:
    assert two_sessions.session_size("alice") == 1
    assert two_sessions.session_size("bob") == 1

    two_sessions.delete_session("alice")

    assert two_sessions.session_size("alice") == 0
    assert two_sessions.session_size("bob") == 1, "bob's data must survive"
    assert two_sessions.search_session_docs("bill", "alice", k=10) == []


def test_delete_requires_a_session_id(two_sessions: VectorStore) -> None:
    with pytest.raises(SessionScopeError):
        two_sessions.delete_session("")
