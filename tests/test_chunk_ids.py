"""Chunk identity.

The store writes with an upsert, so the id *is* the identity: two chunks that
share an id are the same row. These tests pin the properties that follow.
"""

from __future__ import annotations

import hashlib

from te_assistant.ingest.chunk import chunk_id_for, chunks_from_document
from te_assistant.schemas import SourceKind

TEXT = "A line of document text that is comfortably long enough to be kept. " * 6


def _upload(session_id: str, name: str = "bill.pdf"):
    return chunks_from_document(
        text=TEXT, title=name, source_kind=SourceKind.USER_DOCUMENT,
        session_id=session_id, doc_name=name, page=1,
    )


def test_same_filename_in_different_sessions_gets_different_ids() -> None:
    alice = {c.chunk_id for c in _upload("alice")}
    bob = {c.chunk_id for c in _upload("bob")}
    assert alice and bob
    assert alice.isdisjoint(bob), "an upsert would let one session overwrite the other"


def test_re_uploading_in_the_same_session_is_idempotent() -> None:
    first = [c.chunk_id for c in _upload("alice")]
    second = [c.chunk_id for c in _upload("alice")]
    assert first == second, "the same file in the same session must upsert, not duplicate"


def test_ids_within_one_document_are_unique() -> None:
    ids = [c.chunk_id for c in _upload("alice")]
    assert len(ids) == len(set(ids))


def test_knowledge_base_ids_are_unchanged() -> None:
    """The committed index was built with the old derivation; it must stay valid."""
    chunks = chunks_from_document(
        text=TEXT, title="Plans", url="https://te.eg/plans", source_kind=SourceKind.TE_EG
    )
    joined = "||".join(["https://te.eg/plans", "0", "0"]).encode()
    digest = hashlib.sha1(joined, usedforsecurity=False)
    assert chunks[0].chunk_id == digest.hexdigest()[:20]
    assert chunks[0].chunk_id == chunk_id_for("https://te.eg/plans", "0", "0")
