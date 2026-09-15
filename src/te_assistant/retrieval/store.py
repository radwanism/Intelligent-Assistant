"""Vector store access.

This module owns the single most important security property in the system
(brief B.8): **one session must never retrieve another session's documents.**

The design rule is that isolation is *enforced here*, not requested by callers:

  * `kb_te_eg` is shared and read-only. Anyone may search it.
  * `kb_sessions` is reachable only through `search_session_docs`, which
    **requires** a session_id and always attaches the metadata filter.
  * There is deliberately no public method that searches the session collection
    without a filter. The failure mode we are designing against is a caller who
    forgets to pass one, so we removed the ability to forget.

`tests/test_session_isolation.py` asserts the negative case directly.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import Settings, get_settings
from ..schemas import Chunk, Language, ScoredChunk, SourceKind

log = logging.getLogger(__name__)


class SessionScopeError(RuntimeError):
    """Raised when a session-scoped operation is attempted without a session."""


class VectorStore:
    def __init__(self, settings: Settings | None = None, embedder: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._kb = self._client.get_or_create_collection(
            name=self.settings.kb_collection, metadata={"hnsw:space": "cosine"}
        )
        self._sessions = self._client.get_or_create_collection(
            name=self.settings.session_collection, metadata={"hnsw:space": "cosine"}
        )

    # -- embedding ---------------------------------------------------------
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from .embedder import get_embedder  # local import keeps tests light

            self._embedder = get_embedder(self.settings)
        return self._embedder

    # -- writes ------------------------------------------------------------
    def add_kb_chunks(self, chunks: list[Chunk], *, batch_size: int = 128) -> int:
        """Index shared te.eg content. Rejects anything session-scoped."""
        for chunk in chunks:
            if chunk.source_kind is not SourceKind.TE_EG:
                raise SessionScopeError(
                    f"refusing to write {chunk.source_kind} into the shared KB"
                )
        return self._add(self._kb, chunks, batch_size=batch_size)

    def add_session_chunks(
        self, chunks: list[Chunk], session_id: str, *, batch_size: int = 64
    ) -> int:
        """Index an uploaded document under exactly one session."""
        if not session_id:
            raise SessionScopeError("session_id is required to write session documents")
        for chunk in chunks:
            if chunk.session_id != session_id:
                raise SessionScopeError(
                    "chunk.session_id does not match the writing session — refusing"
                )
            if chunk.source_kind is not SourceKind.USER_DOCUMENT:
                raise SessionScopeError("session collection accepts user documents only")
        return self._add(self._sessions, chunks, batch_size=batch_size)

    def _add(self, collection: Any, chunks: list[Chunk], *, batch_size: int) -> int:
        if not chunks:
            return 0
        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_passages([c.text for c in batch])
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=[list(map(float, v)) for v in vectors],
                documents=[c.text for c in batch],
                metadatas=[_to_metadata(c) for c in batch],
            )
            written += len(batch)
        return written

    # -- reads -------------------------------------------------------------
    def search_kb(self, query: str, *, k: int = 40) -> list[ScoredChunk]:
        return self._search(self._kb, query, k=k, where=None)

    def search_session_docs(self, query: str, session_id: str, *, k: int = 20
                            ) -> list[ScoredChunk]:
        """Search uploads. A missing session_id is an error, never 'search all'.

        This is the enforcement point: there is no code path that reaches the
        session collection with `where=None`.
        """
        if not session_id:
            raise SessionScopeError(
                "search_session_docs requires a session_id; "
                "an unscoped session search is never valid"
            )
        return self._search(
            self._sessions, query, k=k, where={"session_id": session_id}
        )

    def _search(
        self, collection: Any, query: str, *, k: int, where: dict[str, Any] | None
    ) -> list[ScoredChunk]:
        if collection.count() == 0:
            return []
        vector = self.embedder.embed_query(query)
        result = collection.query(
            query_embeddings=[list(map(float, vector))],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _to_scored(result)

    # -- lifecycle (B.8: eviction at session end) --------------------------
    def delete_session(self, session_id: str) -> int:
        if not session_id:
            raise SessionScopeError("delete_session requires a session_id")
        before = self._sessions.count()
        self._sessions.delete(where={"session_id": session_id})
        removed = before - self._sessions.count()
        log.info("evicted %d chunks for session %s", removed, session_id[:8])
        return removed

    def kb_size(self) -> int:
        return self._kb.count()

    def session_size(self, session_id: str) -> int:
        if not session_id:
            raise SessionScopeError("session_size requires a session_id")
        return len(
            self._sessions.get(where={"session_id": session_id}, include=[])["ids"]
        )

    def reset_kb(self) -> None:
        self._client.delete_collection(self.settings.kb_collection)
        self._kb = self._client.get_or_create_collection(
            name=self.settings.kb_collection, metadata={"hnsw:space": "cosine"}
        )


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    # Chroma metadata values must be scalars, so None is dropped rather than stored.
    meta: dict[str, Any] = {
        "title": chunk.title,
        "url": chunk.url,
        "language": chunk.language.value,
        "source_kind": chunk.source_kind.value,
    }
    if chunk.session_id:
        meta["session_id"] = chunk.session_id
    if chunk.doc_name:
        meta["doc_name"] = chunk.doc_name
    if chunk.page is not None:
        meta["page"] = chunk.page
    if chunk.section:
        meta["section"] = chunk.section
    return meta


def _to_scored(result: dict[str, Any]) -> list[ScoredChunk]:
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    out: list[ScoredChunk] = []
    for rank, (chunk_id, text, meta, distance) in enumerate(
        zip(ids, docs, metas, distances, strict=False)
    ):
        meta = meta or {}
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text or "",
            title=meta.get("title", ""),
            url=meta.get("url", ""),
            language=_lang(meta.get("language")),
            source_kind=SourceKind(meta.get("source_kind", SourceKind.TE_EG.value)),
            session_id=meta.get("session_id"),
            doc_name=meta.get("doc_name"),
            page=meta.get("page"),
            section=meta.get("section"),
        )
        # Chroma returns cosine distance; convert to a similarity for fusion.
        out.append(ScoredChunk(chunk=chunk, score=1.0 - float(distance), dense_rank=rank))
    return out


def _lang(value: Any) -> Language:
    try:
        return Language(value)
    except (ValueError, TypeError):
        return Language.UNKNOWN
