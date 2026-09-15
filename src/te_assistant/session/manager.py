"""Session lifecycle — brief B.8.

The requirement is that concurrent users never retrieve each other's uploaded
documents. Two mechanisms, and it is worth being clear about which does the work:

  * **Unguessable ids** (`secrets.token_urlsafe(32)`, 256 bits) stop a caller
    *guessing* another session. This is necessary but it is not the isolation
    property — it only makes the identifier hard to forge.
  * **Mandatory filtering** in `retrieval/store.py` is what actually isolates.
    Even holding a valid session id, a query can only ever reach chunks tagged
    with that same id, because the unfiltered code path does not exist.

There is deliberately no login (B.2 puts authentication in future work), so a
session is an anonymous capability: possession of the token is the only claim
it makes, and it grants access to nothing but that session's own uploads.

Eviction is real, not decorative: at expiry the uploaded files are deleted from
disk and their embeddings are removed from the vector store. A POC that leaves
customer documents lying around after the session ends would be making the
opposite point to the one this case study is about.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..schemas import Language

log = logging.getLogger(__name__)

TOKEN_BYTES = 32  # 256 bits


@dataclass
class TurnHistory:
    """Conversation memory for one session (A.2: display chat history)."""

    turns: list[dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, text: str, **extra: Any) -> None:
        self.turns.append({"role": role, "text": text, **extra})

    def recent_questions(self, limit: int = 4) -> list[str]:
        return [t["text"] for t in self.turns if t["role"] == "user"][-limit:]

    def as_messages(self, limit: int = 6) -> list[dict[str, str]]:
        return [
            {"role": t["role"], "content": t["text"]}
            for t in self.turns[-limit:]
        ]


@dataclass
class Session:
    session_id: str
    created_at: float
    last_seen: float
    files_dir: Path
    history: TurnHistory = field(default_factory=TurnHistory)
    documents: list[str] = field(default_factory=list)
    language: Language = Language.UNKNOWN
    handed_off: bool = False
    # Rolling frustration evidence, used by frustration/detect.py.
    low_confidence_streak: int = 0
    repeat_streak: int = 0

    def touch(self) -> None:
        self.last_seen = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.last_seen) > ttl_seconds


class SessionManager:
    """In-process session store.

    In-process is the right call for a POC on a single host, and the interface
    is deliberately narrow (`create/get/touch/end`) so swapping in Redis for a
    multi-replica deployment is a one-file change. That tradeoff is named in the
    Scalability section rather than pre-solved here.
    """

    def __init__(self, settings: Settings | None = None, store: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._store = store  # VectorStore; injected to keep this module testable

    # -- lifecycle ---------------------------------------------------------
    def create(self) -> Session:
        session_id = secrets.token_urlsafe(TOKEN_BYTES)
        now = time.time()
        files_dir = self.settings.session_files_dir / session_id
        files_dir.mkdir(parents=True, exist_ok=True)
        session = Session(
            session_id=session_id, created_at=now, last_seen=now, files_dir=files_dir
        )
        with self._lock:
            self._sessions[session_id] = session
        log.info("session created %s", session_id[:8])
        return session

    def get(self, session_id: str) -> Session | None:
        """Look up a session. Expired sessions are reaped, not returned."""
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(self.settings.session_ttl_seconds):
                self._end_locked(session_id)
                return None
            session.touch()
            return session

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            raise KeyError("unknown or expired session")
        return session

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create()

    def end(self, session_id: str) -> None:
        with self._lock:
            self._end_locked(session_id)

    def _end_locked(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        # Embeddings first: a crash midway should not leave retrievable chunks
        # pointing at files that are already gone.
        if self._store is not None:
            try:
                self._store.delete_session(session_id)
            except Exception:
                log.exception("failed to evict embeddings for %s", session_id[:8])
        try:
            shutil.rmtree(session.files_dir, ignore_errors=True)
        except Exception:
            log.exception("failed to remove files for %s", session_id[:8])
        log.info("session ended %s", session_id[:8])

    def sweep(self) -> int:
        """Reap expired sessions. Called periodically by the core service."""
        with self._lock:
            expired = [
                sid
                for sid, session in self._sessions.items()
                if session.is_expired(self.settings.session_ttl_seconds)
            ]
            for sid in expired:
                self._end_locked(sid)
        if expired:
            log.info("swept %d expired sessions", len(expired))
        return len(expired)

    # -- introspection -----------------------------------------------------
    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def safe_upload_path(self, session: Session, filename: str) -> Path:
        """Resolve an upload path inside the session directory.

        Filenames arrive from the browser, so they are attacker-controlled:
        `../../etc/passwd` and absolute paths both have to be neutralised before
        the name touches the filesystem.
        """
        cleaned = Path(filename).name  # strips any directory component
        cleaned = cleaned.replace("\x00", "").strip() or "upload.bin"
        target = (session.files_dir / cleaned).resolve()
        root = session.files_dir.resolve()
        if not str(target).startswith(str(root)):
            raise ValueError("upload path escapes the session directory")
        return target
