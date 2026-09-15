"""Per-session document ingestion (brief B.3, B.8).

The upload path from file to retrievable chunk, with session scoping applied at
every step rather than at the end:

    validate -> save inside the session directory -> parse (OCR if needed)
             -> chunk -> tag with session_id -> write to kb_sessions

Every chunk is stamped with `session_id` *before* it reaches the store, and the
store re-validates that stamp on write (`add_session_chunks` rejects a mismatch).
Two independent checks for one property, because this is the isolation
requirement the brief calls out specifically and a single check is a single
point of failure.

Files live under `data/sessions/<session_id>/` and are deleted with the session,
along with their embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Settings, get_settings
from ..ingest.chunk import chunks_from_document
from ..schemas import Chunk, SourceKind, UploadResponse
from ..session.manager import Session
from .parse import SUPPORTED_SUFFIXES, UnsupportedDocument, parse_document

log = logging.getLogger(__name__)


class UploadTooLarge(ValueError):
    pass


def validate_upload(filename: str, size_bytes: int, settings: Settings) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(
            f"unsupported file type '{suffix or 'unknown'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    limit = settings.max_upload_mb * 1024 * 1024
    if size_bytes > limit:
        raise UploadTooLarge(f"file exceeds the {settings.max_upload_mb} MB limit")
    if size_bytes == 0:
        raise UnsupportedDocument("file is empty")


def ingest_upload(
    *,
    session: Session,
    filename: str,
    payload: bytes,
    store,
    settings: Settings | None = None,
) -> UploadResponse:
    """Parse, chunk, embed and index one uploaded file into its session."""
    settings = settings or get_settings()
    validate_upload(filename, len(payload), settings)

    # Neutralises traversal in the browser-supplied filename.
    target = _safe_path(session, filename)
    target.write_bytes(payload)

    try:
        document = parse_document(target)
    except UnsupportedDocument:
        target.unlink(missing_ok=True)
        raise

    if not document.text.strip():
        target.unlink(missing_ok=True)
        raise UnsupportedDocument(
            "no readable text found — if this is a scan, try a clearer image"
        )

    chunks: list[Chunk] = []
    for page in document.pages:
        if not page.text.strip():
            continue
        chunks.extend(
            chunks_from_document(
                text=page.text,
                title=document.name,
                url="",
                language=document.language,
                source_kind=SourceKind.USER_DOCUMENT,
                session_id=session.session_id,
                doc_name=document.name,
                page=page.page,
                max_tokens=settings.chunk_tokens,
                overlap_tokens=settings.chunk_overlap,
                min_chars=settings.min_chunk_chars,
            )
        )

    if not chunks:
        target.unlink(missing_ok=True)
        raise UnsupportedDocument("document produced no indexable content")

    store.add_session_chunks(chunks, session.session_id)

    if document.name not in session.documents:
        session.documents.append(document.name)

    log.info(
        "ingested %s for session %s: %d pages, %d chunks, ocr=%s",
        document.name, session.session_id[:8], len(document.pages), len(chunks),
        document.used_ocr,
    )

    return UploadResponse(
        doc_name=document.name,
        pages=len(document.pages),
        chunks=len(chunks),
        used_ocr=document.used_ocr,
        language=document.language,
    )


def _safe_path(session: Session, filename: str) -> Path:
    """Resolve an upload path that cannot escape the session directory."""
    cleaned = Path(filename).name.replace("\x00", "").strip() or "upload.bin"
    root = session.files_dir.resolve()
    target = (root / cleaned).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError("upload path escapes the session directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
