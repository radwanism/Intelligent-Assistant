"""Chunking.

Heading-aware and paragraph-respecting: we pack whole lines into a chunk until
it would exceed the budget, rather than cutting at a fixed offset. Splitting
mid-sentence is especially costly in Arabic retrieval because the orthography
gives the embedder fewer sub-word cues to recover from a truncated clause.

Token budgeting uses a character proxy rather than a real tokenizer. The two
profiles use different tokenizers (Qwen2.5 vs Qwen3) and loading either just to
measure chunk length would pull a dependency into the ingest path for no gain.
Arabic runs ~2.6 chars/token and English ~4.0, so we estimate per-script and
stay conservative — the cost of a slightly short chunk is nil, the cost of one
that overflows the context window is a truncated prompt.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..retrieval.normalize_ar import script_ratios
from ..schemas import Chunk, Language, SourceKind

# Markdown-ish heading, or a short line that looks like a section label.
HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[^\s].{0,70})$")

CHARS_PER_TOKEN_AR = 2.6
CHARS_PER_TOKEN_EN = 4.0


def estimate_tokens(text: str) -> int:
    """Script-aware token estimate."""
    if not text:
        return 0
    arabic_ratio, _ = script_ratios(text)
    chars_per_token = (
        CHARS_PER_TOKEN_AR * arabic_ratio + CHARS_PER_TOKEN_EN * (1 - arabic_ratio)
    )
    return int(len(text) / max(chars_per_token, 1.0)) + 1


def _looks_like_heading(line: str) -> bool:
    if line.startswith("#"):
        return True
    # Short, few words, no terminal punctuation: typical of a section label on
    # te.eg. The word-count bound matters — without it every item in a bulleted
    # plan list reads as a heading and the section label churns line by line.
    stripped = line.rstrip()
    if stripped.endswith((".", "،", "؟", "!", ":", "?")):
        return False
    return len(stripped) <= 70 and len(stripped.split()) <= 8


@dataclass
class TextChunk:
    text: str
    section: str
    index: int


def split_text(
    text: str,
    *,
    max_tokens: int = 450,
    overlap_tokens: int = 60,
    min_chars: int = 120,
) -> list[TextChunk]:
    """Pack lines into token-budgeted chunks, carrying the current heading."""
    lines = [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if not lines:
        return []

    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    section = ""
    index = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens, index
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if body:
            if len(body) >= min_chars:
                chunks.append(TextChunk(text=body, section=section, index=index))
                index += 1
            elif chunks and estimate_tokens(chunks[-1].text) + estimate_tokens(
                body
            ) <= max_tokens:
                # Too small to stand alone, but dropping it loses content — a
                # document of short lines (a bulleted plan list, a table of
                # codes) would otherwise vanish entirely. Merge backwards, but
                # only while the result still fits the budget: an unbounded
                # merge collapses such a document into one oversized chunk,
                # which overflows the prompt and retrieves as a single blob.
                previous = chunks[-1]
                chunks[-1] = TextChunk(
                    text=f"{previous.text}\n{body}",
                    section=previous.section,
                    index=previous.index,
                )
            else:
                # Nothing to merge into yet: keep it and let a later flush grow
                # it, rather than starting the document by discarding its head.
                chunks.append(TextChunk(text=body, section=section, index=index))
                index += 1
        buffer = []
        buffer_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)

        if _looks_like_heading(line) and buffer_tokens > max_tokens * 0.4:
            # A heading is a natural boundary, but only once the current chunk
            # has enough substance to stand alone.
            flush()
            section = line.lstrip("# ").strip()
            buffer = [line]
            buffer_tokens = line_tokens
            continue

        if buffer_tokens + line_tokens > max_tokens and buffer:
            tail = _overlap_tail(buffer, overlap_tokens)
            flush()
            buffer = [*tail, line]
            buffer_tokens = sum(estimate_tokens(item) for item in buffer)
            continue

        if _looks_like_heading(line) and not buffer:
            section = line.lstrip("# ").strip()

        buffer.append(line)
        buffer_tokens += line_tokens

    flush()

    # A document that is entirely too short to be useful is dropped, but a long
    # document is never silently truncated by the per-chunk minimum above.
    if len(text.strip()) < min_chars:
        return []
    return chunks


def _overlap_tail(buffer: list[str], overlap_tokens: int) -> list[str]:
    """Trailing lines worth about `overlap_tokens`, to bridge the boundary.

    Overlap matters for answer quality: a plan's price frequently sits one line
    below its name, and a hard cut between them produces a chunk that retrieves
    well and answers nothing.
    """
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for line in reversed(buffer):
        line_tokens = estimate_tokens(line)
        if total + line_tokens > overlap_tokens:
            break
        tail.insert(0, line)
        total += line_tokens
    return tail


def chunk_id_for(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:20]


def chunks_from_document(
    *,
    text: str,
    title: str,
    url: str = "",
    language: Language = Language.UNKNOWN,
    source_kind: SourceKind = SourceKind.TE_EG,
    session_id: str | None = None,
    doc_name: str | None = None,
    page: int | None = None,
    max_tokens: int = 450,
    overlap_tokens: int = 60,
    min_chars: int = 120,
) -> list[Chunk]:
    """Turn one cleaned document into retrievable `Chunk`s.

    The title is prepended to each chunk's text. It is a cheap, large win for
    retrieval: te.eg body copy often never repeats the product name that the
    user is searching for ("WE Bonus"), because the page title already said it.
    """
    pieces = split_text(
        text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_chars=min_chars
    )
    out: list[Chunk] = []
    for piece in pieces:
        header = " — ".join(part for part in (title, piece.section) if part)
        body = f"{header}\n{piece.text}" if header else piece.text
        out.append(
            Chunk(
                chunk_id=chunk_id_for(url or doc_name or "", str(page or 0), str(piece.index)),
                text=body,
                title=title,
                url=url,
                language=language,
                source_kind=source_kind,
                session_id=session_id,
                doc_name=doc_name,
                page=page,
                section=piece.section or None,
            )
        )
    return out
