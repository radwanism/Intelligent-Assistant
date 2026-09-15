"""Grounded answer generation with citations.

Two properties this module is responsible for, both of which are requirements
rather than nice-to-haves (A.2):

**Every answer carries citations.** Passages go in numbered; the numbers come
back out and are resolved against the actual retrieved chunks. A marker the
model invents for a passage that was never supplied is dropped rather than
rendered, so a citation in the UI always points at a real retrieved source.

**Refusing is a valid answer.** When retrieval returns nothing, we do not call
the model at all — there is no context to ground in, and asking a 3B model to
answer from parametric memory about telecom tariffs is precisely how wrong
prices get quoted with confidence.
"""

from __future__ import annotations

import logging
import re

from ..retrieval.hybrid import confidence_of
from ..schemas import Citation, Language, ScoredChunk
from .client import LLM
from .prompts import build_answer_messages, no_context_message

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d{1,2})\]")
SNIPPET_CHARS = 220

# Below this, treat retrieval as having failed. Tuned against the golden set:
# the cost of refusing a marginal answer is far lower than the cost of an
# authoritative wrong price.
MIN_RETRIEVAL_CONFIDENCE = 0.18


class GroundedAnswer:
    def __init__(
        self,
        text: str,
        citations: list[Citation],
        *,
        grounded: bool,
        retrieval_confidence: float,
    ) -> None:
        self.text = text
        self.citations = citations
        self.grounded = grounded
        self.retrieval_confidence = retrieval_confidence


def _snippet(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def extract_citations(answer: str, chunks: list[ScoredChunk]) -> tuple[str, list[Citation]]:
    """Resolve [n] markers to real sources, dropping any that do not exist.

    Renumbering matters: if the model cites [1] and [4], the user should see [1]
    and [2] against a two-item source list, not two entries numbered 1 and 4.
    """
    used_indices: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= len(chunks) and index not in used_indices:
            used_indices.append(index)

    if not used_indices:
        return answer, []

    renumber = {old: new for new, old in enumerate(sorted(used_indices), start=1)}

    def _replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index in renumber:
            return f"[{renumber[index]}]"
        return ""  # a marker for a passage that was never supplied

    cleaned = CITATION_RE.sub(_replace, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    citations: list[Citation] = []
    for old in sorted(used_indices):
        chunk = chunks[old - 1].chunk
        citations.append(
            Citation(
                marker=f"[{renumber[old]}]",
                title=chunk.title or chunk.doc_name or chunk.url or "Source",
                url=chunk.url,
                doc_name=chunk.doc_name,
                page=chunk.page,
                snippet=_snippet(chunk.text),
            )
        )
    return cleaned, citations


def generate_answer(
    llm: LLM,
    *,
    question: str,
    chunks: list[ScoredChunk],
    language: Language,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 320,
    temperature: float = 0.2,
) -> GroundedAnswer:
    confidence = confidence_of(chunks)

    if not chunks or confidence < MIN_RETRIEVAL_CONFIDENCE:
        log.info("refusing: retrieval confidence %.3f below threshold", confidence)
        return GroundedAnswer(
            no_context_message(language), [], grounded=False, retrieval_confidence=confidence
        )

    messages = build_answer_messages(
        question=question, chunks=chunks, language=language, history=history
    )
    try:
        raw = llm.generate(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        log.exception("generation failed")
        return GroundedAnswer(
            no_context_message(language), [], grounded=False, retrieval_confidence=confidence
        )

    if not raw.strip():
        return GroundedAnswer(
            no_context_message(language), [], grounded=False, retrieval_confidence=confidence
        )

    text, citations = extract_citations(raw, chunks)

    # An answer with no citations at all, against retrieved context, means the
    # model ignored the instruction. We surface the text but mark it ungrounded
    # so the insights view and the eval harness both count it honestly rather
    # than quietly reporting a grounded answer rate that is not real.
    grounded = bool(citations)
    if not grounded:
        log.info("answer produced no resolvable citations")

    return GroundedAnswer(
        text, citations, grounded=grounded, retrieval_confidence=confidence
    )


def render_with_sources(answer: GroundedAnswer, *, arabic: bool) -> str:
    """Append a readable source list. The UI also renders them structurally."""
    if not answer.citations:
        return answer.text
    header = "المصادر:" if arabic else "Sources:"
    lines = [answer.text, "", header]
    for citation in answer.citations:
        where = citation.url or citation.doc_name or ""
        if citation.page:
            where = f"{where} (p.{citation.page})"
        lines.append(f"{citation.marker} {citation.title} — {where}".rstrip(" —"))
    return "\n".join(lines)
