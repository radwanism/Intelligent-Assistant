"""Shared data contracts.

These types are the integration surface between the three services (brief B.1):
`ui` <-> `core` <-> `speech`. Keeping them in one module means a change to the
contract is a change to one file, and the unit tests can assert on it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Language / dialect
# --------------------------------------------------------------------------
class Language(str, Enum):
    AR = "ar"          # Modern Standard Arabic
    ARZ = "arz"        # Egyptian Arabic (ISO 639-3)
    EN = "en"
    MIXED = "mixed"    # AR/EN code-switching — common in Egyptian telecom speech
    UNKNOWN = "unknown"

    @property
    def is_arabic(self) -> bool:
        return self in (Language.AR, Language.ARZ, Language.MIXED)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
class SourceKind(str, Enum):
    TE_EG = "te.eg"              # shared knowledge base
    USER_DOCUMENT = "user_doc"   # session-scoped upload


class Chunk(BaseModel):
    """One retrievable unit. `source_kind` + `session_id` drive isolation (B.8)."""

    chunk_id: str
    text: str
    title: str = ""
    url: str = ""
    language: Language = Language.UNKNOWN
    source_kind: SourceKind = SourceKind.TE_EG
    # Populated only for USER_DOCUMENT. A chunk with source_kind=USER_DOCUMENT
    # and session_id=None is a bug, and the store refuses to write it.
    session_id: str | None = None
    doc_name: str | None = None
    page: int | None = None
    section: str | None = None


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """What the UI renders under an answer (A.2: citations for every answer)."""

    marker: str                # "[1]"
    title: str
    url: str = ""
    doc_name: str | None = None
    page: int | None = None
    snippet: str = ""


# --------------------------------------------------------------------------
# Intent + actions (B.4)
# --------------------------------------------------------------------------
class IntentName(str, Enum):
    ANSWER_QUESTION = "answer_question"     # pure RAG, no side effects
    CHECK_BILL_BALANCE = "check_bill_balance"   # DB read
    CREATE_SUPPORT_TICKET = "create_support_ticket"  # DB write
    HUMAN_HANDOFF = "human_handoff"
    OUT_OF_SCOPE = "out_of_scope"

    @property
    def is_action(self) -> bool:
        """Action intents must go through permission checks before execution."""
        return self in (IntentName.CHECK_BILL_BALANCE, IntentName.CREATE_SUPPORT_TICKET)


class Intent(BaseModel):
    """The LLM's ONLY output in the action path.

    The model never touches the database. It emits this, and the backend
    decides whether the caller may do it (B.4 steps 3-4).
    """

    name: IntentName = IntentName.ANSWER_QUESTION
    confidence: float = 0.0
    slots: dict[str, Any] = Field(default_factory=dict)


class ActionOutcome(BaseModel):
    executed: bool
    intent: IntentName
    reason: str = ""             # why it was refused, when it was
    result: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Guardrails + PII (B.4 steps 1-2)
# --------------------------------------------------------------------------
class GuardrailVerdict(BaseModel):
    allowed: bool
    rule: str | None = None
    detail: str = ""


class PIIMaskResult(BaseModel):
    masked_text: str
    # placeholder -> original. Never leaves the backend; used to restore the
    # answer for display only (B.4 step 2).
    mapping: dict[str, str] = Field(default_factory=dict)

    @property
    def found(self) -> list[str]:
        return sorted({p.split("_")[1] for p in self.mapping if "_" in p})


# --------------------------------------------------------------------------
# Frustration + handoff (B.6)
# --------------------------------------------------------------------------
class FrustrationSignal(BaseModel):
    score: float = 0.0                       # 0..1
    reasons: list[str] = Field(default_factory=list)
    should_handoff: bool = False


class HandoffTicket(BaseModel):
    """The handoff *contract* — what actually transfers to a human (B.6)."""

    ticket_id: str
    session_id: str
    opened_at: datetime = Field(default_factory=_utcnow)
    reason: str = ""
    transcript: list[dict[str, str]] = Field(default_factory=list)
    detected_language: Language = Language.UNKNOWN
    last_intent: IntentName = IntentName.ANSWER_QUESTION
    agent_available: bool = True


# --------------------------------------------------------------------------
# Turn records + insights (B.11)
# --------------------------------------------------------------------------
class StageTimings(BaseModel):
    """Per-stage latency, so the budget in B.10 is measured and not asserted."""

    asr_ms: float | None = None
    guardrail_ms: float | None = None
    retrieval_ms: float | None = None
    rerank_ms: float | None = None
    generation_ms: float | None = None
    tts_ms: float | None = None
    total_ms: float = 0.0


class TurnRecord(BaseModel):
    """The structured output A.1 promises, emitted once per turn."""

    turn_id: str
    session_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    input_mode: Literal["voice", "text"] = "text"
    language: Language = Language.UNKNOWN
    transcript: str | None = None          # ASR output, voice turns only
    asr_confidence: float | None = None
    question: str = ""
    intent: IntentName = IntentName.ANSWER_QUESTION
    entities: dict[str, Any] = Field(default_factory=dict)
    pii_types_masked: list[str] = Field(default_factory=list)
    grounded: bool = False
    citations: list[Citation] = Field(default_factory=list)
    frustration: FrustrationSignal = Field(default_factory=FrustrationSignal)
    handed_off: bool = False
    action: ActionOutcome | None = None
    blocked_by_guardrail: bool = False
    resolution: Literal["answered", "refused", "escalated", "blocked", "no_answer"] = "answered"
    timings: StageTimings = Field(default_factory=StageTimings)
    profile: str = ""


# --------------------------------------------------------------------------
# API request/response
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str
    input_mode: Literal["voice", "text"] = "text"
    transcript_confidence: float | None = None
    language_hint: Language | None = None


class ChatResponse(BaseModel):
    """A.2: text is ALWAYS present. Voice is additive, never a replacement."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    language: Language = Language.UNKNOWN
    record: TurnRecord
    handoff: HandoffTicket | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: Language = Language.UNKNOWN
    confidence: float = 0.0
    duration_seconds: float = 0.0
    filtered_hallucination: bool = False


class UploadResponse(BaseModel):
    doc_name: str
    pages: int
    chunks: int
    used_ocr: bool = False
    language: Language = Language.UNKNOWN
