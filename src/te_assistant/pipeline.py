"""The turn orchestrator — where the B.4 chain is actually sequenced.

The order below is the architecture, and it is fixed:

    1. input guardrails        before the model sees anything
    2. PII masking             the model never sees raw sensitive values
    3. intent detection        the model's ONLY job in the action path
    4. permission check        the backend decides, using no model output as authority
    5. execute or answer       action via the registry, or grounded RAG
    6. output guardrail        catch prompt-scaffolding leakage
    7. unmask for display      restore PII for the user who supplied it
    8. frustration + record    escalate if needed, emit the structured record

Steps 1 and 2 cannot be reordered: masking first would let an injection string
through to the masker, and guarding after the model defeats the purpose.

Everything is timed per stage so the latency budget in B.10 is measured rather
than estimated.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from .actions.db import ActionDB
from .actions.registry import ActionRegistry, describe_result
from .config import Settings, get_settings
from .frustration import detect as frustration
from .frustration.handoff import HandoffQueue, handoff_message, should_escalate
from .insights.records import RecordStore
from .llm.client import get_llm
from .llm.generate import generate_answer
from .llm.intent import detect_intent
from .monitoring.metrics import METRICS
from .retrieval.hybrid import HybridRetriever, diversify
from .retrieval.normalize_ar import detect_language, response_language
from .retrieval.rerank import get_reranker
from .schemas import (
    ChatRequest,
    ChatResponse,
    IntentName,
    Language,
    StageTimings,
    TurnRecord,
)
from .security import guardrails, pii
from .security.permissions import PermissionChecker
from .session.manager import SessionManager

log = logging.getLogger(__name__)


def _emit(services: Services, record: TurnRecord) -> None:
    """Persist the turn record and feed the metrics registry.

    One helper rather than two calls at each of the pipeline's several exit
    points — a turn that returns early (blocked, refused, escalated) must still
    be counted, and pairing them here is what stops one of those paths silently
    dropping out of the numbers.
    """
    services.records.add(record)
    METRICS.record_turn(record)


class _Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self._start) * 1000
        self._start = now
        return round(elapsed, 1)


@dataclass
class Services:
    """Everything a turn needs. Constructed once at startup."""

    settings: Settings
    sessions: SessionManager
    retriever: HybridRetriever
    db: ActionDB
    permissions: PermissionChecker
    actions: ActionRegistry
    handoffs: HandoffQueue
    records: RecordStore

    @classmethod
    def build(cls, settings: Settings | None = None) -> Services:
        settings = settings or get_settings()
        retriever = HybridRetriever(settings=settings)
        sessions = SessionManager(settings=settings, store=retriever.store)
        db = ActionDB(settings.sqlite_path)
        db.seed()
        return cls(
            settings=settings,
            sessions=sessions,
            retriever=retriever,
            db=db,
            permissions=PermissionChecker(db),
            actions=ActionRegistry(db),
            handoffs=HandoffQueue(),
            records=RecordStore(settings.traces_path),
        )

    def warmup(self) -> None:
        self.retriever.warmup()


def handle_turn(services: Services, request: ChatRequest) -> ChatResponse:
    settings = services.settings
    timer = _Timer()
    timings = StageTimings()
    turn_id = uuid.uuid4().hex[:12]

    session = services.sessions.get_or_create(request.session_id)
    detected = request.language_hint or detect_language(request.message)
    arabic = detected.is_arabic or detected is Language.UNKNOWN

    record = TurnRecord(
        turn_id=turn_id,
        session_id=session.session_id,
        input_mode=request.input_mode,
        language=detected,
        question=pii.scrub_for_logs(request.message),
        transcript=pii.scrub_for_logs(request.message) if request.input_mode == "voice" else None,
        asr_confidence=request.transcript_confidence,
        profile=settings.profile.value,
    )

    # -- 1. input guardrails ------------------------------------------------
    verdict = guardrails.check_input(request.message)
    timings.guardrail_ms = timer.lap()
    if not verdict.allowed:
        log.info("guardrail blocked turn %s: %s", turn_id, verdict.rule)
        answer = guardrails.refusal_message(arabic)
        record.blocked_by_guardrail = True
        record.resolution = "blocked"
        record.timings = timings
        timings.total_ms = timings.guardrail_ms
        _emit(services, record)
        session.history.add("user", request.message)
        session.history.add("assistant", answer)
        return ChatResponse(
            answer=answer, citations=[], language=detected, record=record
        )

    # -- 2. PII masking -----------------------------------------------------
    masked = pii.mask(request.message)
    record.pii_types_masked = masked.found
    question_for_model = masked.masked_text

    llm = get_llm(settings)

    # -- 3. intent detection ------------------------------------------------
    intent = detect_intent(llm, question_for_model)
    record.intent = intent.name
    record.entities = dict(intent.slots)

    # -- explicit handoff short-circuits ------------------------------------
    if intent.name is IntentName.HUMAN_HANDOFF:
        return _escalate(
            services, session, request, record, timings, timer,
            reason="customer asked for a human agent", arabic=arabic, detected=detected,
        )

    # -- 4/5. action path ---------------------------------------------------
    if intent.name.is_action:
        # Slots may carry PII placeholders; resolve them before the backend
        # looks anything up, since the DB stores real values.
        resolved_slots = {
            key: pii.unmask(str(value), masked.mapping)
            for key, value in intent.slots.items()
        }
        decision = services.permissions.check(
            session_id=session.session_id, intent=intent.name, slots=resolved_slots
        )
        intent.slots = resolved_slots
        outcome = services.actions.execute(
            intent=intent, session_id=session.session_id, decision=decision
        )
        record.action = outcome
        timings.generation_ms = timer.lap()

        if outcome.executed:
            answer = describe_result(outcome, arabic=arabic)
            record.resolution = "answered"
        else:
            answer = _refusal_for(outcome.reason, arabic)
            record.resolution = "refused"

        timings.total_ms = _total(timings)
        record.timings = timings
        session.history.add("user", request.message)
        session.history.add("assistant", answer)
        _emit(services, record)
        return ChatResponse(
            answer=answer, citations=[], language=detected, record=record
        )

    if intent.name is IntentName.OUT_OF_SCOPE:
        answer = guardrails.refusal_message(arabic)
        record.resolution = "refused"
        timings.total_ms = _total(timings)
        record.timings = timings
        session.history.add("user", request.message)
        session.history.add("assistant", answer)
        _emit(services, record)
        return ChatResponse(answer=answer, citations=[], language=detected, record=record)

    # -- 5b. grounded RAG ---------------------------------------------------
    retrieval = services.retriever.retrieve(
        question_for_model, session_id=session.session_id
    )
    timings.retrieval_ms = timer.lap()

    candidates = diversify(retrieval.chunks)
    reranker = get_reranker(settings)
    if reranker is not None and candidates:
        candidates = reranker.rerank(question_for_model, candidates, top_k=settings.rerank_k)
        timings.rerank_ms = timer.lap()
    else:
        candidates = candidates[: settings.rerank_k]

    answer_obj = generate_answer(
        llm,
        question=question_for_model,
        chunks=candidates,
        language=response_language(detected),
        history=session.history.as_messages(),
        max_tokens=settings.max_answer_tokens,
        temperature=settings.temperature,
    )
    timings.generation_ms = timer.lap()

    # -- 6. output guardrail ------------------------------------------------
    text = answer_obj.text
    if not guardrails.check_output(text).allowed:
        log.warning("output guardrail tripped on turn %s", turn_id)
        text = guardrails.refusal_message(arabic)
        answer_obj.citations = []
        answer_obj.grounded = False

    # -- 7. unmask for display ----------------------------------------------
    text = pii.unmask(text, masked.mapping)

    record.grounded = answer_obj.grounded
    record.citations = answer_obj.citations
    record.resolution = "answered" if answer_obj.grounded else "no_answer"

    # -- 8. frustration + record -------------------------------------------
    repeated = frustration.is_repeat(request.message, session.history.recent_questions())
    signals = frustration.TurnSignals(
        text=request.message,
        retrieval_confidence=answer_obj.retrieval_confidence,
        previous_questions=session.history.recent_questions(),
        low_confidence_streak=session.low_confidence_streak,
        repeat_streak=session.repeat_streak,
        answered=answer_obj.grounded,
    )
    signal = frustration.score(signals)
    record.frustration = signal

    session.low_confidence_streak, session.repeat_streak = frustration.update_streaks(
        retrieval_confidence=answer_obj.retrieval_confidence,
        repeated=repeated,
        low_streak=session.low_confidence_streak,
        repeat_streak=session.repeat_streak,
    )

    session.history.add("user", request.message)
    session.history.add("assistant", text)

    if should_escalate(signal, already_handed_off=session.handed_off):
        ticket = services.handoffs.open(
            session_id=session.session_id,
            reason="; ".join(signal.reasons) or "frustration detected",
            transcript=session.history.as_messages(limit=20),
            language=detected,
            last_intent=intent.name,
        )
        session.handed_off = True
        record.handed_off = True
        record.resolution = "escalated"
        text = f"{text}\n\n{handoff_message(ticket, arabic=arabic)}"
        timings.total_ms = _total(timings)
        record.timings = timings
        _emit(services, record)
        return ChatResponse(
            answer=text,
            citations=answer_obj.citations,
            language=detected,
            record=record,
            handoff=ticket,
        )

    timings.total_ms = _total(timings)
    record.timings = timings
    services.records.add(record)

    return ChatResponse(
        answer=text, citations=answer_obj.citations, language=detected, record=record
    )


def _escalate(
    services: Services, session, request, record, timings, timer, *,
    reason: str, arabic: bool, detected: Language,
) -> ChatResponse:
    ticket = services.handoffs.open(
        session_id=session.session_id,
        reason=reason,
        transcript=session.history.as_messages(limit=20),
        language=detected,
        last_intent=IntentName.HUMAN_HANDOFF,
    )
    session.handed_off = True
    answer = handoff_message(ticket, arabic=arabic)
    record.handed_off = True
    record.resolution = "escalated"
    timings.generation_ms = timer.lap()
    timings.total_ms = _total(timings)
    record.timings = timings
    session.history.add("user", request.message)
    session.history.add("assistant", answer)
    services.records.add(record)
    return ChatResponse(
        answer=answer, citations=[], language=detected, record=record, handoff=ticket
    )


def _refusal_for(reason: str, arabic: bool) -> str:
    """Explain a refusal without revealing whether an account exists."""
    if arabic:
        return (
            "معلش، مقدرش أنفذ الطلب ده على الحساب المطلوب. "
            "لو ده حسابك، سجّل دخولك الأول أو كلّم خدمة العملاء على 111."
        )
    return (
        "I can't perform that action on the requested account. "
        "If this is your account, please sign in first or call customer service on 111."
    )


def _total(timings: StageTimings) -> float:
    parts = [
        timings.asr_ms, timings.guardrail_ms, timings.retrieval_ms,
        timings.rerank_ms, timings.generation_ms, timings.tts_ms,
    ]
    return round(sum(p for p in parts if p), 1)
