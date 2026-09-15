"""Frustration detection and human handoff — brief B.6."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from te_assistant.frustration import detect
from te_assistant.frustration.handoff import (
    HandoffQueue,
    handoff_message,
    should_escalate,
)
from te_assistant.schemas import FrustrationSignal, IntentName, Language


def signals(text: str, **overrides) -> detect.TurnSignals:
    base = dict(
        text=text,
        retrieval_confidence=0.8,
        previous_questions=[],
        low_confidence_streak=0,
        repeat_streak=0,
        answered=True,
    )
    base.update(overrides)
    return detect.TurnSignals(**base)


# --------------------------------------------------------------------------
# Explicit requests short-circuit
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "I want to talk to a human agent",
        "can I speak with a representative please",
        "عايز أكلم موظف",
        "ممكن تحولني لخدمة العملاء",
    ],
)
def test_explicit_request_escalates_immediately(text: str) -> None:
    """Asking for a human should not require a frustration score first."""
    result = detect.score(signals(text))
    assert result.should_handoff
    assert result.score == 1.0


# --------------------------------------------------------------------------
# Accumulating evidence
# --------------------------------------------------------------------------
def test_a_calm_question_does_not_escalate() -> None:
    result = detect.score(signals("عايز أعرف أسعار باقات الإنترنت"))
    assert not result.should_handoff
    assert result.score == 0.0


def test_negative_sentiment_alone_is_not_enough() -> None:
    """One sharp word is not a stuck conversation."""
    result = detect.score(signals("this is terrible"))
    assert result.score > 0
    assert not result.should_handoff


def test_mild_signals_accumulate_to_escalation() -> None:
    """Several mild indications together is what being stuck looks like."""
    result = detect.score(
        signals(
            "النت مش شغال خالص وده سيء",
            retrieval_confidence=0.05,
            previous_questions=["النت مش شغال خالص"],
            low_confidence_streak=2,
            repeat_streak=2,
            answered=False,
        )
    )
    assert result.should_handoff
    assert len(result.reasons) >= 3


def test_repeated_question_is_detected() -> None:
    assert detect.is_repeat(
        "عايز أعرف أسعار باقات الإنترنت",
        ["عايز أعرف أسعار باقات الإنترنت النهاردة"],
    )
    assert not detect.is_repeat(
        "عايز أعرف أسعار الباقات", ["فين أقرب فرع ليا"]
    )


def test_reasons_are_recorded_for_the_agent() -> None:
    """The human agent needs to know why, without reading the transcript."""
    result = detect.score(
        signals("زهقت من الموضوع ده", retrieval_confidence=0.05, low_confidence_streak=3)
    )
    assert result.reasons
    assert any("unhelpful" in reason for reason in result.reasons)


# --------------------------------------------------------------------------
# Streaks reset on success
# --------------------------------------------------------------------------
def test_streaks_reset_after_a_good_turn() -> None:
    low, repeat = detect.update_streaks(
        retrieval_confidence=0.9, repeated=False, low_streak=3, repeat_streak=2
    )
    assert low == 0 and repeat == 0


def test_streaks_grow_on_consecutive_failures() -> None:
    low, repeat = detect.update_streaks(
        retrieval_confidence=0.05, repeated=True, low_streak=1, repeat_streak=1
    )
    assert low == 2 and repeat == 2


# --------------------------------------------------------------------------
# The handoff contract
# --------------------------------------------------------------------------
def test_ticket_carries_what_the_agent_needs() -> None:
    queue = HandoffQueue()
    ticket = queue.open(
        session_id="s1",
        reason="3 unhelpful answers in a row",
        transcript=[{"role": "user", "content": "النت مش شغال"}],
        language=Language.ARZ,
        last_intent=IntentName.ANSWER_QUESTION,
    )
    assert ticket.ticket_id.startswith("HO-")
    assert ticket.transcript, "the agent must not have to ask the customer to repeat"
    assert ticket.detected_language is Language.ARZ
    assert ticket.reason


def test_escalation_happens_once_per_session() -> None:
    """Re-escalating every turn would flood the queue with duplicates."""
    signal = FrustrationSignal(score=0.9, should_handoff=True)
    assert should_escalate(signal, already_handed_off=False)
    assert not should_escalate(signal, already_handed_off=True)


def test_no_agent_available_still_gives_the_user_a_path() -> None:
    """Out of hours, the conversation must not dead-end."""
    queue = HandoffQueue()
    ticket = queue.open(
        session_id="s1", reason="test", transcript=[],
        language=Language.AR, last_intent=IntentName.HUMAN_HANDOFF,
    )
    ticket.agent_available = False

    message = handoff_message(ticket, arabic=False)
    assert ticket.ticket_id in message
    assert "09:00" in message  # tells them when someone will be there


def test_agent_availability_respects_business_hours() -> None:
    queue = HandoffQueue()
    # 03:00 UTC == 05:00 Cairo — outside the 09:00-22:00 window.
    night = datetime(2026, 9, 15, 3, 0, tzinfo=UTC)
    assert not queue.agent_available(now=night)
    # 12:00 UTC == 14:00 Cairo.
    day = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
    assert queue.agent_available(now=day)


def test_handoff_message_matches_the_users_language() -> None:
    queue = HandoffQueue()
    ticket = queue.open(
        session_id="s1", reason="test", transcript=[],
        language=Language.ARZ, last_intent=IntentName.HUMAN_HANDOFF,
    )
    arabic = handoff_message(ticket, arabic=True)
    english = handoff_message(ticket, arabic=False)
    assert arabic != english
    assert ticket.ticket_id in arabic and ticket.ticket_id in english
