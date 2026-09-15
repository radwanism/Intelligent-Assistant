"""Intent parsing robustness and the service degradation contract.

Two themes:

**Intent parsing** — small quantised models emit imperfect JSON. The safe
default on any parse failure is the read-only path; a malformed response must
never resolve to an action. These tests drive a fake LLM so the behaviour is
asserted without loading a model.

**Degradation** — the plan claims specific failures are survivable. A claim
that is never exercised is a guess, so each one is tested here.
"""

from __future__ import annotations

import pytest

from te_assistant.llm.generate import extract_citations
from te_assistant.llm.intent import detect_intent
from te_assistant.schemas import Chunk, IntentName, Language, ScoredChunk, SourceKind


class FakeLLM:
    """Returns a scripted response, or raises, to drive the failure paths."""

    def __init__(self, response: str = "", *, raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.calls = 0

    def generate(self, messages, *, max_tokens, temperature, stop=None) -> str:
        self.calls += 1
        if self.raises:
            raise RuntimeError("model unavailable")
        return self.response

    def stream(self, messages, *, max_tokens, temperature, stop=None):
        yield self.response


# --------------------------------------------------------------------------
# Intent parsing
# --------------------------------------------------------------------------
def test_clean_json_is_parsed() -> None:
    llm = FakeLLM('{"intent": "check_bill_balance", "confidence": 0.9, "slots": {}}')
    intent = detect_intent(llm, "كام فاتورتي؟")
    assert intent.name is IntentName.CHECK_BILL_BALANCE
    assert intent.confidence == pytest.approx(0.9)


def test_fenced_json_is_parsed() -> None:
    """Small models wrap JSON in code fences constantly."""
    llm = FakeLLM('```json\n{"intent": "answer_question", "confidence": 0.8}\n```')
    assert detect_intent(llm, "ايه الباقات؟").name is IntentName.ANSWER_QUESTION


def test_json_with_trailing_prose_is_parsed() -> None:
    llm = FakeLLM(
        '{"intent": "answer_question", "confidence": 0.7}\n'
        "I hope this helps with your question!"
    )
    assert detect_intent(llm, "ايه الباقات؟").name is IntentName.ANSWER_QUESTION


@pytest.mark.parametrize(
    "garbage",
    ["not json at all", "", "{broken", "[1, 2, 3]", "null"],
)
def test_unparseable_output_falls_back_to_the_safe_path(garbage: str) -> None:
    """A parse failure must never resolve to an action."""
    intent = detect_intent(FakeLLM(garbage), "كام فاتورتي؟")
    assert intent.name is IntentName.ANSWER_QUESTION
    assert not intent.name.is_action


def test_model_exception_falls_back_to_the_safe_path() -> None:
    intent = detect_intent(FakeLLM(raises=True), "افتح لي شكوى")
    assert intent.name is IntentName.ANSWER_QUESTION


def test_low_confidence_action_is_downgraded() -> None:
    """Being wrong about a write is not symmetric with being wrong about a read."""
    llm = FakeLLM('{"intent": "create_support_ticket", "confidence": 0.3, "slots": {}}')
    assert detect_intent(llm, "ممكن؟").name is IntentName.ANSWER_QUESTION


def test_high_confidence_action_survives() -> None:
    llm = FakeLLM('{"intent": "create_support_ticket", "confidence": 0.95, "slots": {}}')
    assert detect_intent(llm, "افتح شكوى").name is IntentName.CREATE_SUPPORT_TICKET


def test_explicit_handoff_bypasses_the_model_entirely() -> None:
    """A clear request for a human should not depend on a 3B model's judgement."""
    llm = FakeLLM('{"intent": "answer_question", "confidence": 0.9}')
    intent = detect_intent(llm, "I want to speak to a human agent")
    assert intent.name is IntentName.HUMAN_HANDOFF
    assert llm.calls == 0, "should not have called the model at all"


def test_unknown_slots_are_discarded() -> None:
    llm = FakeLLM(
        '{"intent": "create_support_ticket", "confidence": 0.9, '
        '"slots": {"subject": "No internet", "evil": "DROP TABLE", "admin": true}}'
    )
    slots = detect_intent(llm, "افتح شكوى").slots
    assert slots == {"subject": "No internet"}


def test_masked_pii_placeholder_survives_into_slots() -> None:
    """The model must be able to reference a value it cannot see."""
    llm = FakeLLM(
        '{"intent": "check_bill_balance", "confidence": 0.9, '
        '"slots": {"msisdn": "<PII_PHONE_1>"}}'
    )
    assert detect_intent(llm, "رصيد <PII_PHONE_1>").slots["msisdn"] == "<PII_PHONE_1>"


# --------------------------------------------------------------------------
# Citation resolution
# --------------------------------------------------------------------------
def _chunk(index: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"c{index}",
            text=f"Passage number {index} about WE plans.",
            title=f"Page {index}",
            url=f"https://te.eg/page-{index}",
            language=Language.EN,
            source_kind=SourceKind.TE_EG,
        ),
        score=1.0,
    )


def test_citations_resolve_to_real_sources() -> None:
    chunks = [_chunk(i) for i in range(1, 4)]
    text, citations = extract_citations("The plan costs 300 EGP [1] and includes data [3].", chunks)
    assert len(citations) == 2
    assert citations[0].url == "https://te.eg/page-1"
    assert citations[1].url == "https://te.eg/page-3"


def test_citations_are_renumbered_contiguously() -> None:
    """Citing [1] and [3] should display as [1] and [2], not [1] and [3]."""
    chunks = [_chunk(i) for i in range(1, 4)]
    text, citations = extract_citations("A [1] and B [3].", chunks)
    assert "[1]" in text and "[2]" in text
    assert "[3]" not in text
    assert [c.marker for c in citations] == ["[1]", "[2]"]


def test_invented_citation_markers_are_dropped() -> None:
    """A marker for a passage that was never supplied cannot invent a source."""
    chunks = [_chunk(1)]
    text, citations = extract_citations("Real [1] but invented [9].", chunks)
    assert len(citations) == 1
    assert "[9]" not in text


def test_answer_without_citations_returns_none() -> None:
    text, citations = extract_citations("An answer with no markers.", [_chunk(1)])
    assert citations == []
