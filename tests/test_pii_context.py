"""Masking beyond the question: retrieved passages, and the persistence boundary."""

from __future__ import annotations

from te_assistant.insights.records import RecordStore, for_storage
from te_assistant.schemas import (
    ActionOutcome,
    Chunk,
    Citation,
    IntentName,
    ScoredChunk,
    SourceKind,
    TurnRecord,
)
from te_assistant.security import pii

QUESTION_PHONE = "01012345678"
DOC_PHONE = "01198765432"


def _scored(text: str, kind: SourceKind, **extra) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id="c", text=text, source_kind=kind, **extra), score=1.0
    )


# --------------------------------------------------------------------------
# mask(seed=...) — one numbering across the question and the passages
# --------------------------------------------------------------------------
def test_seeded_masking_continues_the_numbering() -> None:
    question = pii.mask(f"my number is {QUESTION_PHONE}")
    passage = pii.mask(f"call {DOC_PHONE}", seed=question.mapping)

    assert "<PII_PHONE_1>" in question.masked_text
    assert "<PII_PHONE_2>" in passage.masked_text, "must not reuse a placeholder for a new value"
    assert passage.mapping["<PII_PHONE_1>"] == QUESTION_PHONE
    assert passage.mapping["<PII_PHONE_2>"] == DOC_PHONE


def test_seeded_masking_reuses_the_placeholder_for_a_known_value() -> None:
    question = pii.mask(f"my number is {QUESTION_PHONE}")
    passage = pii.mask(f"registered to {QUESTION_PHONE}", seed=question.mapping)
    assert "<PII_PHONE_1>" in passage.masked_text
    assert len(passage.mapping) == 1


def test_unmasking_with_the_combined_mapping_restores_the_right_value() -> None:
    """Two independent masks would both say PHONE_1 — and restore the wrong number."""
    question = pii.mask(f"my number is {QUESTION_PHONE}")
    passage = pii.mask(f"call {DOC_PHONE}", seed=question.mapping)
    answer = "You gave <PII_PHONE_1>; the bill lists <PII_PHONE_2>."
    restored = pii.unmask(answer, passage.mapping)
    assert QUESTION_PHONE in restored and DOC_PHONE in restored
    assert restored.index(QUESTION_PHONE) < restored.index(DOC_PHONE)


def test_unseeded_mask_is_unchanged() -> None:
    """The original contract still holds for every existing caller."""
    result = pii.mask(f"call {QUESTION_PHONE}")
    assert result.mapping == {"<PII_PHONE_1>": QUESTION_PHONE}
    assert pii.mask("").mapping == {}


# --------------------------------------------------------------------------
# mask_chunks — customer documents are masked, public content is not
# --------------------------------------------------------------------------
def test_user_document_passages_are_masked() -> None:
    chunk = _scored(f"Line {DOC_PHONE} is billed 300 EGP", SourceKind.USER_DOCUMENT,
                    session_id="s", doc_name="bill.pdf")
    masked, mapping = pii.mask_chunks([chunk], {})
    assert DOC_PHONE not in masked[0].chunk.text
    assert "300 EGP" in masked[0].chunk.text, "non-PII facts must survive"
    assert mapping and DOC_PHONE in mapping.values()


def test_public_te_eg_content_is_not_masked() -> None:
    """A hotline or support address on te.eg is the answer, not a secret."""
    text = "Write to Customer.care@te.eg or call 0227402000"
    chunk = _scored(text, SourceKind.TE_EG, url="https://te.eg/contact")
    masked, mapping = pii.mask_chunks([chunk], {})
    assert masked[0].chunk.text == text
    assert mapping == {}


def test_mask_chunks_returns_copies() -> None:
    chunk = _scored(f"call {DOC_PHONE}", SourceKind.USER_DOCUMENT, session_id="s")
    pii.mask_chunks([chunk], {})
    assert DOC_PHONE in chunk.chunk.text, "the stored chunk must never be modified"


def test_file_names_carrying_pii_are_masked_too() -> None:
    chunk = _scored("ordinary text", SourceKind.USER_DOCUMENT, session_id="s",
                    title=f"bill-{DOC_PHONE}.pdf", doc_name=f"bill-{DOC_PHONE}.pdf")
    masked, _ = pii.mask_chunks([chunk], {})
    assert DOC_PHONE not in masked[0].chunk.title
    assert DOC_PHONE not in (masked[0].chunk.doc_name or "")


# --------------------------------------------------------------------------
# for_storage — the trace file is where PII must not end up
# --------------------------------------------------------------------------
def _record(**kwargs) -> TurnRecord:
    return TurnRecord(turn_id="t", session_id="s", **kwargs)


def test_free_text_is_masked_at_the_write_even_if_the_caller_forgot() -> None:
    record = _record(
        question=f"call me on {QUESTION_PHONE}",
        transcript=f"my number {QUESTION_PHONE}",
        entities={"msisdn": QUESTION_PHONE, "subject": "Outage"},
        citations=[Citation(
            marker="[1]", title="t", snippet=f"line {DOC_PHONE}", doc_name=f"{DOC_PHONE}.pdf"
        )],
    )
    stored = for_storage(record)
    blob = stored.model_dump_json()
    assert QUESTION_PHONE not in blob and DOC_PHONE not in blob
    assert stored.entities["subject"] == "Outage", "non-PII values survive"


def test_action_result_is_reduced_to_field_names() -> None:
    record = _record(action=ActionOutcome(
        executed=True, intent=IntentName.CHECK_BILL_BALANCE,
        result={"plan": "WE Nitro 200", "balance_egp": 213.5},
    ))
    stored = for_storage(record)
    assert stored.action.executed is True
    assert stored.action.result == {"redacted": True, "fields": ["balance_egp", "plan"]}
    assert "213.5" not in stored.model_dump_json()


def test_for_storage_does_not_modify_the_original() -> None:
    record = _record(action=ActionOutcome(
        executed=True, intent=IntentName.CHECK_BILL_BALANCE, result={"balance_egp": 1.0},
    ))
    for_storage(record)
    assert record.action.result == {"balance_egp": 1.0}, "the user still gets their own result"


def test_record_store_persists_the_redacted_form(tmp_path) -> None:
    store = RecordStore(tmp_path / "traces.jsonl")
    store.add(_record(
        question=f"call {QUESTION_PHONE}",
        action=ActionOutcome(executed=True, intent=IntentName.CHECK_BILL_BALANCE,
                             result={"balance_egp": 213.5}),
    ))
    assert QUESTION_PHONE not in (tmp_path / "traces.jsonl").read_text(encoding="utf-8")
    assert "213.5" not in (tmp_path / "traces.jsonl").read_text(encoding="utf-8")
    assert store.for_session("s")[0].action.executed is True
