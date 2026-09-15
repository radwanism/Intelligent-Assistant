"""Frustration detection — brief B.6.

Design choice worth defending: this is **signal-based, not model-based**.

An LLM judgement on every turn would roughly double per-turn latency on the CPU
profile, for a decision that is mostly determined by things we already know for
free — whether retrieval keeps failing, whether the user is asking the same
thing again, whether they swore. MARBERT-style Arabic sentiment (C.4) is the
natural upgrade and is named in Future Work; it is not needed to make the
mechanism real, and the handoff path is the part that actually matters.

Signals are combined additively with a cap, rather than by taking a maximum, so
that several mild indications (one repeat, one retrieval miss, mild negative
wording) can escalate — which is what a genuinely stuck conversation looks like,
as opposed to one sharp word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..retrieval.normalize_ar import normalize_arabic
from ..schemas import FrustrationSignal

HANDOFF_THRESHOLD = 0.6

# Explicit escalation requests. These alone are sufficient — if someone asks for
# a human, inferring a "frustration score" first would be perverse.
EXPLICIT_HANDOFF = re.compile(
    r"(?i)\b(human|agent|representative|supervisor|manager)\b"
    r"|(موظف|بشري|مسؤول|مشرف|خدمة العملاء)"
)

# Negative sentiment markers, Egyptian dialect first since that is the register
# frustrated customers actually use.
NEGATIVE_AR = frozenset({
    "زهقت", "زهقان", "تعبت", "مليت", "فاشل", "فشل", "وحش", "سيء", "سيئ", "زفت",
    "مش عاجبني", "مستاء", "غضبان", "معصب", "بطيء", "بطيئة", "مش شغال", "مبيشتغلش",
    "كارثة", "مصيبة", "حرام", "استغلال", "نصب", "مش راضي", "شكوى", "بشتكي",
})
NEGATIVE_EN = frozenset({
    "useless", "terrible", "awful", "horrible", "ridiculous", "unacceptable",
    "frustrated", "frustrating", "annoyed", "angry", "disappointed", "worst",
    "broken", "doesn't work", "does not work", "not working", "waste",
})

# Repeated punctuation and full-caps are register signals, not content.
SHOUTING = re.compile(r"[!؟?]{2,}|[A-Z]{5,}")
PROFANITY_HINT = re.compile(r"(?i)\b(wtf|damn|hell)\b|(زفت|خرا|لعنة)")


@dataclass
class TurnSignals:
    """What the orchestrator observed on this turn."""

    text: str
    retrieval_confidence: float
    previous_questions: list[str]
    low_confidence_streak: int
    repeat_streak: int
    answered: bool = True


def _similar(a: str, b: str) -> float:
    """Token Jaccard on normalised text.

    Cheap and adequate: we are detecting "they asked the same thing again",
    which shows up as high lexical overlap. Embedding the history for this would
    cost a model call per turn to answer a question overlap already answers.
    """
    tokens_a = set(normalize_arabic(a.lower()).split())
    tokens_b = set(normalize_arabic(b.lower()).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def is_repeat(text: str, previous: list[str], *, threshold: float = 0.6) -> bool:
    return any(_similar(text, earlier) >= threshold for earlier in previous)


def _negative_hits(text: str) -> int:
    lowered = text.lower()
    normalized = normalize_arabic(lowered)
    hits = sum(1 for phrase in NEGATIVE_EN if phrase in lowered)
    hits += sum(1 for phrase in NEGATIVE_AR if normalize_arabic(phrase) in normalized)
    return hits


def score(signals: TurnSignals) -> FrustrationSignal:
    """Combine signals into a 0..1 score plus the reasons behind it.

    The reasons are not decoration — they go into the handoff ticket so the human
    agent sees *why* the conversation was escalated without reading the whole
    transcript.
    """
    if EXPLICIT_HANDOFF.search(signals.text) and re.search(
        r"(?i)(want|need|talk|speak|عايز|عاوز|ممكن|حولني)", signals.text
    ):
        return FrustrationSignal(
            score=1.0, reasons=["explicit request for a human agent"], should_handoff=True
        )

    total = 0.0
    reasons: list[str] = []

    negatives = _negative_hits(signals.text)
    if negatives:
        total += min(0.20 * negatives, 0.40)
        reasons.append(f"negative sentiment ({negatives} marker(s))")

    if PROFANITY_HINT.search(signals.text):
        total += 0.15
        reasons.append("profanity")

    if SHOUTING.search(signals.text):
        total += 0.10
        reasons.append("emphatic punctuation or shouting")

    if is_repeat(signals.text, signals.previous_questions):
        total += 0.25
        reasons.append("question repeated")
    if signals.repeat_streak >= 2:
        total += 0.20
        reasons.append(f"repeated {signals.repeat_streak} times")

    if signals.retrieval_confidence < 0.2:
        total += 0.15
        reasons.append("retrieval found nothing relevant")
    if signals.low_confidence_streak >= 2:
        total += 0.25
        reasons.append(f"{signals.low_confidence_streak} unhelpful answers in a row")

    if not signals.answered:
        total += 0.10
        reasons.append("assistant could not answer")

    total = min(total, 1.0)
    return FrustrationSignal(
        score=round(total, 3), reasons=reasons, should_handoff=total >= HANDOFF_THRESHOLD
    )


def update_streaks(
    *, retrieval_confidence: float, repeated: bool, low_streak: int, repeat_streak: int
) -> tuple[int, int]:
    """Roll the per-session counters. Reset on success, so one bad turn in an
    otherwise healthy conversation does not accumulate toward escalation."""
    new_low = low_streak + 1 if retrieval_confidence < 0.2 else 0
    new_repeat = repeat_streak + 1 if repeated else 0
    return new_low, new_repeat
