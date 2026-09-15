"""Intent detection — step 3 of the B.4 chain.

This is the *only* thing the model contributes to the action path. It returns a
name and some slots. It does not decide whether the action is permitted, it does
not touch the database, and nothing downstream treats its output as authority —
`security/permissions.py` re-derives every fact it needs.

That framing is what makes the parsing strategy here acceptable. Small
quantised models emit imperfect JSON: fenced code blocks, a trailing sentence,
occasionally a near-miss intent name. We repair what we can and fall back to
`answer_question` when we cannot, because the safe default is the read-only
path. A parse failure must never fall back to an action.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..schemas import Intent, IntentName
from ..security.pii import PLACEHOLDER_RE
from .client import LLM
from .prompts import build_intent_messages

log = logging.getLogger(__name__)

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

MAX_INTENT_TOKENS = 128

# Cheap pre-checks. An explicit request for a human should not depend on a 3B
# model's classification, and these phrases are unambiguous.
HANDOFF_PATTERNS = re.compile(
    r"(?i)(speak|talk|connect)\s+(to|with)\s+(a\s+)?(human|person|agent|representative)"
    r"|\b(human|live)\s+agent\b"
    r"|(عايز|عاوز|أريد|اريد|ممكن)\s*(أكلم|اكلم|أتكلم|اتكلم|تحويل)\s*.{0,15}"
    r"(موظف|بشري|إنسان|انسان|خدمة العملاء|حد)"
    r"|حولني\s*(ل|إلى|الى)?\s*(موظف|حد|خدمة)"
)

SLOT_KEYS = {"msisdn", "customer_id", "subject", "body", "description"}


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = FENCE.sub("", raw.strip())
    match = JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        # Single-quoted keys are the most common malformation from small models.
        try:
            parsed = json.loads(match.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_intent_name(value: Any) -> IntentName:
    if not isinstance(value, str):
        return IntentName.ANSWER_QUESTION
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return IntentName(normalized)
    except ValueError:
        pass
    # Tolerate near-misses ("bill_balance", "support_ticket") without ever
    # widening this into fuzzy matching that could promote a read into a write.
    for candidate in IntentName:
        if normalized in candidate.value or candidate.value in normalized:
            return candidate
    return IntentName.ANSWER_QUESTION


def _clean_slots(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    slots: dict[str, Any] = {}
    for key, item in value.items():
        if key not in SLOT_KEYS or item in (None, "", []):
            continue
        if key == "msisdn":
            raw = str(item).strip()
            # A masked placeholder must survive verbatim — the backend unmasks it
            # to find the real number. Digit-stripping it would turn
            # "<PII_PHONE_1>" into "1", which unmasks to nothing and makes the
            # permission check fail to resolve an account that does exist.
            if PLACEHOLDER_RE.search(raw):
                slots[key] = raw
            else:
                slots[key] = re.sub(r"\D", "", raw) or raw
        else:
            slots[key] = str(item).strip()[:2000]
    return slots


def detect_intent(llm: LLM, message: str, *, temperature: float = 0.0) -> Intent:
    """Classify a message. Never raises — a failure degrades to the safe default."""
    if not message or not message.strip():
        return Intent(name=IntentName.OUT_OF_SCOPE, confidence=1.0)

    if HANDOFF_PATTERNS.search(message):
        return Intent(name=IntentName.HUMAN_HANDOFF, confidence=0.99)

    try:
        raw = llm.generate(
            build_intent_messages(message),
            max_tokens=MAX_INTENT_TOKENS,
            temperature=temperature,
            stop=["\n\n"],
        )
    except Exception:
        log.exception("intent detection failed; defaulting to answer_question")
        return Intent(name=IntentName.ANSWER_QUESTION, confidence=0.0)

    parsed = _extract_json(raw)
    if parsed is None:
        log.warning("intent output was not JSON: %r", raw[:120])
        return Intent(name=IntentName.ANSWER_QUESTION, confidence=0.0)

    name = _coerce_intent_name(parsed.get("intent"))

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    intent = Intent(name=name, confidence=confidence, slots=_clean_slots(parsed.get("slots")))

    # A low-confidence *action* is downgraded to the read-only path. Being wrong
    # about "answer_question" costs a mediocre answer; being wrong about
    # "create_support_ticket" costs a real write, so the two are not symmetric.
    if intent.name.is_action and intent.confidence < 0.55:
        log.info(
            "downgrading low-confidence action %s (%.2f) to answer_question",
            intent.name.value, intent.confidence,
        )
        return Intent(
            name=IntentName.ANSWER_QUESTION, confidence=intent.confidence, slots=intent.slots
        )

    return intent
