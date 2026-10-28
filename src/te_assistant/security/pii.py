"""PII masking — step 2 of the B.4 chain.

The requirement is precise: sensitive data must **never be visible to the model
directly**. So masking happens between the guardrail and the prompt, and the
mapping back to real values stays in the backend process. The model sees
`<PII_PHONE_1>`; only the response rendered to the user gets the real value
restored, and only for values that user themselves supplied.

Patterns are Egypt-specific on purpose. A generic phone regex either misses
`01012345678` or matches every 11-digit number in a tariff table; the national
ID pattern is validated structurally rather than by length alone, and cards go
through a Luhn check so that order numbers and quota figures are not masked as
payment data.

Why regex and not Presidio: Presidio pulls spaCy plus a language model (~500 MB)
and its Arabic support would need a model we are not otherwise loading. On the
cpu-lite profile that cost buys nothing over well-tested patterns, and these
patterns are directly asserted in `tests/test_security.py` (and, for masking of
retrieved passages and of what is persisted, `tests/test_pii_context.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import PIIMaskResult, ScoredChunk, SourceKind


@dataclass(frozen=True)
class PIIPattern:
    label: str
    pattern: re.Pattern[str]
    validator: str | None = None


# Egyptian mobile: 010/011/012/015 + 8 digits, optionally +20 / 0020 prefixed.
EG_MOBILE = re.compile(r"(?<!\d)(?:(?:\+?20|0020)\s?)?0?1[0125]\d{8}(?!\d)")
# Egyptian landline: area code (2-3 digits) + 7-8 digits.
EG_LANDLINE = re.compile(r"(?<!\d)0(?:2|3|4[0-8]|5[0-7]|6[2-9]|8[2-8]|9[2-7])\d{7,8}(?!\d)")
# National ID: century digit 2/3, YYMMDD, governorate, serial, checksum = 14.
EG_NATIONAL_ID = re.compile(r"(?<!\d)[23]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}(?!\d)")
EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
# Telecom Egypt account / service numbers as they appear on bills.
ACCOUNT_NO = re.compile(r"(?i)\b(?:account|acct|حساب|رقم\s*الحساب)\s*[:#]?\s*(\d{6,14})\b")

PATTERNS: tuple[PIIPattern, ...] = (
    PIIPattern("EMAIL", EMAIL),
    PIIPattern("NATIONALID", EG_NATIONAL_ID),
    PIIPattern("CARD", CARD, validator="luhn"),
    PIIPattern("IBAN", IBAN),
    PIIPattern("PHONE", EG_MOBILE),
    PIIPattern("PHONE", EG_LANDLINE),
    PIIPattern("ACCOUNT", ACCOUNT_NO),
)


def luhn_valid(digits: str) -> bool:
    """Standard Luhn check.

    Without this, any 13-19 digit run — a long reference number, a concatenated
    quota figure — would be masked as a payment card. False-positive masking is
    not harmless: it removes information the model needs to answer.
    """
    stripped = re.sub(r"\D", "", digits)
    if not 13 <= len(stripped) <= 19:
        return False
    total = 0
    parity = len(stripped) % 2
    for index, char in enumerate(stripped):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


_VALIDATORS = {"luhn": luhn_valid}


def mask(text: str, *, seed: dict[str, str] | None = None) -> PIIMaskResult:
    """Replace PII with stable placeholders.

    Repeated occurrences of the same value get the same placeholder, so the
    model can still reason about "the number you gave me" without ever seeing it.

    `seed` is a mapping from an earlier call. Masking the question and then the
    retrieved passages must share one numbering: two independent calls would
    both emit `<PII_PHONE_1>` for different numbers, and unmasking the answer
    would then restore the wrong one. With a seed, numbering continues and a
    value already seen reuses its placeholder. The returned mapping is the seed
    plus anything new.
    """
    mapping: dict[str, str] = dict(seed or {})
    if not text:
        return PIIMaskResult(masked_text="", mapping=mapping)

    reverse: dict[str, str] = {value: key for key, value in mapping.items()}
    counters: dict[str, int] = {}
    for placeholder in mapping:
        parsed = _PLACEHOLDER_PARTS.fullmatch(placeholder)
        if parsed:
            label, number = parsed.group(1), int(parsed.group(2))
            counters[label] = max(counters.get(label, 0), number)
    masked = text

    for spec in PATTERNS:
        validator = _VALIDATORS.get(spec.validator or "")

        def _replace(match: re.Match[str], label: str = spec.label, check=validator) -> str:
            value = match.group(0)
            if check and not check(value):
                return value
            if value in reverse:
                return reverse[value]
            counters[label] = counters.get(label, 0) + 1
            placeholder = f"<PII_{label}_{counters[label]}>"
            mapping[placeholder] = value
            reverse[value] = placeholder
            return placeholder

        masked = spec.pattern.sub(_replace, masked)

    return PIIMaskResult(masked_text=masked, mapping=mapping)


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Restore originals for display only — never before sending to the model."""
    if not mapping:
        return text
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


PLACEHOLDER_RE = re.compile(r"<PII_[A-Z]+_\d+>")
_PLACEHOLDER_PARTS = re.compile(r"<PII_([A-Z]+)_(\d+)>")


def contains_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_RE.search(text))


def mask_chunks(
    chunks: list[ScoredChunk], mapping: dict[str, str]
) -> tuple[list[ScoredChunk], dict[str, str]]:
    """Mask the customer-supplied passages that are about to enter a prompt.

    Masking the question alone does not meet the requirement that sensitive
    data is never visible to the model: when a user uploads their bill and asks
    about it, the number printed on the bill is retrieved straight into the
    prompt. So passages from user documents are masked too, continuing the
    question's numbering (see `mask`), and the caller unmasks the answer with
    the returned mapping.

    Public te.eg content is deliberately left alone. Its phone numbers and
    addresses are the *answer* — "call 19777", "write to Customer.care@te.eg" —
    and masking them would remove information the model needs.

    Returns copies; the stored chunks are never modified.
    """
    combined = dict(mapping)
    out: list[ScoredChunk] = []
    for scored in chunks:
        chunk = scored.chunk
        if chunk.source_kind is not SourceKind.USER_DOCUMENT:
            out.append(scored)
            continue

        def _masked(value: str | None) -> str | None:
            nonlocal combined
            if not value:
                return value
            result = mask(value, seed=combined)
            combined = result.mapping
            return result.masked_text

        updated = chunk.model_copy(update={
            "text": _masked(chunk.text) or "",
            "title": _masked(chunk.title) or "",
            "doc_name": _masked(chunk.doc_name),
        })
        out.append(scored.model_copy(update={"chunk": updated}))
    return out, combined


def scrub_for_logs(text: str) -> str:
    """Mask before writing anywhere persistent.

    Traces and turn records are written to disk and read during evaluation, so
    they must not become the place PII ends up.
    """
    return mask(text).masked_text
