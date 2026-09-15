"""Whisper hallucination filtering.

Whisper's characteristic failure on silence or noise is not silence — it is a
fluent, confident sentence learned from the subtitle corpora in its training
data. In Arabic the recurring artifacts are subtitle credits ("ترجمة نانسي
قنقر", "اشترك في القناة"); in English they are "Thank you for watching" and
similar. On a customer-service line, background hiss becomes a question the
user never asked, and the assistant then answers it.

This mirrors the regex hallucination filters described in C.1, which is prior
art from exactly this problem on exactly this kind of audio.

Two detectors:
  * a known-artifact list, matched on normalised text;
  * a repetition detector, because the other common failure is a decode loop
    that emits the same phrase over and over.

VAD upstream prevents most of these. This is the second line, and it is cheap.
"""

from __future__ import annotations

import re
from collections import Counter

from ..retrieval.normalize_ar import normalize_arabic

# Subtitle-corpus artifacts. Matched after normalisation, so orthographic
# variants of the Arabic ones collapse to the same string.
KNOWN_ARTIFACTS: tuple[str, ...] = (
    "ترجمة نانسي قنقر",
    "ترجمة",
    "اشترك في القناة",
    "اشتركوا في القناة",
    "لا تنسى الاشتراك",
    "شكرا على المشاهدة",
    "شكرا لكم على المشاهدة",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "www.",
    ".com",
    "amara.org",
    "المزيد من الفيديوهات",
)

# A transcript that is *only* punctuation or filler is not speech.
FILLER_ONLY = re.compile(r"^[\s.,!?؟،…\-–—\"'()\[\]]*$")

MIN_MEANINGFUL_CHARS = 2
REPEAT_THRESHOLD = 0.6      # share of tokens taken by one repeated phrase
MIN_TOKENS_FOR_REPEAT = 6


def _is_known_artifact(normalized: str) -> bool:
    if not normalized:
        return False
    for artifact in KNOWN_ARTIFACTS:
        normalized_artifact = normalize_arabic(artifact)
        if not normalized_artifact:
            continue
        # Whole-transcript match, or an artifact dominating a short transcript:
        # a passing mention of "ترجمة" inside a real sentence is legitimate.
        if normalized == normalized_artifact:
            return True
        if normalized_artifact in normalized and len(normalized) < len(
            normalized_artifact
        ) * 2.5:
            return True
    return False


def _is_repetitive(text: str) -> bool:
    """Detect decode loops: one phrase repeated to fill the output."""
    tokens = normalize_arabic(text.lower()).split()
    if len(tokens) < MIN_TOKENS_FOR_REPEAT:
        return False

    # Single token dominating.
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(tokens) >= REPEAT_THRESHOLD:
        return True

    # A short n-gram repeated back to back.
    for size in (2, 3, 4):
        if len(tokens) < size * 3:
            continue
        grams = [
            " ".join(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1)
        ]
        gram_counts = Counter(grams)
        top_gram, top_count = gram_counts.most_common(1)[0]
        if top_count * size / len(tokens) >= REPEAT_THRESHOLD:
            return True
    return False


def filter_transcript(text: str) -> tuple[str, bool]:
    """Return (clean_text, was_hallucination).

    An empty string with `True` means "this was not real speech" — the caller
    asks the user to repeat rather than answering it.
    """
    if not text or not text.strip():
        return "", False

    stripped = text.strip()

    if FILLER_ONLY.match(stripped):
        return "", True

    normalized = normalize_arabic(stripped.lower())
    if len(normalized) < MIN_MEANINGFUL_CHARS:
        return "", True

    if _is_known_artifact(normalized):
        return "", True

    if _is_repetitive(stripped):
        return "", True

    return stripped, False
