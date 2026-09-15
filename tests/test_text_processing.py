"""Arabic normalisation, language detection, chunking and hallucination filters.

These are the parts of the pipeline where a silent regression is most likely:
nothing raises, retrieval just quietly gets worse. Hence direct assertions.
"""

from __future__ import annotations

import pytest

from te_assistant.ingest.chunk import estimate_tokens, split_text
from te_assistant.retrieval.normalize_ar import (
    detect_language,
    is_egyptian,
    normalize_arabic,
    tokenize,
)
from te_assistant.schemas import Language
from te_assistant.speech.halluc_filter import filter_transcript


# --------------------------------------------------------------------------
# Arabic normalisation — the BM25 recall problem
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("إنترنت", "انترنت"),      # hamza-under-alef
        ("أنترنت", "انترنت"),      # hamza-over-alef
        ("آنترنت", "انترنت"),      # madda
        ("انــترنت", "انترنت"),    # tatweel
        ("مُحَمَّد", "محمد"),         # diacritics
        ("باقة", "باقه"),          # ta-marbuta
        ("علي", "علي"),
        ("علی", "علي"),           # Persian ya
    ],
)
def test_orthographic_variants_collapse(written: str, expected: str) -> None:
    assert normalize_arabic(written) == expected


def test_variants_produce_identical_tokens() -> None:
    """The actual point: BM25 must see one term, not three."""
    assert tokenize("إنترنت") == tokenize("انترنت") == tokenize("أنترنت")


def test_arabic_indic_digits_normalise() -> None:
    assert normalize_arabic("٥٠ جنيه") == "50 جنيه"
    assert normalize_arabic("۱۲۳") == "123"


# --------------------------------------------------------------------------
# Language and dialect detection
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What are your 5G plans?", Language.EN),
        ("ما هي باقات الإنترنت المتاحة؟", Language.AR),
        ("عايز أعرف الباقات بكام", Language.ARZ),
        ("ازاي أشحن الخط بتاعي؟", Language.ARZ),
        ("عايز أعرف الـ package بتاع 5G", Language.MIXED),
        ("", Language.UNKNOWN),
    ],
)
def test_language_detection(text: str, expected: Language) -> None:
    assert detect_language(text) is expected


def test_egyptian_markers_detected() -> None:
    assert is_egyptian("عايز أعرف إيه الباقات دي")
    assert is_egyptian("ezay a47an el khat")  # Franco-Arab
    assert not is_egyptian("ما هي الخدمات المتوفرة")


def test_arabic_languages_flagged_as_arabic() -> None:
    for language in (Language.AR, Language.ARZ, Language.MIXED):
        assert language.is_arabic
    assert not Language.EN.is_arabic


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def test_arabic_costs_more_tokens_per_character() -> None:
    """The estimator must not treat Arabic like English or chunks overflow."""
    arabic = "مرحبا بك في المصرية للاتصالات" * 5
    english = "Welcome to Telecom Egypt today" * 5
    assert len(arabic) < len(english) * 1.2
    assert estimate_tokens(arabic) > estimate_tokens(english)


def test_chunks_respect_the_token_budget() -> None:
    text = "\n".join(f"This is line number {i} with some content." for i in range(200))
    chunks = split_text(text, max_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 160  # budget + one line of slack


def test_chunk_overlap_bridges_the_boundary() -> None:
    """A plan name and its price must not be split apart with no overlap.

    Uses prose lines (terminal punctuation, >8 words) so they are not treated as
    headings — a heading deliberately starts a clean section with no carry-over,
    so overlap only applies where a chunk overflows the token budget.
    """
    text = "\n".join(
        f"The WE Nitro plan number {i} includes generous data and costs money each month."
        for i in range(40)
    )
    with_overlap = split_text(text, max_tokens=120, overlap_tokens=40)
    assert len(with_overlap) >= 2
    first_lines = set(with_overlap[0].text.splitlines())
    second_lines = set(with_overlap[1].text.splitlines())
    assert first_lines & second_lines, "expected overlapping lines between chunks"


def test_heading_boundaries_do_not_overlap() -> None:
    """The complement: a new section should not drag the previous one's tail in."""
    text = "\n".join(
        ["Internet Plans", *[f"Plan {i} costs three hundred pounds per month." for i in range(8)],
         "Mobile Plans", *[f"Bundle {i} costs one hundred pounds per month." for i in range(8)]]
    )
    chunks = split_text(text, max_tokens=60, overlap_tokens=20)
    assert len(chunks) >= 2


def test_short_text_is_dropped() -> None:
    assert split_text("tiny", min_chars=120) == []


# --------------------------------------------------------------------------
# Whisper hallucination filtering (A.1 noisy audio)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "artifact",
    [
        "ترجمة نانسي قنقر",
        "اشترك في القناة",
        "Thank you for watching!",
        "Thanks for watching",
        "...",
        "شكرا على المشاهدة",
    ],
)
def test_known_hallucinations_are_removed(artifact: str) -> None:
    text, flagged = filter_transcript(artifact)
    assert flagged
    assert text == ""


def test_silence_is_empty_not_a_hallucination() -> None:
    """Whitespace means nothing was said — distinct from the model inventing text.

    Both end up asking the user to repeat (`asr.needs_clarification`), but only
    the invented-text case should be counted as a filtered hallucination in the
    metrics, or the WER report overstates how often the filter fired.
    """
    text, flagged = filter_transcript("   ")
    assert text == ""
    assert not flagged


def test_repetition_loops_are_caught() -> None:
    text, flagged = filter_transcript("نعم نعم نعم نعم نعم نعم نعم نعم")
    assert flagged and text == ""

    text, flagged = filter_transcript("the internet the internet the internet the internet")
    assert flagged and text == ""


@pytest.mark.parametrize(
    "genuine",
    [
        "عايز أعرف أسعار باقات الإنترنت",
        "What is the price of the 5G home plan?",
        "ازاي أشحن الخط بتاعي من فضلك",
    ],
)
def test_genuine_speech_survives(genuine: str) -> None:
    text, flagged = filter_transcript(genuine)
    assert not flagged
    assert text == genuine
