"""Arabic normalisation and language/dialect detection.

Two jobs, both cheap and both load-bearing:

1. `normalize_arabic` — BM25 is a *lexical* matcher, so "إنترنت" and "انترنت"
   are different terms to it even though they are the same word. Arabic writing
   varies freely in hamza placement, ta-marbuta, tatweel and diacritics, so
   without normalisation the sparse half of hybrid retrieval quietly loses most
   of its recall on Arabic. This is the single highest-value 40 lines in the
   retrieval path.

2. `detect_language` — decides the answer language and the TTS voice (brief
   D.9). Deliberately a heuristic, not a model: it runs on every turn, a
   classifier would cost more than the rest of the pipeline on CPU, and the
   distinction we actually need (Arabic vs English vs code-switched, and MSA vs
   Egyptian) is well served by script ratios plus a dialect marker list.
"""

from __future__ import annotations

import re
import unicodedata

from ..schemas import Language

# --- character classes -----------------------------------------------------
ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

# Arabic-Indic and extended Arabic-Indic digits -> ASCII. te.eg mixes both, and
# "٥٠ جنيه" must match a query typed as "50 جنيه".
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})

_LETTER_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",   # hamza forms on alef
    "ى": "ي", "ئ": "ي",                        # alef maqsura / hamza-on-ya
    "ة": "ه",                                  # ta-marbuta
    "ؤ": "و",
    "گ": "ك", "ک": "ك",                        # Persian kaf variants
    "ی": "ي",
    "ٓ": "",
})

# Egyptian-dialect markers. Presence of any of these is strong evidence the
# speaker is using ECA rather than MSA — which changes the TTS voice and tells
# the evaluation harness this is a dialect query.
EGYPTIAN_MARKERS: frozenset[str] = frozenset({
    "عايز", "عاوز", "عايزة", "ايه", "إيه", "ازاي", "إزاي", "ليه", "فين", "امتى", "إمتى",
    "دلوقتي", "دلوقت", "بتاع", "بتاعت", "بتاعي", "كده", "كدا", "مش", "خالص", "اوي", "أوي",
    "بص", "يعني", "علشان", "عشان", "لسه", "برضو", "برضه", "معلش", "ماشي", "طب", "طيب",
    "زي", "دي", "ده", "دول", "احنا", "إحنا", "انتو", "إنتو", "هيا", "هوا", "مفيش", "فيه",
    "عندي", "عندك", "ممكن", "بكام", "كام", "شوية", "حاجة", "حاجه", "بقى", "خلاص",
})

# Latin-script Egyptian ("Franco-Arab") — people type this constantly.
FRANCO_MARKERS: frozenset[str] = frozenset({
    "3ayez", "3awez", "eh", "ezay", "leh", "fen", "delwa2ti", "mesh", "keda", "kda",
    "3ashan", "bkam", "kam", "7aga", "ana", "enta", "fe", "msh",
})


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic(text: str, *, fold_letters: bool = True) -> str:
    """Canonicalise Arabic text for lexical matching.

    `fold_letters=False` keeps orthography intact — used when we want the text
    for *display* rather than for BM25.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text)
    text = text.replace(TATWEEL, "")
    text = text.translate(_DIGIT_MAP)
    if fold_letters:
        text = text.translate(_LETTER_MAP)
    # Collapse the whitespace zoo that HTML extraction leaves behind.
    text = re.sub(r"[​-‏‪-‮﻿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Light-stemming affixes. Arabic is templatic and heavily agglutinative: the
# definite article, conjunctions, prepositions and person markers all attach
# directly to the word. Without stripping them, BM25 treats "أشحن" (I recharge),
# "شحن" (recharge) and "الشحن" (the recharge) as three unrelated terms, and a
# question phrased naturally misses the page that answers it. This is the single
# cheapest large gain available to the sparse half of hybrid retrieval.
#
# Deliberately *light*: prefix/suffix stripping only, no root extraction. An
# aggressive stemmer (ISRI-style) over-conflates and starts matching unrelated
# words that share a root, which is worse than under-matching on a corpus this
# small.
# Multi-character definite-article forms only. Single-letter prefixes (و ف ب ك ل)
# are NOT stripped: far too many words legitimately begin with them, and removing
# one eats the first root letter. Concretely, stripping "ب" turned "باقات" into
# "قات" while "باقة" became "اقه" — so the plural and singular of the word this
# corpus is most often asked about stopped matching each other.
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_VERB_PREFIXES = ("ا", "ي", "ت", "ن")
_SUFFIXES = (
    "هما", "كما", "هن", "كن", "ها", "هم", "كم", "نا",
    "ية", "ين", "ون", "ات", "ان", "ه", "ك", "ي", "ا",
)

_MIN_STEM = 3


def light_stem(token: str) -> str:
    """Strip common Arabic affixes, never below `_MIN_STEM` characters."""
    if not token or not ARABIC_RANGE.search(token):
        return token

    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= _MIN_STEM:
            token = token[len(prefix) :]
            break

    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            token = token[: -len(suffix)]
            break

    # Verb person markers, applied after the nominal affixes above so that
    # "أشحن" -> "اشحن" -> "شحن" matches the noun form on the page.
    for prefix in _VERB_PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= _MIN_STEM:
            token = token[len(prefix) :]
            break

    return token


def tokenize(text: str, *, stem: bool = True) -> list[str]:
    """Tokens for BM25. Normalised and light-stemmed, so index and query agree.

    The index and the query must use identical settings — BM25 is rebuilt from
    chunk text at startup, so changing this takes effect without re-embedding.
    """
    normalized = normalize_arabic(text.lower())
    tokens = re.findall(r"[\w؀-ۿ]+", normalized)
    if not stem:
        return tokens
    return [light_stem(token) for token in tokens]


def script_ratios(text: str) -> tuple[float, float]:
    """(arabic_ratio, latin_ratio) over letter characters only."""
    arabic = len(ARABIC_RANGE.findall(text))
    latin = len(LATIN_RANGE.findall(text))
    total = arabic + latin
    if total == 0:
        return 0.0, 0.0
    return arabic / total, latin / total


def is_egyptian(text: str) -> bool:
    # Unstemmed: the marker list holds surface forms, and stemming "عايز" or
    # "دلوقتي" would stop them matching the very words that identify the dialect.
    tokens = set(tokenize(text, stem=False))
    if tokens & EGYPTIAN_MARKERS:
        return True
    return bool(tokens & FRANCO_MARKERS)


def detect_language(text: str) -> Language:
    """Classify into ar / arz / en / mixed.

    Order matters: we check code-switching before dialect, because "عايز أعرف
    الـ package بتاع 5G" is genuinely mixed and should be treated as such for
    retrieval even though it carries Egyptian markers.
    """
    if not text or not text.strip():
        return Language.UNKNOWN

    arabic_ratio, latin_ratio = script_ratios(text)
    if arabic_ratio == 0.0 and latin_ratio == 0.0:
        return Language.UNKNOWN

    # Both scripts materially present => code-switched.
    if arabic_ratio >= 0.15 and latin_ratio >= 0.15:
        return Language.MIXED

    if arabic_ratio > latin_ratio:
        return Language.ARZ if is_egyptian(text) else Language.AR

    # Latin-dominant, but Franco-Arab is Egyptian written in Latin letters.
    return Language.ARZ if is_egyptian(text) else Language.EN


def response_language(detected: Language) -> Language:
    """Which language to answer in.

    A.2 requires answering in the user's language. We collapse dialect and
    code-switching to Arabic for *generation* (the model should reply in
    natural Arabic, not attempt to mimic dialect orthography) while the
    original label is kept on the TurnRecord for the insights view.
    """
    if detected in (Language.ARZ, Language.MIXED, Language.AR):
        return Language.AR
    if detected is Language.EN:
        return Language.EN
    return Language.AR
