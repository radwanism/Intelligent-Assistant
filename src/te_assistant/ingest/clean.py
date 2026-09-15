"""Boilerplate removal and document normalisation.

This module is the "Data quality" pillar (brief B.5) in concrete form.

Why it matters more than usual here: te.eg is a Liferay portal that renders the
same ~2-3k character navigation menu into every single page. Measured on the
live site, a typical content page has 5-11k visible characters of which roughly
a third is that menu. If it is left in:

  * every chunk shares most of its tokens with every other chunk, so cosine
    similarity between unrelated pages goes up and dense retrieval stops
    discriminating;
  * BM25 term statistics are dominated by menu vocabulary ("موبايل", "إنترنت")
    which are exactly the words users search for, so the sparse side degrades too.

Two passes, because neither alone is sufficient:
  1. `extract_main` — per-page main-content extraction via trafilatura.
  2. `drop_repeated_lines` — corpus-level: any line appearing on more than
     `repeat_threshold` of pages is template furniture by definition. This is
     the pass that actually catches the portal chrome, because it needs no
     knowledge of the site's markup.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

from selectolax.parser import HTMLParser

from ..retrieval.normalize_ar import detect_language, normalize_arabic
from ..schemas import Language

log = logging.getLogger(__name__)

# Nodes that are never content, removed before extraction.
STRIP_TAGS = ("script", "style", "noscript", "svg", "iframe", "form", "button")
STRIP_SELECTORS = ("nav", "header", "footer", "[role=navigation]", ".breadcrumb", ".cookie")

# Phrases that mark a line as chrome even if it survives the other passes.
CHROME_MARKERS = (
    "تخطي إلى المحتوى الرئيسي",   # "skip to main content"
    "skip to main content",
    "main menu",
    "جميع الحقوق محفوظة",          # "all rights reserved"
    "all rights reserved",
    "©",
)

MIN_LINE_CHARS = 3


@dataclass
class CleanDoc:
    url: str
    title: str
    text: str
    language: Language
    lines: list[str]

    @property
    def char_count(self) -> int:
        return len(self.text)


def _title_of(html: str) -> str:
    tree = HTMLParser(html)
    node = tree.css_first("title")
    title = (node.text() if node else "") or ""
    # te.eg titles are all suffixed " - Telecom Egypt"; keep it off the chunks.
    return re.sub(r"\s*[-|]\s*Telecom Egypt\s*$", "", title).strip()


def _strip_nodes(html: str) -> str:
    tree = HTMLParser(html)
    for selector in (*STRIP_TAGS, *STRIP_SELECTORS):
        for node in tree.css(selector):
            node.decompose()
    return tree.html or ""


def extract_main(html: str) -> str:
    """Per-page main-content extraction.

    trafilatura is the primary; it is good at templated CMS pages. We fall back
    to a plain text dump of the stripped tree when it returns nothing, which
    happens on short promo pages with unusual markup — better a noisy page than
    a missing one, since pass 2 will clean it anyway.
    """
    stripped = _strip_nodes(html)
    try:
        import trafilatura  # noqa: PLC0415 - optional at import time for tests

        extracted = trafilatura.extract(
            stripped,
            include_comments=False,
            include_tables=True,      # te.eg puts plan pricing in tables
            favor_recall=True,
            no_fallback=False,
        )
        if extracted and extracted.strip():
            return extracted
    except Exception as exc:
        log.debug("trafilatura failed, falling back to text dump: %s", exc)

    tree = HTMLParser(stripped)
    return tree.body.text(separator="\n") if tree.body else ""


def _split_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) >= MIN_LINE_CHARS:
            out.append(line)
    return out


def _is_chrome(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in CHROME_MARKERS)


def build_line_frequency(docs_lines: list[list[str]]) -> Counter[str]:
    """Count how many DISTINCT pages each line appears on.

    Keyed on the normalised form so Arabic orthography variants of the same menu
    item collapse together.
    """
    counter: Counter[str] = Counter()
    for lines in docs_lines:
        for line in {normalize_arabic(line) for line in lines}:
            counter[line] += 1
    return counter


def drop_repeated_lines(
    lines: list[str],
    frequency: Counter[str],
    total_docs: int,
    *,
    repeat_threshold: float = 0.35,
) -> list[str]:
    """Remove lines that appear on more than `repeat_threshold` of all pages.

    0.35 is deliberately not aggressive. A genuine content line shared by a
    third of the site is vanishingly rare, while the portal menu appears on
    ~100% of pages, so the two populations are far apart and the exact cut point
    barely matters. Erring low would start eating real content on sites where
    several pages legitimately repeat a disclaimer.
    """
    if total_docs <= 1:
        return [line for line in lines if not _is_chrome(line)]

    cutoff = max(2, int(total_docs * repeat_threshold))
    kept: list[str] = []
    for line in lines:
        if _is_chrome(line):
            continue
        if frequency[normalize_arabic(line)] >= cutoff:
            continue
        kept.append(line)
    return kept


def clean_corpus(
    raw: list[tuple[str, str]],
    *,
    repeat_threshold: float = 0.35,
    min_chars: int = 200,
    max_chars: int = 40_000,
) -> list[CleanDoc]:
    """Clean a whole crawl at once.

    Takes (url, html) pairs. Corpus-level, because pass 2 needs to see every
    page before it can tell content from furniture.
    """
    titles = [_title_of(html) for _, html in raw]
    per_doc_lines = [_split_lines(extract_main(html)) for _, html in raw]
    frequency = build_line_frequency(per_doc_lines)
    total = len(per_doc_lines)

    docs: list[CleanDoc] = []
    dropped_short = 0
    truncated = 0
    for (url, _), title, lines in zip(raw, titles, per_doc_lines, strict=True):
        kept = drop_repeated_lines(
            lines, frequency, total, repeat_threshold=repeat_threshold
        )
        text = "\n".join(kept).strip()
        if len(text) < min_chars:
            # Nothing survived cleaning: a pure navigation or redirect page.
            dropped_short += 1
            continue
        if len(text) > max_chars:
            # One page must not dominate the index. Listing pages (press
            # releases, archives) run to six figures of characters and would
            # otherwise contribute hundreds of chunks that no customer question
            # ever wants.
            truncated += 1
            text = text[:max_chars]
        docs.append(
            CleanDoc(
                url=url,
                title=title,
                text=text,
                language=detect_language(text),
                lines=kept,
            )
        )

    log.info(
        "cleaned %d pages -> %d docs (%d dropped as boilerplate-only, %d truncated)",
        total, len(docs), dropped_short, truncated,
    )
    return docs


def dedupe(docs: list[CleanDoc], *, shingle_size: int = 12) -> list[CleanDoc]:
    """Drop near-duplicate pages.

    te.eg serves the same promo under several URLs, and AR/EN mirrors of a page
    are separate documents we WANT to keep (cross-lingual retrieval depends on
    them), so dedupe is keyed on exact-ish text, not on semantics.
    """
    seen: set[frozenset[str]] = set()
    out: list[CleanDoc] = []
    for doc in docs:
        tokens = normalize_arabic(doc.text).split()
        if len(tokens) < shingle_size:
            fingerprint = frozenset({" ".join(tokens)})
        else:
            fingerprint = frozenset(
                " ".join(tokens[i : i + shingle_size])
                for i in range(0, len(tokens) - shingle_size, shingle_size * 4)
            )
        if fingerprint and fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(doc)
    log.info("dedupe: %d -> %d docs", len(docs), len(out))
    return out
