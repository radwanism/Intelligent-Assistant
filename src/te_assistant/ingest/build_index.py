"""Knowledge-base build: crawl -> clean -> chunk -> embed -> index.

Run as `te-ingest`. The stages are separable on purpose — cleaning and chunking
are the parts worth iterating on, and re-crawling te.eg every time you adjust a
threshold is both slow and rude to the site. `--stage` lets you redo the cheap
half against a cached crawl.

Artifacts:
    data/corpus/raw.jsonl    raw HTML, cache for re-running later stages
    data/corpus/te_eg.jsonl  cleaned documents — this is what gets committed
    data/chroma/             the vector index — also committed, so Colab
                             clones and runs without re-crawling or re-embedding
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ..config import Settings, get_settings
from ..schemas import Chunk, Language, SourceKind
from .chunk import chunks_from_document
from .clean import CleanDoc, clean_corpus, dedupe
from .crawl import crawl

log = logging.getLogger("te_assistant.ingest")


def _raw_path(settings: Settings) -> Path:
    return settings.corpus_path.parent / "raw.jsonl"


# --------------------------------------------------------------------------
def stage_crawl(
    settings: Settings,
    *,
    max_pages: int,
    delay: float,
    seeds: list[str] | None = None,
    append: bool = False,
) -> int:
    """Fetch pages into raw.jsonl.

    `append` lets a targeted follow-up crawl (a language section that the first
    seed set under-reached, say) extend the corpus without re-fetching — and
    without hammering the site for pages we already hold.
    """
    seeds = seeds or list(settings.crawl_seeds)
    path = _raw_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if append and path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = {json.loads(line)["url"] for line in handle}
        log.info("append mode: %d pages already held", len(existing))

    log.info("crawling te.eg (max_pages=%d, delay=%.2fs)", max_pages, delay)
    started = time.time()
    pages, stats = crawl(
        seeds,
        max_pages=max_pages,
        delay=delay,
        timeout=settings.crawl_timeout_seconds,
        seen=set(existing),
    )

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for page in pages:
            if page.url in existing:
                continue
            handle.write(
                json.dumps({"url": page.url, "html": page.html, "depth": page.depth}) + "\n"
            )
    log.info(
        "crawl done: %d fetched, %d failed, %d skipped in %.1fs -> %s",
        stats.fetched, stats.failed, stats.skipped, time.time() - started, path.name,
    )
    return len(pages)


def stage_clean(settings: Settings) -> list[CleanDoc]:
    path = _raw_path(settings)
    if not path.exists():
        raise SystemExit(f"no raw crawl at {path}; run with --stage crawl first")

    raw: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            raw.append((record["url"], record["html"]))

    log.info("cleaning %d pages", len(raw))
    docs = dedupe(clean_corpus(raw))

    out = settings.corpus_path
    with out.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(
                json.dumps(
                    {
                        "url": doc.url,
                        "title": doc.title,
                        "text": doc.text,
                        "language": doc.language.value,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    by_language: dict[str, int] = {}
    for doc in docs:
        by_language[doc.language.value] = by_language.get(doc.language.value, 0) + 1
    total_chars = sum(doc.char_count for doc in docs)
    log.info(
        "cleaned -> %d docs, %d chars (avg %d/doc), languages: %s",
        len(docs), total_chars, total_chars // max(len(docs), 1), by_language,
    )
    return docs


def load_clean_docs(settings: Settings) -> list[CleanDoc]:
    path = settings.corpus_path
    if not path.exists():
        raise SystemExit(f"no cleaned corpus at {path}; run with --stage clean first")
    docs: list[CleanDoc] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            docs.append(
                CleanDoc(
                    url=record["url"],
                    title=record["title"],
                    text=record["text"],
                    language=Language(record.get("language", "unknown")),
                    lines=[],
                )
            )
    return docs


def stage_index(settings: Settings, docs: list[CleanDoc], *, reset: bool) -> int:
    # Imported here so `--stage crawl` does not pay chromadb's import cost.
    from ..retrieval.store import VectorStore

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunks_from_document(
                text=doc.text,
                title=doc.title,
                url=doc.url,
                language=doc.language,
                source_kind=SourceKind.TE_EG,
                max_tokens=settings.chunk_tokens,
                overlap_tokens=settings.chunk_overlap,
                min_chars=settings.min_chunk_chars,
            )
        )

    if not chunks:
        raise SystemExit("no chunks produced — check the cleaning thresholds")

    log.info(
        "chunked %d docs -> %d chunks (avg %.1f per doc)",
        len(docs), len(chunks), len(chunks) / max(len(docs), 1),
    )

    store = VectorStore(settings)
    if reset:
        store.reset_kb()

    started = time.time()
    written = store.add_kb_chunks(chunks)
    elapsed = time.time() - started
    log.info(
        "indexed %d chunks in %.1fs (%.1f chunks/s); collection now holds %d",
        written, elapsed, written / max(elapsed, 0.01), store.kb_size(),
    )
    return written


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the te.eg knowledge base")
    parser.add_argument(
        "--stage",
        choices=("all", "crawl", "clean", "index"),
        default="all",
        help="run one stage only; later stages reuse the cached artifacts",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument(
        "--append", action="store_true", help="extend raw.jsonl instead of replacing it"
    )
    parser.add_argument(
        "--seeds", nargs="*", default=None, help="override the configured seed URLs"
    )
    parser.add_argument(
        "--reset", action="store_true", help="drop the existing KB collection first"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = get_settings()
    max_pages = args.max_pages or settings.crawl_max_pages
    delay = args.delay if args.delay is not None else settings.crawl_delay_seconds

    if args.stage in ("all", "crawl"):
        stage_crawl(
            settings,
            max_pages=max_pages,
            delay=delay,
            seeds=args.seeds,
            append=args.append,
        )

    docs: list[CleanDoc] | None = None
    if args.stage in ("all", "clean"):
        docs = stage_clean(settings)

    if args.stage in ("all", "index"):
        stage_index(settings, docs or load_clean_docs(settings), reset=args.reset)

    return 0


if __name__ == "__main__":
    sys.exit(main())
