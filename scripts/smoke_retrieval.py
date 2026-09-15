"""Retrieval smoke test.

Checks the property the whole design turns on (brief D.9): an Egyptian-dialect
question must retrieve the right te.eg page even when that page is in MSA or
English. Run it after any change to chunking, cleaning or the embedder.
"""

from __future__ import annotations

import sys
import time

from te_assistant.config import get_settings
from te_assistant.retrieval.hybrid import HybridRetriever, confidence_of, diversify
from te_assistant.retrieval.normalize_ar import detect_language

QUERIES: list[tuple[str, str]] = [
    ("عايز أعرف أسعار باقات الإنترنت", "Egyptian dialect — internet pricing"),
    ("ازاي أشحن الخط بتاعي؟", "Egyptian dialect — how to recharge"),
    ("ما هي خدمة الجيل الخامس؟", "MSA — 5G service"),
    ("What are the 5G home internet options?", "English — 5G home"),
    ("رقم خدمة العملاء كام؟", "Egyptian dialect — customer service number"),
    ("ايه هو الـ eKYC؟", "Code-switched — eKYC"),
    ("How do I check my remaining balance?", "English — balance code"),
]


def main() -> int:
    settings = get_settings()
    retriever = HybridRetriever(settings=settings)

    print(f"profile     : {settings.profile.value}")
    print(f"embedder    : {settings.slots.embedder}")
    print("building BM25 index...")
    started = time.perf_counter()
    retriever.warmup()
    print(f"kb chunks   : {retriever.store.kb_size()}")
    print(f"bm25 ready  : {retriever.bm25.ready} ({time.perf_counter() - started:.1f}s)\n")

    failures = 0
    for query, label in QUERIES:
        started = time.perf_counter()
        result = retriever.retrieve(query)
        elapsed = (time.perf_counter() - started) * 1000
        top = diversify(result.chunks)[:3]

        print("=" * 78)
        print(f"{label}\n  query: {query}")
        print(
            f"  lang={detect_language(query).value}  "
            f"dense={result.dense_hits} sparse={result.sparse_hits}  "
            f"conf={confidence_of(result.chunks):.3f}  {elapsed:.0f}ms"
        )
        if not top:
            print("  !! NO RESULTS")
            failures += 1
            continue
        for rank, scored in enumerate(top, start=1):
            chunk = scored.chunk
            preview = " ".join(chunk.text.split())[:110]
            print(f"  {rank}. [{chunk.language.value}] {chunk.title[:58]}")
            print(f"     {chunk.url}")
            print(f"     {preview}…")

    print("=" * 78)
    print(f"\n{len(QUERIES) - failures}/{len(QUERIES)} queries returned results")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
