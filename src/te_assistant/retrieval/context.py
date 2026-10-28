"""Which chunks reach the prompt.

One function owns this decision so that the serving path and the evaluation
harness cannot drift apart. They used to each carry their own copy of the
ordering — the pipeline applied the cross-encoder, the harness stopped after
diversification — so any figure the harness produced for the GPU profile
described an ordering no user ever received.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..schemas import ScoredChunk
from .hybrid import diversify
from .rerank import get_reranker


@dataclass
class ContextSelection:
    chunks: list[ScoredChunk]
    reranked: bool


def select_context(
    query: str,
    chunks: list[ScoredChunk],
    settings: Settings,
    *,
    use_reranker: bool = True,
) -> ContextSelection:
    """Diversify, rerank when the profile has a reranker, and cut to `rerank_k`.

    `use_reranker=False` exists so the harness can report the fusion-only
    ordering next to the reranked one, which is what shows whether the
    reranker earns its latency.
    """
    candidates = diversify(chunks)
    reranker = get_reranker(settings) if use_reranker else None
    if reranker is not None and candidates:
        ranked = reranker.rerank(query, candidates, top_k=settings.rerank_k)
        return ContextSelection(chunks=ranked, reranked=True)
    return ContextSelection(chunks=candidates[: settings.rerank_k], reranked=False)
