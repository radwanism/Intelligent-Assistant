"""Cross-encoder reranking — GPU profile only.

Bi-encoder retrieval scores a query and a passage independently, so it can only
compare them through the geometry of the embedding space. A cross-encoder reads
both together and is markedly better at the discrimination this corpus needs:
telling "Nitro 200" from "Nitro 100", or an Egyptian-dialect question from the
MSA page that actually answers it.

It is off on `cpu-lite` and that is a deliberate, documented cut. Reranking 40
candidates with bge-reranker-v2-m3 on four CPU cores costs several seconds —
comparable to generating the entire answer — which would blow the latency budget
in B.10 to buy a quality improvement the RRF fusion already partly captures.

`get_reranker` returns None on CPU, and the pipeline falls back to RRF ordering.
That is the documented degradation path, not a silent failure.
"""

from __future__ import annotations

import logging
from typing import Protocol

from ..config import Profile, Settings, get_settings
from ..schemas import ScoredChunk

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]: ...


class BGEReranker:
    """BAAI/bge-reranker-v2-m3 (Apache-2.0), multilingual including Arabic.

    Chosen over jina-reranker-v2 on licensing: that model is CC-BY-NC-4.0 and is
    therefore disqualified from the serving path of a commercial deployment.
    """

    def __init__(self, model_name: str) -> None:
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(model_name, use_fp16=True)  # T4: fp16 only
        log.info("reranker loaded: %s", model_name)

    def rerank(
        self, query: str, chunks: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [[query, chunk.chunk.text] for chunk in chunks]
        try:
            scores = self._model.compute_score(pairs, normalize=True)
        except Exception:
            # Falling back to the fusion order is strictly better than failing
            # the turn: the candidates are already relevant, just less well sorted.
            log.exception("reranking failed; keeping RRF order")
            return chunks[:top_k]

        if not isinstance(scores, list):
            scores = [scores]
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.rerank_score = float(score)

        ranked = sorted(
            chunks, key=lambda c: c.rerank_score if c.rerank_score is not None else 0.0,
            reverse=True,
        )
        return ranked[:top_k]


_CACHE: dict[str, Reranker | None] = {}


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    """Return a reranker, or None when the active profile does not use one."""
    settings = settings or get_settings()
    model_name = settings.slots.reranker
    if model_name is None or settings.profile is not Profile.GPU_COLAB:
        return None

    if model_name in _CACHE:
        return _CACHE[model_name]

    try:
        reranker: Reranker | None = BGEReranker(model_name)
    except Exception as exc:
        log.warning("reranker unavailable (%s); using RRF ordering", exc)
        reranker = None

    _CACHE[model_name] = reranker
    return reranker
