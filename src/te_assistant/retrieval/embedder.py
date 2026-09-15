"""Embedding backends, one interface per profile.

Both backends implement the same two methods, because retrieval quality depends
on using the *asymmetric* form of these models correctly:

    embed_query(text)     -> vector for a question
    embed_passages(texts) -> vectors for documents

E5 and BGE-M3 are both trained with distinct query/passage roles. E5 in
particular requires literal "query: " / "passage: " prefixes, and omitting them
is a silent quality regression — retrieval still returns results, they are just
measurably worse. Putting the prefixes behind this interface means no caller can
get it wrong.

Licensing (brief B.3): both defaults are commercially usable — multilingual-e5-small
is MIT and BAAI/bge-m3 is MIT. jina-embeddings-v3 is deliberately absent: it is
CC-BY-NC-4.0 and therefore disqualified from the core path. It may be measured
in scripts/bench_models.py under A.6.3.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from ..config import CPU_SLOTS, Profile, Settings, get_settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> Sequence[float]: ...
    def embed_passages(self, texts: list[str]) -> list[Sequence[float]]: ...


class OnnxEmbedder:
    """cpu-lite backend: fastembed / ONNX Runtime, no torch.

    The E5 family needs literal "query: " / "passage: " prefixes and degrades
    silently without them; the MiniLM/paraphrase family is symmetric and must
    NOT be prefixed, or the prefix becomes noise in the embedding. Getting this
    backwards costs retrieval quality without raising an error, so the prefix
    decision is derived from the model name once, here, rather than left to
    callers.
    """

    def __init__(self, model_name: str, dim: int, threads: int | None = None) -> None:
        import os

        from fastembed import TextEmbedding

        self.dim = dim
        self.model_name = model_name
        self._asymmetric = "e5" in model_name.lower()
        # ONNX Runtime defaults to a single intra-op thread here, which left
        # indexing at ~0.8 chunks/s on a 4-core machine. Pinning to the physical
        # core count is the difference between a 20-minute and a 5-minute index
        # build, and it also cuts per-query embedding latency.
        self._threads = threads or max(2, (os.cpu_count() or 4) // 2)
        self._model = TextEmbedding(model_name=model_name, threads=self._threads)
        log.info(
            "loaded ONNX embedder %s (dim=%d, prefixes=%s, threads=%d)",
            model_name, dim, self._asymmetric, self._threads,
        )

    def embed_query(self, text: str) -> Sequence[float]:
        payload = f"query: {text}" if self._asymmetric else text
        return next(iter(self._model.embed([payload])))

    def embed_passages(self, texts: list[str]) -> list[Sequence[float]]:
        payload = [f"passage: {t}" for t in texts] if self._asymmetric else texts
        return list(self._model.embed(payload))


class BGEM3Embedder:
    """gpu-colab: BAAI/bge-m3 (MIT). No prefixes — BGE-M3 does not use them."""

    def __init__(self, model_name: str, dim: int) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.dim = dim
        self._model = BGEM3FlagModel(model_name, use_fp16=True)  # T4: fp16, never bf16
        log.info("loaded BGE-M3 embedder %s (dim=%d)", model_name, dim)

    def embed_query(self, text: str) -> Sequence[float]:
        return self._model.encode([text], max_length=512)["dense_vecs"][0]

    def embed_passages(self, texts: list[str]) -> list[Sequence[float]]:
        return list(self._model.encode(texts, max_length=1024)["dense_vecs"])


_CACHE: dict[str, Embedder] = {}


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    key = f"{settings.profile.value}:{settings.slots.embedder}"
    if key in _CACHE:
        return _CACHE[key]

    slots = settings.slots
    if settings.profile is Profile.GPU_COLAB:
        try:
            embedder: Embedder = BGEM3Embedder(slots.embedder, slots.embed_dim)
        except Exception as exc:
            # A missing GPU extra must degrade, not crash: the CPU embedder is
            # always installed, so the system stays usable.
            log.warning("BGE-M3 unavailable (%s); falling back to the ONNX embedder", exc)
            embedder = OnnxEmbedder(CPU_SLOTS.embedder, CPU_SLOTS.embed_dim)
    else:
        embedder = OnnxEmbedder(slots.embedder, slots.embed_dim)

    _CACHE[key] = embedder
    return embedder
