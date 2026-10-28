"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

Why both halves are needed for this corpus specifically:

  * Dense alone misses exact identifiers. Telecom content is full of them —
    "WE Bonus", "Nitro 200", "5G", plan prices, short codes. Embeddings smear
    these together; a user asking about "Nitro 200" gets "Nitro 100".
  * BM25 alone fails the central requirement of this case study: an Egyptian
    dialect question must retrieve an MSA or English page (brief D.9). There is
    no lexical overlap between "عايز أعرف أسعار النت" and an English tariff
    page, so sparse retrieval returns nothing useful.

So the sparse side carries exact matches and the dense side carries
cross-lingual meaning. RRF is used to combine them rather than a score-weighted
sum because the two scores are not on comparable scales and normalising them
per-query is fragile — RRF only needs the ranks.

The session documents (B.8) are retrieved through the same fusion but always
via the session-scoped API, and are given a small boost: when a user has just
uploaded a document, a question is far more likely to be about it than about
the general website.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ..config import Settings, get_settings
from ..schemas import ScoredChunk, SourceKind
from .normalize_ar import tokenize
from .store import VectorStore

log = logging.getLogger(__name__)

RRF_K = 60           # standard RRF damping constant
SESSION_BOOST = 1.15  # modest: never let an upload drown out te.eg entirely
SESSION_TOKEN_CACHE = 16  # sessions whose tokenised uploads are kept warm


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    used_session_docs: bool
    dense_hits: int
    sparse_hits: int

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class BM25Index:
    """In-memory sparse index over the shared KB.

    Rebuilt at startup from the vector store's documents. At ~3-4k chunks this
    costs well under a second and a few tens of MB, so persisting it would add
    a staleness bug for no real saving.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if not texts:
            self._bm25 = None
            self._chunk_ids = []
            return
        corpus = [tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(corpus)
        self._chunk_ids = chunk_ids
        log.info("BM25 index built over %d chunks", len(texts))

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._chunk_ids, scores, strict=True), key=lambda p: p[1], reverse=True
        )
        return [(cid, float(score)) for cid, score in ranked[:k] if score > 0.0]


def bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """BM25 over a small, throwaway corpus — one session's uploads.

    Not `rank_bm25`, because its classic IDF, log((N - n + 0.5) / (n + 0.5)), is
    *negative* for any term that appears in more than half the documents. On a
    shared knowledge base of hundreds of pages that never happens; on one
    uploaded bill of two chunks it is the normal case, and every matching term
    would score below zero and be discarded. This uses the Lucene form,
    log(1 + (N - n + 0.5) / (n + 0.5)), which is always positive.
    """
    count = len(documents)
    if not count or not query_tokens:
        return [0.0] * count

    lengths = [len(doc) for doc in documents]
    average = (sum(lengths) / count) or 1.0
    frequencies = [{} for _ in documents]
    document_frequency: dict[str, int] = {}
    for index, doc in enumerate(documents):
        for token in doc:
            frequencies[index][token] = frequencies[index].get(token, 0) + 1
        for token in frequencies[index]:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    scores: list[float] = []
    for index in range(count):
        total = 0.0
        for token in set(query_tokens):
            tf = frequencies[index].get(token, 0)
            if not tf:
                continue
            n = document_frequency[token]
            idf = math.log(1.0 + (count - n + 0.5) / (n + 0.5))
            norm = tf + k1 * (1.0 - b + b * lengths[index] / average)
            total += idf * tf * (k1 + 1.0) / norm
        scores.append(total)
    return scores


def reciprocal_rank_fusion(
    rankings: list[list[str]], weights: list[float] | None = None, k: int = RRF_K
) -> dict[str, float]:
    """Fuse ranked id lists. Score for one list is weight / (k + rank)."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or VectorStore(self.settings)
        self.bm25 = bm25 or BM25Index()
        # session_id -> (chunk count when tokenised, chunks, tokens). Tokenising a
        # large upload on every question would be wasted work; the count is the
        # cheap staleness check, and the cap keeps ended sessions from piling up.
        self._session_tokens: OrderedDict[
            str, tuple[int, list[ScoredChunk], list[list[str]]]
        ] = OrderedDict()
        self._session_lock = threading.Lock()

    def _session_sparse(self, query: str, session_id: str, k: int) -> list[ScoredChunk]:
        """Lexical ranking over *this session's* uploads only.

        Closes the gap where exact identifiers in the user's own document — a
        reference number, a plan name — were matched by embedding similarity
        alone. Scoped by `store.session_chunks`, which cannot be called without
        a session id.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        chunks = self.store.session_chunks(session_id)
        if not chunks:
            return []

        with self._session_lock:
            cached = self._session_tokens.get(session_id)
            if cached is None or cached[0] != len(chunks):
                documents = [tokenize(sc.chunk.text) for sc in chunks]
                self._session_tokens[session_id] = (len(chunks), chunks, documents)
                self._session_tokens.move_to_end(session_id)
                while len(self._session_tokens) > SESSION_TOKEN_CACHE:
                    self._session_tokens.popitem(last=False)
            else:
                _, chunks, documents = cached

        scores = bm25_scores(tokens, documents)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [sc for sc, score in ranked[:k] if score > 0.0]

    def forget_session(self, session_id: str) -> None:
        """Drop a session's cached tokens (called when the session ends)."""
        with self._session_lock:
            self._session_tokens.pop(session_id, None)

    def warmup(self) -> None:
        """Build the sparse index from whatever is in the shared KB."""
        collection = self.store._kb  # noqa: SLF001 - intentional, same package
        if collection.count() == 0:
            log.warning("knowledge base is empty — run `te-ingest` first")
            return
        data = collection.get(include=["documents"])
        self.bm25.build(data["ids"], [d or "" for d in data["documents"]])

    def retrieve(
        self, query: str, *, session_id: str | None = None, k: int | None = None
    ) -> RetrievalResult:
        k = k or self.settings.retrieve_k

        dense = self.store.search_kb(query, k=k)
        sparse = self.bm25.search(query, k=k) if self.bm25.ready else []

        by_id: dict[str, ScoredChunk] = {sc.chunk.chunk_id: sc for sc in dense}

        session_hits: list[ScoredChunk] = []
        session_sparse: list[ScoredChunk] = []
        if session_id:
            # Always the scoped API — there is no unscoped path (B.8).
            session_hits = self.store.search_session_docs(
                query, session_id, k=max(8, k // 3)
            )
            for hit in session_hits:
                by_id[hit.chunk.chunk_id] = hit
            session_sparse = self._session_sparse(query, session_id, k=max(8, k // 3))
            for hit in session_sparse:
                # A lexical hit that dense search did not return still needs a
                # place in the pool; one dense already scored keeps its cosine.
                by_id.setdefault(hit.chunk.chunk_id, hit)

        # Sparse hits reference chunks we may not have fetched densely.
        missing = [cid for cid, _ in sparse if cid not in by_id]
        if missing:
            for sc in self._fetch_by_ids(missing):
                by_id[sc.chunk.chunk_id] = sc

        rankings = [[sc.chunk.chunk_id for sc in dense]]
        weights = [1.0]
        if sparse:
            rankings.append([cid for cid, _ in sparse])
            weights.append(1.0)
        if session_hits:
            rankings.append([sc.chunk.chunk_id for sc in session_hits])
            weights.append(SESSION_BOOST)
        if session_sparse:
            rankings.append([sc.chunk.chunk_id for sc in session_sparse])
            weights.append(SESSION_BOOST)

        fused = reciprocal_rank_fusion(rankings, weights)

        for cid, rank in ((cid, r) for r, (cid, _) in enumerate(sparse)):
            if cid in by_id:
                by_id[cid].bm25_rank = rank

        ordered = sorted(
            (by_id[cid] for cid in fused if cid in by_id),
            key=lambda sc: fused[sc.chunk.chunk_id],
            reverse=True,
        )
        for sc in ordered:
            sc.score = fused[sc.chunk.chunk_id]

        return RetrievalResult(
            chunks=ordered[:k],
            used_session_docs=bool(session_hits or session_sparse),
            dense_hits=len(dense),
            sparse_hits=len(sparse),
        )

    def _fetch_by_ids(self, ids: list[str]) -> list[ScoredChunk]:
        from .store import _to_scored  # noqa: PLC0415 - same package helper

        raw = self.store._kb.get(  # noqa: SLF001
            ids=ids, include=["documents", "metadatas"]
        )
        shaped = {
            "ids": [raw["ids"]],
            "documents": [raw["documents"]],
            "metadatas": [raw["metadatas"]],
            # No distance for a direct get: the chunk was found lexically and has
            # no cosine to the query, so it must not be given one. RRF decides.
            "distances": [[None] * len(raw["ids"])],
        }
        return _to_scored(shaped)


def diversify(chunks: list[ScoredChunk], *, max_per_url: int = 2) -> list[ScoredChunk]:
    """Cap how many chunks one page may contribute.

    Without this a single long FAQ page fills the whole context window and the
    answer cites one source for everything, which both looks wrong and loses
    genuinely relevant material from other pages.
    """
    seen: dict[str, int] = {}
    out: list[ScoredChunk] = []
    for sc in chunks:
        key = sc.chunk.url or sc.chunk.doc_name or sc.chunk.chunk_id
        count = seen.get(key, 0)
        # Uploaded documents are exempt: if the user asks about their file, a
        # cap that starves it of context defeats the purpose of the upload.
        if count >= max_per_url and sc.chunk.source_kind is not SourceKind.USER_DOCUMENT:
            continue
        seen[key] = count + 1
        out.append(sc)
    return out


LEXICAL_ONLY_CONFIDENCE = 0.5


def confidence_of(chunks: list[ScoredChunk]) -> float:
    """How relevant is what was retrieved? 0..1, from real evidence of relevance.

    Feeds two things: the groundedness decision (refuse rather than guess when
    nothing relevant was found) and the frustration detector, which treats
    repeated low-confidence retrieval as evidence the user is stuck (B.6).

    This used to squash the *fused RRF score*. That score is 1/(60 + rank) summed
    over lists, so it depends only on position: the top result scores about
    1/61 whether it is a perfect match or complete nonsense, because dense search
    always returns its k nearest neighbours however far away they are. The
    resulting confidence never fell below ~0.45, so a refusal threshold of 0.18
    could not be reached by any non-empty result. Confidence now comes from
    signals that actually vary with relevance, best first:

      1. the cross-encoder score, when the profile has a reranker (already 0..1);
      2. otherwise the best raw cosine similarity among the chunks in play;
      3. otherwise — every chunk was found only lexically — a fixed moderate
         value, since an exact-term match is evidence but not a calibrated one.
    """
    if not chunks:
        return 0.0
    reranked = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if reranked:
        return max(0.0, min(1.0, max(reranked)))
    dense = [c.dense_score for c in chunks if c.dense_score is not None]
    if dense:
        return max(0.0, min(1.0, max(dense)))
    return LEXICAL_ONLY_CONFIDENCE
