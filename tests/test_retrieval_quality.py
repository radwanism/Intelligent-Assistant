"""Retrieval confidence, the refusal gate, and sparse matching over uploads."""

from __future__ import annotations

import hashlib

import pytest
from rank_bm25 import BM25Okapi

from te_assistant.config import Settings, get_settings
from te_assistant.llm.generate import MIN_RETRIEVAL_CONFIDENCE, generate_answer
from te_assistant.retrieval.hybrid import (
    LEXICAL_ONLY_CONFIDENCE,
    HybridRetriever,
    bm25_scores,
    confidence_of,
)
from te_assistant.retrieval.store import SessionScopeError, VectorStore
from te_assistant.schemas import Chunk, Language, ScoredChunk, SourceKind
from te_assistant.session.manager import SessionManager


def _scored(**kwargs) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=kwargs.pop("cid", "c"), text="some passage text", title="t",
                    url="https://te.eg/x", language=Language.EN),
        score=kwargs.pop("score", 0.0),
        **kwargs,
    )


class ExplodingLLM:
    """Fails the test if the model is called at all."""

    def generate(self, *a, **k):
        raise AssertionError("the model must not be called when retrieval found nothing relevant")


class EchoLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *a, **k) -> str:
        self.calls += 1
        return "An answer [1]."


# --------------------------------------------------------------------------
# confidence_of — a number that varies with relevance
# --------------------------------------------------------------------------
def test_no_chunks_means_zero_confidence() -> None:
    assert confidence_of([]) == 0.0


def test_confidence_follows_the_cosine_similarity() -> None:
    weak = confidence_of([_scored(dense_score=0.04)])
    strong = confidence_of([_scored(dense_score=0.62)])
    assert weak < MIN_RETRIEVAL_CONFIDENCE < strong
    assert strong == pytest.approx(0.62)


def test_a_huge_fused_rank_score_cannot_vouch_for_an_irrelevant_chunk() -> None:
    """The original defect. The fused RRF score is positional: the top result
    scores ~1/61 whether it matches or not, so squashing it gave >0.44 for
    nonsense and made the refusal threshold unreachable."""
    nonsense = _scored(score=0.0328, dense_score=0.01, dense_rank=0)
    assert confidence_of([nonsense]) < MIN_RETRIEVAL_CONFIDENCE


def test_the_cross_encoder_score_takes_precedence_when_present() -> None:
    chunk = _scored(dense_score=0.9, rerank_score=0.03)
    assert confidence_of([chunk]) == pytest.approx(0.03)


def test_best_signal_across_the_chunks_in_play_is_used() -> None:
    chunks = [_scored(cid="a", dense_score=0.05), _scored(cid="b", dense_score=0.48)]
    assert confidence_of(chunks) == pytest.approx(0.48)


def test_chunks_found_only_lexically_get_a_moderate_fixed_confidence() -> None:
    assert confidence_of([_scored()]) == LEXICAL_ONLY_CONFIDENCE
    assert LEXICAL_ONLY_CONFIDENCE > MIN_RETRIEVAL_CONFIDENCE, "exact matches must not be refused"


def test_negative_cosine_is_clamped() -> None:
    assert confidence_of([_scored(dense_score=-0.3)]) == 0.0


# --------------------------------------------------------------------------
# generate_answer — the gate actually closes
# --------------------------------------------------------------------------
def test_irrelevant_retrieval_refuses_without_calling_the_model() -> None:
    answer = generate_answer(
        ExplodingLLM(), question="what is the weather in Paris",
        chunks=[_scored(dense_score=0.03)], language=Language.EN,
    )
    assert not answer.grounded
    assert answer.citations == []
    assert "customer service" in answer.text.lower()


def test_relevant_retrieval_still_reaches_the_model() -> None:
    llm = EchoLLM()
    answer = generate_answer(
        llm, question="5G plans", chunks=[_scored(dense_score=0.55)], language=Language.EN
    )
    assert llm.calls == 1
    assert answer.grounded


def test_the_threshold_is_overridable_per_call() -> None:
    llm = EchoLLM()
    chunks = [_scored(dense_score=0.30)]
    generate_answer(llm, question="q", chunks=chunks, language=Language.EN, min_confidence=0.5)
    assert llm.calls == 0, "a stricter deployment threshold must be able to refuse this"
    generate_answer(llm, question="q", chunks=chunks, language=Language.EN, min_confidence=0.1)
    assert llm.calls == 1


def test_threshold_is_configurable_from_the_environment(monkeypatch) -> None:
    base = get_settings()
    monkeypatch.setenv("TE_MIN_RETRIEVAL_CONFIDENCE", "0.33")
    assert Settings(profile=base.profile, slots=base.slots).min_retrieval_confidence == 0.33


@pytest.mark.parametrize("bad", ["", "abc", "  "])
def test_a_bad_threshold_value_falls_back_instead_of_crashing(monkeypatch, bad) -> None:
    base = get_settings()
    monkeypatch.setenv("TE_MIN_RETRIEVAL_CONFIDENCE", bad)
    settings = Settings(profile=base.profile, slots=base.slots)
    assert settings.min_retrieval_confidence == MIN_RETRIEVAL_CONFIDENCE


# --------------------------------------------------------------------------
# bm25_scores — usable on a corpus of one
# --------------------------------------------------------------------------
def test_the_classic_idf_fails_on_a_tiny_corpus_which_is_why_it_is_not_used() -> None:
    """Documents the premise: rank_bm25 cannot score a one- or two-chunk upload."""
    assert BM25Okapi([["monthly", "zx9000"]]).get_scores(["zx9000"])[0] <= 0.0
    assert max(BM25Okapi([["a", "b"], ["zx9000", "c"]]).get_scores(["zx9000"])) <= 0.0


def test_a_matching_term_scores_positively_even_in_a_single_chunk() -> None:
    assert bm25_scores(["zx9000"], [["monthly", "charge", "zx9000"]])[0] > 0.0


def test_the_matching_chunk_outranks_the_one_without_the_term() -> None:
    scores = bm25_scores(["zx9000"], [["monthly", "charge"], ["reference", "zx9000"]])
    assert scores[1] > scores[0] == 0.0


def test_rarer_terms_weigh_more() -> None:
    docs = [["common", "rare"], ["common", "x"], ["common", "y"]]
    assert bm25_scores(["rare"], docs)[0] > bm25_scores(["common"], docs)[0]


@pytest.mark.parametrize(("query", "docs"), [([], [["a"]]), (["a"], []), ([], [])])
def test_degenerate_inputs_do_not_raise(query, docs) -> None:
    assert bm25_scores(query, docs) == [0.0] * len(docs)


# --------------------------------------------------------------------------
# Session sparse retrieval — scoped, cached, evicted
# --------------------------------------------------------------------------
class StubEmbedder:
    dim = 16

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dim)]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


@pytest.fixture()
def retriever(tmp_path) -> HybridRetriever:
    base = get_settings()
    settings = Settings(
        profile=base.profile, slots=base.slots, data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma", corpus_path=tmp_path / "c.jsonl",
        sqlite_path=tmp_path / "t.db", session_files_dir=tmp_path / "sessions",
        traces_path=tmp_path / "traces.jsonl",
    )
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return HybridRetriever(store=VectorStore(settings, embedder=StubEmbedder()), settings=settings)


def _upload(retriever: HybridRetriever, session_id: str, texts: list[str]) -> None:
    retriever.store.add_session_chunks(
        [
            Chunk(chunk_id=f"{session_id}-{i}", text=text, title="bill.pdf",
                  source_kind=SourceKind.USER_DOCUMENT, session_id=session_id, doc_name="bill.pdf")
            for i, text in enumerate(texts)
        ],
        session_id,
    )


FILLER = [f"Paragraph {i} describes general charges and the plan. " * 3 for i in range(6)]


def test_an_exact_identifier_in_an_upload_is_found_lexically(retriever) -> None:
    _upload(retriever, "alice", [*FILLER, "Your reference number is ZX-9000-REF for this invoice."])
    hits = retriever._session_sparse("what is ZX-9000-REF", "alice", k=5)  # noqa: SLF001
    assert hits and "ZX-9000-REF" in hits[0].chunk.text


def test_lexical_session_search_never_crosses_sessions(retriever) -> None:
    _upload(retriever, "alice", ["Alice reference ZX-9000-REF is here."])
    _upload(retriever, "bob", ["Bob has an unrelated invoice about data usage."])
    assert retriever._session_sparse("ZX-9000-REF", "bob", k=5) == []  # noqa: SLF001
    assert all(
        h.chunk.session_id == "alice"
        for h in retriever._session_sparse("ZX-9000-REF", "alice", k=5)  # noqa: SLF001
    )


def test_session_chunks_requires_a_session_id(retriever) -> None:
    with pytest.raises(SessionScopeError):
        retriever.store.session_chunks("")


def test_unknown_session_has_no_sparse_hits(retriever) -> None:
    assert retriever._session_sparse("anything", "nobody", k=5) == []  # noqa: SLF001


def test_a_later_upload_is_visible_to_the_cached_index(retriever) -> None:
    _upload(retriever, "alice", ["First document about the monthly charge."])
    assert retriever._session_sparse("ZX-9000-REF", "alice", k=5) == []  # noqa: SLF001
    retriever.store.add_session_chunks(
        [Chunk(chunk_id="late", text="Second document mentions ZX-9000-REF clearly.",
               source_kind=SourceKind.USER_DOCUMENT, session_id="alice", doc_name="b.pdf")],
        "alice",
    )
    assert retriever._session_sparse("ZX-9000-REF", "alice", k=5)  # noqa: SLF001


def test_retrieve_surfaces_a_lexical_only_upload_hit_without_a_cosine(retriever) -> None:
    _upload(retriever, "alice", [*FILLER, "Your reference number is ZX-9000-REF for this invoice."])
    result = retriever.retrieve("ZX-9000-REF", session_id="alice")
    assert any("ZX-9000-REF" in sc.chunk.text for sc in result.chunks)
    assert result.used_session_docs
    for sc in result.chunks:
        assert sc.chunk.session_id == "alice"


def test_dense_scores_survive_fusion(retriever) -> None:
    """`score` is overwritten by fusion; the raw cosine must not be."""
    _upload(retriever, "alice", FILLER)
    result = retriever.retrieve("charges and the plan", session_id="alice")
    assert result.chunks
    scored_densely = [sc for sc in result.chunks if sc.dense_score is not None]
    assert scored_densely
    for sc in scored_densely:
        # Fused rank scores are ~1/(60+rank): tiny. The stub's cosines are not.
        assert sc.score < 0.1 < sc.dense_score, "fusion must not overwrite the cosine"


def test_ending_a_session_drops_its_cached_tokens(retriever, tmp_path) -> None:
    sessions = SessionManager(settings=retriever.settings, store=retriever.store)
    sessions.on_end(retriever.forget_session)
    session = sessions.create()
    _upload(retriever, session.session_id, ["Invoice mentions ZX-9000-REF here."])
    retriever._session_sparse("ZX-9000-REF", session.session_id, k=3)  # noqa: SLF001
    assert session.session_id in retriever._session_tokens  # noqa: SLF001

    sessions.end(session.session_id)

    assert session.session_id not in retriever._session_tokens  # noqa: SLF001
    assert retriever.store.session_size(session.session_id) == 0


def test_a_failing_end_hook_does_not_stop_eviction(retriever) -> None:
    sessions = SessionManager(settings=retriever.settings, store=retriever.store)

    def boom(_sid: str) -> None:
        raise RuntimeError("hook failure")

    sessions.on_end(boom)
    session = sessions.create()
    _upload(retriever, session.session_id, ["Some text that is uploaded."])
    sessions.end(session.session_id)
    assert retriever.store.session_size(session.session_id) == 0
