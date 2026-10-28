"""The evaluation harness is code too — a metric that is computed wrongly is worse
than no metric, because it gets quoted."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from te_assistant.config import Settings, get_settings
from te_assistant.retrieval.hybrid import HybridRetriever
from te_assistant.retrieval.store import VectorStore
from te_assistant.schemas import Chunk, Language, SourceKind

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_rag = _load_script("eval_rag")
eval_asr = _load_script("eval_asr")


# --------------------------------------------------------------------------
# The committed golden set
# --------------------------------------------------------------------------
def _golden() -> list[dict]:
    path = ROOT / "data" / "eval" / "golden_set.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_golden_set_has_the_expected_shape() -> None:
    golden = _golden()
    in_scope = [g for g in golden if not g.get("expect_refusal")]
    out_of_scope = [g for g in golden if g.get("expect_refusal")]

    assert len(in_scope) == 38, "the in-scope set the published numbers were measured on"
    assert len(out_of_scope) >= 10
    for item in in_scope:
        assert item["expected_url_contains"], f"no target for: {item['question']}"
    for item in out_of_scope:
        assert item["expected_url_contains"] == [], "an out-of-scope question has no right page"
    assert len({g["question"] for g in golden}) == len(golden), "duplicate questions"


def test_out_of_scope_questions_cover_every_register() -> None:
    languages = {g["language"] for g in _golden() if g.get("expect_refusal")}
    assert {"en", "ar", "arz"} <= languages


# --------------------------------------------------------------------------
# refusal_report
# --------------------------------------------------------------------------
def test_refusal_rates_are_computed_at_the_threshold() -> None:
    report = eval_rag.refusal_report([0.5, 0.6, 0.1, 0.7], [0.05, 0.08, 0.4, 0.02], threshold=0.15)
    assert report["in_scope"]["false_refusal_rate"] == pytest.approx(0.25)
    assert report["out_of_scope"]["refusal_rate"] == pytest.approx(0.75)


def test_separable_populations_get_a_suggested_threshold_between_them() -> None:
    report = eval_rag.refusal_report([0.4, 0.5, 0.6], [0.05, 0.1, 0.2], threshold=0.15)
    assert report["separable"] is True
    assert 0.2 < report["suggested_threshold"] < 0.4
    assert report["overlap"] is None


def test_overlapping_populations_are_reported_honestly_not_papered_over() -> None:
    report = eval_rag.refusal_report([0.1, 0.5, 0.6], [0.05, 0.3, 0.2], threshold=0.15)
    assert report["separable"] is False
    assert report["suggested_threshold"] is None
    assert "no threshold separates" in report["overlap"]


def test_empty_populations_do_not_crash() -> None:
    report = eval_rag.refusal_report([], [], threshold=0.15)
    assert report["in_scope"]["false_refusal_rate"] is None
    assert report["out_of_scope"]["refusal_rate"] is None
    assert "separable" not in report


def test_confidence_summary() -> None:
    summary = eval_rag.summarise_confidence([0.1, 0.2, 0.3, 0.4, 0.5])
    assert summary["min"] == 0.1 and summary["max"] == 0.5 and summary["median"] == 0.3
    assert eval_rag.summarise_confidence([]) == {}


# --------------------------------------------------------------------------
# evaluate()
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
    found = HybridRetriever(store=VectorStore(settings, embedder=StubEmbedder()), settings=settings)
    found.store.add_kb_chunks([
        Chunk(chunk_id=f"k{i}", text=f"Page {i} about WE plans and recharge codes.", title=f"P{i}",
              url=f"https://te.eg/page-{i}", language=Language.EN, source_kind=SourceKind.TE_EG)
        for i in range(3)
    ])
    found.warmup()
    return found


GOLDEN = [
    {"question": "recharge codes", "language": "en", "expected_url_contains": ["page-1"]},
    {"question": "what is the weather", "language": "en", "expected_url_contains": [],
     "expect_refusal": True},
]


def test_out_of_scope_questions_do_not_count_against_recall(retriever) -> None:
    report = eval_rag.evaluate(retriever, GOLDEN, 5)
    assert report["overall"]["n"] == 1, "only in-scope questions are scored for recall"
    assert report["overall"]["recall@5"] == 1.0
    assert report["misses"] == []
    assert report["refusal"]["out_of_scope"]["n"] == 1
    assert report["refusal"]["in_scope"]["n"] == 1


def test_evaluate_uses_the_serving_ordering_function(retriever, monkeypatch) -> None:
    """The harness must call select_context, not carry its own copy of the ordering."""
    calls = []
    real = eval_rag.select_context

    def spy(query, chunks, settings, **kwargs):
        calls.append(kwargs)
        return real(query, chunks, settings, **kwargs)

    monkeypatch.setattr(eval_rag, "select_context", spy)
    eval_rag.evaluate(retriever, GOLDEN, 5)
    assert len(calls) == 2, "one call per question, in-scope and out-of-scope alike"


def test_the_reranker_flag_reaches_the_ordering_and_is_reported(retriever, monkeypatch) -> None:
    class ReverseReranker:
        def rerank(self, query, chunks, *, top_k):
            ordered = list(reversed(chunks))
            for index, sc in enumerate(ordered):
                sc.rerank_score = 0.9 - index * 0.1
            return ordered[:top_k]

    monkeypatch.setattr("te_assistant.retrieval.context.get_reranker", lambda s: ReverseReranker())
    monkeypatch.setattr(eval_rag, "get_reranker", lambda s: ReverseReranker())

    with_rerank = eval_rag.evaluate(retriever, GOLDEN, 5)
    without = eval_rag.evaluate(retriever, GOLDEN, 5, use_reranker=False)

    assert with_rerank["reranker_in_use"] is True
    assert without["reranker_in_use"] is False
    top_with = with_rerank["rows"][0]["top_urls"]
    top_without = without["rows"][0]["top_urls"]
    assert top_with != top_without or len(top_with) < 2, "reranking should change the order"


def test_without_a_reranker_the_report_says_so(retriever, monkeypatch) -> None:
    monkeypatch.setattr(eval_rag, "get_reranker", lambda s: None)
    assert eval_rag.evaluate(retriever, GOLDEN, 5)["reranker_in_use"] is False


def test_a_missing_golden_set_explains_how_to_recover(tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        eval_rag.load_golden(tmp_path / "nope.jsonl")
    assert "git checkout" in str(excinfo.value)
    assert "make_golden_set" not in str(excinfo.value), "that script does not exist"


# --------------------------------------------------------------------------
# ASR manifest validation
# --------------------------------------------------------------------------
def _manifest(**overrides) -> dict:
    item = {"audio": "a.wav", "reference": "عايز أعرف", "dialect": "arz", "condition": "clean"}
    item.update(overrides)
    return item


def test_the_shipped_example_manifest_covers_every_dialect_and_condition() -> None:
    path = ROOT / "data" / "eval" / "audio" / "manifest.example.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    for item in items:
        assert not [f for f in eval_asr.REQUIRED_FIELDS if f not in item]
    _, warnings = eval_asr.validate_manifest(items, path.parent)
    assert warnings == [], "the template should itself be a complete protocol"


def test_a_missing_audio_file_is_a_problem(tmp_path) -> None:
    problems, _ = eval_asr.validate_manifest([_manifest()], tmp_path)
    assert any("file not found" in p for p in problems)


def test_a_present_audio_file_is_fine(tmp_path) -> None:
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    problems, _ = eval_asr.validate_manifest([_manifest()], tmp_path)
    assert problems == []


def test_missing_fields_are_reported_by_name(tmp_path) -> None:
    problems, _ = eval_asr.validate_manifest([{"audio": "a.wav"}], tmp_path)
    joined = " ".join(problems)
    assert "'reference'" in joined and "'dialect'" in joined and "'condition'" in joined


def test_an_empty_reference_is_a_problem_unless_the_clip_is_silence(tmp_path) -> None:
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "s.wav").write_bytes(b"x")
    problems, _ = eval_asr.validate_manifest(
        [_manifest(reference=""), _manifest(audio="s.wav", reference="", condition="silence")],
        tmp_path,
    )
    assert len(problems) == 1 and "a.wav" in problems[0]


def test_missing_noisy_or_silent_coverage_is_warned_about(tmp_path) -> None:
    (tmp_path / "a.wav").write_bytes(b"x")
    _, warnings = eval_asr.validate_manifest([_manifest()], tmp_path)
    joined = " ".join(warnings)
    assert "'noisy'" in joined and "'silence'" in joined
