"""The index and the embedder must agree on dimension, and when they do not the failure
has to be loud, early and actionable.

On the GPU profile `get_embedder` degrades to the 384-d ONNX model when BGE-M3 cannot
load. That is right for serving; for *building* an index it silently produced a 384-d
index that the 1024-d serving embedder could not query, and the only symptom was a
Chroma dimension error in a later notebook cell.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from te_assistant.config import GPU_SLOTS, Profile, Settings, get_settings
from te_assistant.ingest import build_index
from te_assistant.ingest.clean import CleanDoc
from te_assistant.retrieval import store as store_module
from te_assistant.retrieval.store import IndexMismatchError, VectorStore
from te_assistant.schemas import Language

DOC_TEXT = "WE internet plans are listed here with prices and quotas for every tier. " * 8


def _settings(tmp_path, *, embed_dim: int) -> Settings:
    base = get_settings()
    slots = dataclasses.replace(GPU_SLOTS, embed_dim=embed_dim)
    settings = Settings(
        profile=Profile.GPU_COLAB, slots=slots, data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma", corpus_path=tmp_path / "c.jsonl",
        sqlite_path=tmp_path / "t.db", session_files_dir=tmp_path / "sessions",
        traces_path=tmp_path / "traces.jsonl",
    )
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    assert base is not None
    return settings


class FixedDimEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def _docs() -> list[CleanDoc]:
    return [CleanDoc(url="https://te.eg/plans", title="Plans", text=DOC_TEXT,
                     language=Language.EN, lines=[])]


def _patch_store(monkeypatch, loaded_dim: int):
    """Make every VectorStore the builder creates load an embedder of `loaded_dim`."""
    real = store_module.VectorStore

    def factory(settings, *a, **k):
        return real(settings, embedder=FixedDimEmbedder(loaded_dim))

    monkeypatch.setattr(store_module, "VectorStore", factory)


# --------------------------------------------------------------------------
# The builder refuses to write the wrong dimension
# --------------------------------------------------------------------------
def test_a_fallen_back_embedder_cannot_build_the_index(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, embed_dim=1024)
    _patch_store(monkeypatch, loaded_dim=384)   # BGE-M3 fell back to the ONNX model

    with pytest.raises(SystemExit) as excinfo:
        build_index.stage_index(settings, _docs(), reset=True)

    message = str(excinfo.value)
    assert "1024" in message and "384" in message
    assert "fell back" in message
    assert "fetch_models.py --only embedder" in message, "it must say how to recover"


def test_nothing_is_written_when_the_dimension_is_wrong(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, embed_dim=1024)
    real = store_module.VectorStore(settings, embedder=FixedDimEmbedder(1024))
    real.add_kb_chunks([])           # a collection exists and is empty
    _patch_store(monkeypatch, loaded_dim=384)

    with pytest.raises(SystemExit):
        build_index.stage_index(settings, _docs(), reset=True)

    assert real.kb_size() == 0, "a refused build must not touch the index"


def test_a_matching_embedder_builds_normally(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, embed_dim=1024)
    _patch_store(monkeypatch, loaded_dim=1024)
    written = build_index.stage_index(settings, _docs(), reset=True)
    assert written >= 1


# --------------------------------------------------------------------------
# A mismatch that already exists is reported clearly
# --------------------------------------------------------------------------
def test_querying_a_384d_index_with_1024d_vectors_explains_the_cause(tmp_path) -> None:
    settings = _settings(tmp_path, embed_dim=1024)
    old = VectorStore(settings, embedder=FixedDimEmbedder(384))
    old.add_kb_chunks(
        build_index.chunks_from_document(
            text=DOC_TEXT, title="Plans", url="https://te.eg/plans", language=Language.EN,
        )
    )

    serving = VectorStore(settings, embedder=FixedDimEmbedder(1024))
    with pytest.raises(IndexMismatchError) as excinfo:
        serving.search_kb("internet plans", k=3)

    message = str(excinfo.value)
    assert "1024-d" in message
    assert "build_index --stage index --reset" in message
    assert "after the embedder has finished downloading" in message


def test_unrelated_query_errors_are_not_relabelled(tmp_path) -> None:
    class Exploding:
        def count(self) -> int:
            return 3

        def query(self, **kwargs):
            raise RuntimeError("disk is on fire")

    settings = _settings(tmp_path, embed_dim=384)
    store = VectorStore(settings, embedder=FixedDimEmbedder(384))
    with pytest.raises(RuntimeError, match="disk is on fire") as excinfo:
        store._search(Exploding(), "q", k=3, where=None)  # noqa: SLF001
    assert not isinstance(excinfo.value, IndexMismatchError)
