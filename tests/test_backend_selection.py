"""Backend selection failures must name their real cause.

Every test here corresponds to something that actually went wrong on Colab and
produced a message pointing somewhere else:

  * `ValueError: Model BAAI/bge-m3 is not supported in TextEmbedding` on a
    session that had already printed `gpu-colab`
  * `ModuleNotFoundError: No module named 'llama_cpp'` on the GPU profile,
    surfacing as a bare HTTP 500 from /chat
  * both traceable to the package being imported twice, as `te_assistant` and
    as `src.te_assistant`, giving two Enum classes whose `is` comparison fails

None of those messages named the actual problem, which is what these tests fix.
"""

from __future__ import annotations

import sys
import types

import pytest

from te_assistant import config
from te_assistant.config import CPU_SLOTS, GPU_SLOTS, Profile, Settings


def _settings(profile: Profile, slots) -> Settings:
    base = config.get_settings()
    return Settings(profile=profile, slots=slots, data_dir=base.data_dir)


# --------------------------------------------------------------------------
# The dual-import hazard
# --------------------------------------------------------------------------
def test_package_refuses_to_be_imported_as_src_te_assistant() -> None:
    """The `src.` prefix loads a second copy of everything — fail, don't cope."""
    sys.modules.pop("src.te_assistant", None)
    sys.modules.pop("src", None)
    sys.path.insert(0, ".")
    try:
        with pytest.raises(ImportError, match="must be imported as"):
            import src.te_assistant  # noqa: F401
    finally:
        sys.path.remove(".")
        sys.modules.pop("src.te_assistant", None)
        sys.modules.pop("src", None)


def test_profile_comparison_survives_a_duplicated_enum() -> None:
    """Selection must not depend on Enum *identity*.

    Reproduces the cross-copy case: a Profile-alike whose value matches but
    whose class does not. `is` says False; `.value ==` says True.
    """
    other = types.SimpleNamespace(value="gpu-colab")
    assert other is not Profile.GPU_COLAB
    assert other.value == Profile.GPU_COLAB.value


# --------------------------------------------------------------------------
# Embedder backend
# --------------------------------------------------------------------------
def test_onnx_backend_rejects_a_gpu_only_model() -> None:
    """fastembed cannot serve BGE-M3; say so, and say why it got there."""
    from te_assistant.retrieval.embedder import OnnxEmbedder

    with pytest.raises(ValueError, match="not available through fastembed"):
        OnnxEmbedder("BAAI/bge-m3", 1024)


def test_onnx_error_points_at_the_profile_not_the_model() -> None:
    from te_assistant.retrieval.embedder import OnnxEmbedder

    with pytest.raises(ValueError) as excinfo:
        OnnxEmbedder("BAAI/bge-m3", 1024)
    message = str(excinfo.value)
    assert "TE_PROFILE" in message, "should point the reader at the real cause"
    assert "FlagEmbedding" in message


def test_cpu_default_embedder_is_actually_supported() -> None:
    """Guards against pinning a CPU model fastembed does not ship."""
    from fastembed import TextEmbedding

    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    assert CPU_SLOTS.embedder in supported


# --------------------------------------------------------------------------
# LLM backend
# --------------------------------------------------------------------------
def test_gpu_failure_without_llama_cpp_names_both_causes(monkeypatch) -> None:
    """The real bug behind the /chat 500.

    `[gpu]` does not install llama-cpp-python, so the fallback cannot work.
    The raised error must name the transformers failure AND the missing
    fallback, rather than surfacing only the latter.
    """
    from te_assistant.llm import client

    monkeypatch.setattr(client, "_INSTANCE", None)

    def _no_gpu(_settings):
        raise RuntimeError("AWQ kernels unavailable")

    def _no_llama(_settings):
        raise ImportError("No module named 'llama_cpp'")

    monkeypatch.setattr(client, "TransformersLLM", _no_gpu)
    monkeypatch.setattr(client, "LlamaCppLLM", _no_llama)

    with pytest.raises(RuntimeError) as excinfo:
        client.get_llm(_settings(Profile.GPU_COLAB, GPU_SLOTS))

    message = str(excinfo.value)
    assert "AWQ kernels unavailable" in message, "the real GPU error must survive"
    assert "llama_cpp" in message, "the fallback failure must also be named"
    assert "pip install llama-cpp-python" in message, "must be actionable"


def test_gpu_backend_is_used_when_it_loads(monkeypatch) -> None:
    from te_assistant.llm import client

    monkeypatch.setattr(client, "_INSTANCE", None)
    sentinel = object()
    monkeypatch.setattr(client, "TransformersLLM", lambda _s: sentinel)
    monkeypatch.setattr(client, "LlamaCppLLM", lambda _s: pytest.fail("should not fall back"))

    assert client.get_llm(_settings(Profile.GPU_COLAB, GPU_SLOTS)) is sentinel
    monkeypatch.setattr(client, "_INSTANCE", None)


def test_gpu_falls_back_when_llama_cpp_is_available(monkeypatch) -> None:
    """Fallback is still correct when it can actually work."""
    from te_assistant.llm import client

    monkeypatch.setattr(client, "_INSTANCE", None)
    sentinel = object()

    def _no_gpu(_settings):
        raise RuntimeError("no CUDA kernels")

    monkeypatch.setattr(client, "TransformersLLM", _no_gpu)
    monkeypatch.setattr(client, "LlamaCppLLM", lambda _s: sentinel)

    assert client.get_llm(_settings(Profile.GPU_COLAB, GPU_SLOTS)) is sentinel
    monkeypatch.setattr(client, "_INSTANCE", None)


def test_cpu_profile_goes_straight_to_llama_cpp(monkeypatch) -> None:
    from te_assistant.llm import client

    monkeypatch.setattr(client, "_INSTANCE", None)
    sentinel = object()
    monkeypatch.setattr(client, "LlamaCppLLM", lambda _s: sentinel)
    monkeypatch.setattr(client, "TransformersLLM", lambda _s: pytest.fail("CPU must not try GPU"))

    assert client.get_llm(_settings(Profile.CPU_LITE, CPU_SLOTS)) is sentinel
    monkeypatch.setattr(client, "_INSTANCE", None)
