"""Model release and VRAM accounting.

Written after a CUDA OOM that was not a too-large-model problem at all:

    Process 2564 has 11.37 GiB in use     <- the notebook kernel
    this process has  3.18 GiB in use     <- core, loading its own copy
    GPU 0 ... of which 13.81 MiB is free

Every model is a process-wide singleton and the services run in their own
processes, so a notebook that demonstrates the pipeline in-kernel and then
launches the services loads each model twice onto one card. The models fit once
and not twice.

PyTorch reports only its own allocations, so the 11 GiB held by the other
process is invisible in the traceback — which is why `memory_report` breaks out
`other_processes_gib` and `/health` surfaces it.

These tests must pass on a machine with no GPU, which is where the CPU profile
runs, so everything degrades to a no-op rather than raising.
"""

from __future__ import annotations

import sys
import types

import pytest

from te_assistant import gpu


# --------------------------------------------------------------------------
# Must be safe without CUDA
# --------------------------------------------------------------------------
def test_release_all_is_a_noop_without_torch(monkeypatch) -> None:
    """cpu-lite is torch-free; releasing must not raise."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert gpu.release_all() == {}


def test_memory_report_empty_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "cuda_available", lambda: False)
    assert gpu.memory_report() == {}


def test_format_report_handles_no_gpu() -> None:
    assert "no CUDA device" in gpu.format_report({})


def test_individual_releasers_are_safe_without_torch() -> None:
    from te_assistant.llm.client import release_llm
    from te_assistant.retrieval.embedder import release_embedder
    from te_assistant.retrieval.rerank import release_reranker

    release_llm()
    release_embedder()
    release_reranker()


# --------------------------------------------------------------------------
# Accounting, with a stubbed CUDA
# --------------------------------------------------------------------------
GIB = 1024**3


@pytest.fixture()
def fake_cuda(monkeypatch):
    def install(*, total_gib: float, free_gib: float, reserved_gib: float):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (int(free_gib * GIB), int(total_gib * GIB)),
            memory_allocated=lambda: int(reserved_gib * GIB * 0.95),
            memory_reserved=lambda: int(reserved_gib * GIB),
            empty_cache=lambda: None,
            synchronize=lambda: None,
        )
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
        monkeypatch.setattr(gpu, "cuda_available", lambda: True)
    return install


def test_other_process_memory_is_attributed(fake_cuda) -> None:
    """THE REGRESSION: the 11 GiB held elsewhere must be visible.

    Total 14.56, free 0.01, this process reserved 3.18 -> ~11.37 elsewhere.
    Without this breakdown the OOM reads as 'the model is too big'.
    """
    fake_cuda(total_gib=14.56, free_gib=0.013, reserved_gib=3.18)
    report = gpu.memory_report()

    assert report["total_gib"] == pytest.approx(14.56, abs=0.02)
    assert report["other_processes_gib"] == pytest.approx(11.37, abs=0.05)


def test_report_flags_a_foreign_hog(fake_cuda) -> None:
    fake_cuda(total_gib=14.56, free_gib=0.013, reserved_gib=3.18)
    text = gpu.format_report(gpu.memory_report())
    assert "other processes" in text
    assert "notebook kernel" in text, "should name the usual culprit"


def test_healthy_gpu_is_not_flagged(fake_cuda) -> None:
    """A single process using the card normally must not warn."""
    fake_cuda(total_gib=14.56, free_gib=3.0, reserved_gib=11.5)
    report = gpu.memory_report()
    assert report["other_processes_gib"] == pytest.approx(0.06, abs=0.1)
    assert "notebook kernel" not in gpu.format_report(report)


def test_release_all_reports_after_freeing(fake_cuda) -> None:
    fake_cuda(total_gib=14.56, free_gib=12.0, reserved_gib=0.5)
    report = gpu.release_all()
    assert report["free_gib"] == pytest.approx(12.0, abs=0.02)


def test_release_llm_clears_the_singleton(monkeypatch) -> None:
    """Also the only way to retry a failed load: get_llm caches the failure."""
    from te_assistant.llm import client

    monkeypatch.setattr(client, "_INSTANCE", object())
    client.release_llm()
    assert client._INSTANCE is None


def test_release_embedder_clears_the_cache(monkeypatch) -> None:
    from te_assistant.retrieval import embedder

    monkeypatch.setitem(embedder._CACHE, "stub", object())
    embedder.release_embedder()
    assert embedder._CACHE == {}
