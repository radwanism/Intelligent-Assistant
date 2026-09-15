"""Profile selection — including the GPU-detection regression.

This file exists because of a bug that shipped silently: `_cuda_vram_gb` read
`total_mem` instead of `total_memory`, a bare `except` swallowed the
AttributeError, the function returned 0.0, and `0.0 >= 12` meant the GPU profile
could **never** be selected. The system ran the CPU stack on a T4 and said
nothing.

Nothing crashed, no test failed, and the only symptom was a profile name in a
print statement. That is exactly the class of bug that needs a test rather than
a code review.
"""

from __future__ import annotations

import sys
import types

import pytest

from te_assistant import config
from te_assistant.config import CPU_SLOTS, GPU_SLOTS, Profile, detect_profile


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is lru_cached, so the profile sticks for the process."""
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _fake_torch(*, available: bool, total_memory: int | None):
    """A stand-in for torch exposing only what detection touches."""
    props = types.SimpleNamespace()
    if total_memory is not None:
        props.total_memory = total_memory

    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_properties=lambda _index: props,
    )
    return types.SimpleNamespace(cuda=cuda)


@pytest.fixture()
def torch_stub(monkeypatch):
    def install(*, available: bool, total_memory: int | None):
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(available=available, total_memory=total_memory)
        )
    return install


# --------------------------------------------------------------------------
# Environment override
# --------------------------------------------------------------------------
def test_env_override_wins(monkeypatch) -> None:
    """Forcing cpu-lite on a GPU box is how on-prem latency gets *measured*."""
    monkeypatch.setenv("TE_PROFILE", "cpu-lite")
    assert detect_profile() is Profile.CPU_LITE

    monkeypatch.setenv("TE_PROFILE", "gpu-colab")
    assert detect_profile() is Profile.GPU_COLAB


def test_unknown_env_value_is_ignored(monkeypatch, torch_stub) -> None:
    monkeypatch.setenv("TE_PROFILE", "definitely-not-a-profile")
    torch_stub(available=False, total_memory=None)
    assert detect_profile() is Profile.CPU_LITE


# --------------------------------------------------------------------------
# Hardware detection
# --------------------------------------------------------------------------
def test_no_torch_means_cpu(monkeypatch) -> None:
    monkeypatch.delenv("TE_PROFILE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # import returns None -> AttributeError
    assert detect_profile() is Profile.CPU_LITE


def test_no_cuda_means_cpu(monkeypatch, torch_stub) -> None:
    monkeypatch.delenv("TE_PROFILE", raising=False)
    torch_stub(available=False, total_memory=None)
    assert detect_profile() is Profile.CPU_LITE


def test_t4_selects_the_gpu_profile(monkeypatch, torch_stub) -> None:
    """THE REGRESSION: a 16 GB T4 must select gpu-colab.

    This failed before the total_mem -> total_memory fix, silently.
    """
    monkeypatch.delenv("TE_PROFILE", raising=False)
    torch_stub(available=True, total_memory=16 * 1024**3)
    assert detect_profile() is Profile.GPU_COLAB


def test_small_gpu_stays_on_cpu(monkeypatch, torch_stub) -> None:
    """A GPU too small for the 8B stack is worse than no GPU."""
    monkeypatch.delenv("TE_PROFILE", raising=False)
    torch_stub(available=True, total_memory=8 * 1024**3)
    assert detect_profile() is Profile.CPU_LITE


def test_unreadable_vram_degrades_to_cpu(monkeypatch, torch_stub, caplog) -> None:
    """If VRAM cannot be read we fall back — but we must SAY SO.

    Silence here is what let the original bug live.
    """
    monkeypatch.delenv("TE_PROFILE", raising=False)
    torch_stub(available=True, total_memory=None)   # property missing entirely
    with caplog.at_level("WARNING"):
        assert detect_profile() is Profile.CPU_LITE
    assert any("VRAM" in record.message for record in caplog.records), (
        "a silent fallback to CPU on a GPU machine must be logged"
    )


# --------------------------------------------------------------------------
# Slots stay consistent with the profile
# --------------------------------------------------------------------------
def test_slots_match_the_selected_profile(monkeypatch, torch_stub) -> None:
    monkeypatch.delenv("TE_PROFILE", raising=False)
    torch_stub(available=True, total_memory=16 * 1024**3)
    assert config.get_settings().slots is GPU_SLOTS

    config.get_settings.cache_clear()
    monkeypatch.setenv("TE_PROFILE", "cpu-lite")
    assert config.get_settings().slots is CPU_SLOTS


def test_embedding_dimensions_differ_between_profiles() -> None:
    """The reason an index built on one profile cannot be queried by the other.

    384 vs 1024 is why COLAB.md makes re-indexing a mandatory step on GPU.
    """
    assert CPU_SLOTS.embed_dim == 384
    assert GPU_SLOTS.embed_dim == 1024
    assert CPU_SLOTS.embed_dim != GPU_SLOTS.embed_dim


def test_cpu_profile_has_no_reranker() -> None:
    """Documented cut: too slow on 4 cores, falls back to RRF ordering."""
    assert CPU_SLOTS.reranker is None
    assert GPU_SLOTS.reranker is not None
