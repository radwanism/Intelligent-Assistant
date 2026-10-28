"""Licence is a hard filter, so it is a test.

The brief disqualifies non-commercial models from the core path regardless of
how well they score. Both violations found in review — a research-licensed LLM
and a research-licensed voice — were one-line choices that nothing would have
flagged. These tests make the filter mechanical.
"""

from __future__ import annotations

import dataclasses

import pytest

from te_assistant.config import (
    COMMERCIAL_LICENCES,
    CORE_MODEL_LICENCES,
    CPU_SLOTS,
    GPU_SLOTS,
    UNVERIFIED_LICENCES,
    core_models,
)
from te_assistant.speech.tts import VOICE_PATHS

# Models that have been looked at and rejected for the serving path. They may
# appear in scripts/bench_models.py (brief A.6.3) and nowhere else.
DISQUALIFIED = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen-research (non-commercial)",
    "Qwen/Qwen2.5-3B-Instruct-GGUF": "qwen-research (non-commercial)",
    "Qwen/Qwen2.5-72B-Instruct": "qwen (restricted)",
    "jinaai/jina-embeddings-v3": "CC-BY-NC-4.0",
    "jinaai/jina-reranker-v2-base-multilingual": "CC-BY-NC-4.0",
    "MAdel121/f5-tts-egyptian-arabic": "CC-BY-NC-4.0",
    "en_US-lessac-medium": "Lessac Blizzard 2013 — research only, excludes synthesis products",
}


@pytest.mark.parametrize("profile", ["cpu", "gpu"])
def test_every_core_model_has_a_recorded_licence(profile) -> None:
    slots = CPU_SLOTS if profile == "cpu" else GPU_SLOTS
    unregistered = [m for m in core_models(slots) if m not in CORE_MODEL_LICENCES]
    assert not unregistered, f"register after reading their cards: {unregistered}"


@pytest.mark.parametrize("profile", ["cpu", "gpu"])
def test_every_core_model_is_commercially_usable_or_visibly_flagged(profile) -> None:
    slots = CPU_SLOTS if profile == "cpu" else GPU_SLOTS
    for model in core_models(slots):
        licence = CORE_MODEL_LICENCES[model]
        if licence in COMMERCIAL_LICENCES:
            continue
        assert licence == "unverified" and model in UNVERIFIED_LICENCES, (
            f"{model} is {licence!r}: not commercially usable and not on the flagged list"
        )


@pytest.mark.parametrize("profile", ["cpu", "gpu"])
def test_no_disqualified_model_is_in_a_core_slot(profile) -> None:
    slots = CPU_SLOTS if profile == "cpu" else GPU_SLOTS
    present = set(core_models(slots)) & set(DISQUALIFIED)
    assert not present, {m: DISQUALIFIED[m] for m in present}


def test_the_unverified_list_cannot_grow_silently() -> None:
    """Anything 'unverified' in the registry must be named in the flagged set, and vice versa."""
    unverified = {m for m, lic in CORE_MODEL_LICENCES.items() if lic == "unverified"}
    assert unverified == set(UNVERIFIED_LICENCES)


def test_every_registered_licence_is_a_known_value() -> None:
    allowed = COMMERCIAL_LICENCES | {"unverified"}
    assert set(CORE_MODEL_LICENCES.values()) <= allowed


def test_every_selectable_voice_is_registered() -> None:
    assert set(VOICE_PATHS) <= set(CORE_MODEL_LICENCES), "a voice with no recorded licence"


def test_a_swapped_slot_is_caught() -> None:
    """The guard has to bite: put the old research-licensed LLM back and it must fail."""
    swapped = dataclasses.replace(CPU_SLOTS, llm_repo="Qwen/Qwen2.5-3B-Instruct-GGUF")
    assert "Qwen/Qwen2.5-3B-Instruct-GGUF" not in CORE_MODEL_LICENCES
    assert set(core_models(swapped)) & set(DISQUALIFIED)


def test_the_cpu_llm_is_the_non_thinking_variant() -> None:
    """The hybrid Qwen3-4B emits <think> blocks, which the intent call's stop
    sequence would swallow — silently routing every action to the fallback."""
    assert "Instruct-2507" in CPU_SLOTS.llm_repo
    assert CPU_SLOTS.llm_file and "Instruct-2507" in CPU_SLOTS.llm_file
    assert CPU_SLOTS.llm_file.endswith(".gguf")
