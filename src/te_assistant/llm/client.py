"""LLM client — one interface, two backends.

`cpu-lite`  Qwen2.5-3B-Instruct Q4_K_M through llama.cpp
`gpu-colab` Qwen3-8B-AWQ through transformers

Note what is *absent* from this module: any import of `actions/`, `db`, or the
permission checker. The model layer has no route to the database, by
construction (brief B.4). It returns text and JSON; deciding what may be done
with either happens elsewhere.

Generation is capped hard (`max_answer_tokens`, default 320). On four CPU cores
Qwen2.5-3B produces roughly 4-6 tokens/second, so an uncapped answer is a
90-second answer. RAG answers should be short anyway — the citations carry the
detail — so the cap improves both latency and quality.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from ..config import Profile, Settings, get_settings

log = logging.getLogger(__name__)


class LLM(Protocol):
    def generate(
        self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float,
        stop: list[str] | None = None,
    ) -> str: ...

    def stream(
        self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float,
        stop: list[str] | None = None,
    ) -> Iterator[str]: ...


class LlamaCppLLM:
    """CPU backend. Threads are pinned to physical cores.

    llama.cpp defaults to all logical processors, which on a 4-core/8-thread
    laptop makes it slower, not faster — the hyperthread siblings contend for
    the same vector units. Physical core count is the right value.
    """

    def __init__(self, settings: Settings) -> None:
        from llama_cpp import Llama

        model_path = _resolve_gguf(settings)
        n_threads = _physical_cores()
        self._lock = threading.Lock()
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=4096,          # room for ~6 chunks plus history and the answer
            n_threads=n_threads,
            n_batch=256,
            verbose=False,
            logits_all=False,
        )
        log.info("llama.cpp loaded %s (threads=%d)", model_path.name, n_threads)

    def generate(
        self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float,
        stop: list[str] | None = None,
    ) -> str:
        # llama.cpp's Python binding is not re-entrant; concurrent sessions must
        # serialise here or the KV cache interleaves between requests.
        with self._lock:
            result = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
            )
        return (result["choices"][0]["message"]["content"] or "").strip()

    def stream(
        self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        with self._lock:
            for part in self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                stream=True,
            ):
                delta = part["choices"][0].get("delta", {})
                if token := delta.get("content"):
                    yield token


class TransformersLLM:
    """GPU backend: Qwen3-8B-AWQ. fp16 only — a T4 is Turing, no bf16."""

    def __init__(self, settings: Settings) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = settings.slots.llm_repo
        self._tokenizer = AutoTokenizer.from_pretrained(repo)
        self._model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float16, device_map="auto"
        )
        self._model.eval()
        self._torch = torch
        log.info("transformers loaded %s", repo)

    def _prepare(self, messages: list[dict[str, str]]):
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen3 ships a reasoning mode. For grounded RAG it burns latency on
            # tokens the user never sees, so it stays off.
            enable_thinking=False,
        )
        return self._tokenizer([text], return_tensors="pt").to(self._model.device)

    def generate(
        self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float,
        stop: list[str] | None = None,
    ) -> str:
        inputs = self._prepare(messages)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-4),
                do_sample=temperature > 0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _truncate_at_stop(text, stop)

    def stream(
        self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self._prepare(messages)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-4),
            do_sample=temperature > 0.0,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self._model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        yield from streamer


def _truncate_at_stop(text: str, stop: list[str] | None) -> str:
    for token in stop or []:
        index = text.find(token)
        if index != -1:
            text = text[:index]
    return text.strip()


def _physical_cores() -> int:
    try:
        count = os.cpu_count() or 4
    except Exception:
        count = 4
    # os.cpu_count() reports logical processors; halve for physical, floor at 2.
    return max(2, count // 2)


def _resolve_gguf(settings: Settings) -> Path:
    """Find the GGUF locally, or fetch it once from the Hub.

    `TE_MODELS_DIR` lets an air-gapped install point at a pre-staged directory
    so nothing reaches the network at serve time (README: offline install).
    """
    slots = settings.slots
    filename = slots.llm_file or ""
    local_dir = os.getenv("TE_MODELS_DIR")
    if local_dir:
        candidate = Path(local_dir) / filename
        if candidate.exists():
            return candidate

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=slots.llm_repo, filename=filename)
    return Path(path)


_INSTANCE: LLM | None = None
_INSTANCE_LOCK = threading.Lock()


def get_llm(settings: Settings | None = None) -> LLM:
    """Process-wide singleton. Loading a model per request would be fatal on CPU."""
    global _INSTANCE
    settings = settings or get_settings()
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        if settings.profile is Profile.GPU_COLAB:
            try:
                _INSTANCE = TransformersLLM(settings)
            except Exception as exc:
                log.warning("GPU backend unavailable (%s); falling back to llama.cpp", exc)
                _INSTANCE = LlamaCppLLM(settings)
        else:
            _INSTANCE = LlamaCppLLM(settings)
    return _INSTANCE
