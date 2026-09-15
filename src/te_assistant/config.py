"""Runtime configuration and profile selection.

The system ships two profiles over ONE codebase (brief B.3):

  cpu-lite   the on-premises baseline and the claim we actually make. Must run
             with no GPU present, torch-free, on a 4-core laptop.
  gpu-colab  the development/demo runtime (Colab T4, 16 GB VRAM). Larger models,
             same interfaces.

Selection is by runtime device detection, never a code fork. Callers ask for a
slot (`settings.asr_model`) and get whatever the active profile provides.

Licensing note (brief B.3): every model named here for a CORE path carries a
commercially-usable license. CC-BY-NC models (jina-embeddings-v3,
jina-reranker-v2, f5-tts-egyptian-arabic) are deliberately excluded from the
core and may appear only in scripts/bench_models.py, per A.6.3.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = Path(os.getenv("TE_DATA_DIR", PROJECT_ROOT / "data"))


class Profile(str, Enum):
    CPU_LITE = "cpu-lite"
    GPU_COLAB = "gpu-colab"


def detect_profile() -> Profile:
    """Pick a profile from the environment, then from the hardware.

    TE_PROFILE always wins so the CPU path stays testable on a GPU box — we
    have to be able to *measure* the on-prem numbers, not just assert them.
    """
    forced = os.getenv("TE_PROFILE", "").strip().lower()
    if forced in {p.value for p in Profile}:
        return Profile(forced)

    if not _cuda_available():
        return Profile.CPU_LITE
    # A GPU too small for the 8B stack is worse than no GPU: we would spend the
    # VRAM and still have to swap models per request. Below ~12 GB, stay on CPU.
    return Profile.GPU_COLAB if _cuda_vram_gb() >= 12 else Profile.CPU_LITE


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415 - optional, gpu extra only
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_vram_gb() -> float:
    """Total VRAM on device 0, in GiB. Returns 0.0 if it cannot be determined.

    The attribute is `total_memory`. An earlier version read `total_mem`, which
    does not exist — the AttributeError was swallowed by a bare `except`, this
    returned 0.0, and `0.0 >= 12` meant **GPU detection could never succeed**.
    The system silently ran the CPU profile on a T4.

    Hence the logging below: a detection failure here does not raise and does
    not stop anything, so if it is not logged it is invisible. Anything that can
    only be noticed by someone reading the profile name deserves a warning.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return 0.0
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "CUDA is available but VRAM could not be read (%s: %s); "
            "falling back to the CPU profile",
            type(exc).__name__, exc,
        )
        return 0.0


@dataclass(frozen=True)
class ModelSlots:
    """One model per slot. Names are HF repo ids unless noted."""

    asr: str
    asr_compute_type: str
    embedder: str
    embed_dim: int
    reranker: str | None          # None => RRF ordering is final (documented cut line)
    llm_repo: str
    llm_file: str | None          # GGUF filename; None => transformers/AWQ load
    tts_voice_ar: str
    tts_voice_en: str
    ocr: str
    # Models we load on demand and unload after. On a 16 GB T4 the full stack
    # does not fit resident; OCR and TTS are the two that tolerate a cold start.
    lazy_slots: tuple[str, ...] = ()


CPU_SLOTS = ModelSlots(
    asr="small",                                  # faster-whisper, int8
    asr_compute_type="int8",
    # Apache-2.0, ONNX via fastembed, 0.22 GB. Chosen over multilingual-e5-large
    # (1024-d, 2.24 GB) purely on footprint: the on-prem dev target has ~5.5 GB
    # free and the LLM, ASR and TTS weights need ~2.6 GB of that. E5-large is the
    # better retriever, especially cross-lingually, and the gap between them is
    # measured in scripts/eval_rag.py rather than assumed — swap it here if the
    # deployment has the disk. BGE-M3 covers quality on the GPU profile.
    embedder="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embed_dim=384,
    reranker=None,                                # too slow on 4 CPU cores
    llm_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
    llm_file="qwen2.5-3b-instruct-q4_k_m.gguf",
    tts_voice_ar="ar_JO-kareem-medium",
    tts_voice_en="en_US-lessac-medium",
    ocr="rapidocr",
)

GPU_SLOTS = ModelSlots(
    asr="large-v3",                               # faster-whisper, fp16 (T4: no bf16)
    asr_compute_type="float16",
    embedder="BAAI/bge-m3",                       # MIT — replaces CC-BY-NC jina-v3
    embed_dim=1024,
    reranker="BAAI/bge-reranker-v2-m3",           # Apache-2.0
    llm_repo="Qwen/Qwen3-8B-AWQ",                 # Apache-2.0, ~5.5 GB on T4
    llm_file=None,
    tts_voice_ar="ar_JO-kareem-medium",
    tts_voice_en="en_US-lessac-medium",
    ocr="rapidocr",                               # PaddleOCR-VL Arabic support unproven
    lazy_slots=("ocr", "tts"),
)


@dataclass(frozen=True)
class Settings:
    profile: Profile
    slots: ModelSlots

    # --- paths ---
    data_dir: Path = DATA_DIR
    corpus_path: Path = field(default_factory=lambda: DATA_DIR / "corpus" / "te_eg.jsonl")
    chroma_dir: Path = field(default_factory=lambda: DATA_DIR / "chroma")
    eval_dir: Path = field(default_factory=lambda: DATA_DIR / "eval")
    sqlite_path: Path = field(default_factory=lambda: DATA_DIR / "assistant.db")
    session_files_dir: Path = field(default_factory=lambda: DATA_DIR / "sessions")
    traces_path: Path = field(default_factory=lambda: DATA_DIR / "traces.jsonl")

    # --- collections ---
    kb_collection: str = "kb_te_eg"          # shared, read-only
    session_collection: str = "kb_sessions"  # per-session, ALWAYS filtered

    # --- retrieval ---
    retrieve_k: int = 40      # candidates from hybrid fusion
    rerank_k: int = 6         # what actually reaches the prompt
    chunk_tokens: int = 450
    chunk_overlap: int = 60
    min_chunk_chars: int = 120

    # --- generation ---
    max_answer_tokens: int = 320
    temperature: float = 0.2

    # --- sessions (B.8) ---
    session_ttl_seconds: int = 60 * 60
    max_upload_mb: int = 20

    # --- crawl (B.3) ---
    # Seeded for BOTH language versions. te.eg serves English under /en/ as
    # separate URLs, and an Arabic-only seed set reaches almost none of them by
    # link-following — which leaves English questions retrieving Arabic pages
    # through the embedder's cross-lingual alignment alone. A.1 asks for
    # bilingual output, so the corpus has to be bilingual too.
    crawl_seeds: tuple[str, ...] = (
        # Arabic
        "https://te.eg/",
        "https://te.eg/personal/sitemap/",
        "https://te.eg/about-te/faq",
        "https://te.eg/ar/personal/ekyc-faqs",
        # English
        "https://te.eg/en/personal",
        "https://te.eg/en/about-te/faq",
        "https://te.eg/en/personal/sitemap",
        "https://te.eg/en",
    )
    crawl_max_pages: int = 400
    crawl_delay_seconds: float = 0.4
    crawl_timeout_seconds: float = 25.0

    # --- service wiring ---
    core_host: str = "127.0.0.1"
    core_port: int = 8000
    speech_host: str = "127.0.0.1"
    speech_port: int = 8001
    ui_port: int = 7860

    @property
    def speech_url(self) -> str:
        return os.getenv("TE_SPEECH_URL", f"http://{self.speech_host}:{self.speech_port}")

    @property
    def core_url(self) -> str:
        return os.getenv("TE_CORE_URL", f"http://{self.core_host}:{self.core_port}")

    @property
    def is_gpu(self) -> bool:
        return self.profile is Profile.GPU_COLAB


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    profile = detect_profile()
    slots = GPU_SLOTS if profile is Profile.GPU_COLAB else CPU_SLOTS
    settings = Settings(profile=profile, slots=slots)
    for path in (
        settings.data_dir,
        settings.corpus_path.parent,
        settings.chroma_dir,
        settings.session_files_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return settings
