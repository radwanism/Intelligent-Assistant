"""GPU memory accounting and release.

Exists because of a failure that looks like a model being too large but is not.
Every model here is a **process-wide singleton**, and the services run in their
own processes. So a notebook that walks through the pipeline in-kernel — loading
the embedder, the reranker and the LLM to demonstrate them — and then launches
`core` as a subprocess ends up with *two* copies of each on one card:

    Process 2564 has 11.37 GiB in use     <- the notebook kernel
    this process has  3.18 GiB in use     <- core, loading its own copy
    GPU 0 ... of which 13.81 MiB is free

The models fit comfortably once. They do not fit twice. `release_all()` frees
the kernel's copies before the services start, which is what makes the
walkthrough and the launcher able to live in the same notebook.

None of this applies to the CPU profile, which is torch-free.
"""

from __future__ import annotations

import gc
import logging

log = logging.getLogger(__name__)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def memory_report() -> dict[str, float]:
    """Allocated / reserved / total VRAM in GiB, plus what other processes hold.

    `other_gib` is the interesting number when debugging OOM: PyTorch reports
    only its own allocations, so a second process holding 11 GiB is invisible in
    `memory_allocated()` and the error reads as though the model is too big.
    """
    if not cuda_available():
        return {}

    import torch

    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    gib = 1024**3
    return {
        "total_gib": round(total / gib, 2),
        "free_gib": round(free / gib, 2),
        "allocated_here_gib": round(allocated / gib, 2),
        "reserved_here_gib": round(reserved / gib, 2),
        # Used on the card but not by this process: another notebook kernel, a
        # stale service, or a previous run that was never torn down.
        "other_processes_gib": round((total - free - reserved) / gib, 2),
    }


def release_all() -> dict[str, float]:
    """Unload every cached model in THIS process and return the memory report.

    Call before launching the services from a notebook that has already loaded
    models, or the second load will OOM while the first sits idle.
    """
    from .llm.client import release_llm
    from .retrieval.embedder import release_embedder
    from .retrieval.rerank import release_reranker

    release_llm()
    release_embedder()
    release_reranker()

    try:
        from .documents.ocr import release_engine

        release_engine()
    except Exception:
        pass
    try:
        from .speech.tts import release_voices

        release_voices()
    except Exception:
        pass

    gc.collect()
    if cuda_available():
        import torch

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    report = memory_report()
    if report:
        log.info(
            "released models; %.2f GiB free of %.2f GiB (%.2f held by other processes)",
            report["free_gib"], report["total_gib"], report["other_processes_gib"],
        )
    return report


def format_report(report: dict[str, float]) -> str:
    if not report:
        return "no CUDA device (cpu-lite is torch-free)"
    lines = [
        f"  total           {report['total_gib']:>6.2f} GiB",
        f"  free            {report['free_gib']:>6.2f} GiB",
        f"  this process    {report['reserved_here_gib']:>6.2f} GiB reserved",
        f"  other processes {report['other_processes_gib']:>6.2f} GiB",
    ]
    if report["other_processes_gib"] > 1.0:
        lines.append(
            "\n  Another process holds significant VRAM — usually a notebook "
            "kernel\n  that loaded models, or a service still running from an "
            "earlier cell."
        )
    return "\n".join(lines)
