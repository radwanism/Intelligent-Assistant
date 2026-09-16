"""Telecom Egypt Intelligent Assistant.

Import this package as `te_assistant`, never as `src.te_assistant`.

The `src.` form appears to work — `src/` has no `__init__.py`, so Python treats
it as a namespace package and the import succeeds. What it actually does is load
a **second, independent copy** of every module. The two copies have separate
`Settings` objects, separate model caches, and — the part that bites — separate
`Enum` classes, so `settings.profile is Profile.GPU_COLAB` silently evaluates to
False across the boundary.

That is not hypothetical. It selected the ONNX embedder while holding GPU model
slots, and the visible symptom was `ValueError: Model BAAI/bge-m3 is not
supported in TextEmbedding` on a session that had already printed `gpu-colab`.
Nothing in that message points at the import.

So this fails immediately and says why. To make `te_assistant` importable:

    pip install -e .                      # preferred
    export PYTHONPATH=/path/to/repo/src   # no install needed
"""

from __future__ import annotations

__version__ = "0.1.0"

if __name__ != "te_assistant":  # pragma: no cover - import-time guard
    raise ImportError(
        f"This package was imported as {__name__!r}, but it must be imported as "
        "'te_assistant'.\n"
        "Importing it through the 'src.' prefix loads a second, independent copy "
        "of every module: separate settings, separate model caches, and separate "
        "Enum classes whose identity comparisons then silently fail.\n\n"
        "Fix one of these:\n"
        "    from te_assistant.config import get_settings     # not src.te_assistant\n"
        "    pip install -e .                                 # register the package\n"
        "    sys.path.insert(0, '<repo>/src')                 # or point at src/ directly"
    )
