"""OCR via RapidOCR (ONNX Runtime).

Why RapidOCR and not Tesseract: Tesseract is a system binary. Requiring it would
break the "pip install and run" promise that makes the on-premises deliverable
credible, and it is not present on the development machine. RapidOCR ships its
PP-OCR models as ONNX inside the wheel, so `uv pip install` is the whole setup
and the same code path works on CPU and GPU.

Why not PaddleOCR-VL: it is the stronger document-understanding model on paper,
but its Arabic support is unconfirmed — the only evidence found was an open
question on the model's discussion board, not a documented capability. Putting
an unverified model on the critical path of a 12-hour build is how demos fail,
so RapidOCR is the default and PaddleOCR-VL is a benchmark candidate in
`scripts/bench_models.py`.

Arabic OCR is genuinely hard — cursive, context-dependent letterforms, and RTL
line ordering. Results on Arabic scans will be mediocre, and the evaluation
reports that honestly rather than hiding it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Below this confidence a detected box is more likely noise than text. Arabic
# recognition scores lower than Latin on the same image quality, so this is
# deliberately permissive — dropping real text hurts more than a little noise.
MIN_BOX_CONFIDENCE = 0.35

_ENGINE: Any | None = None
_LOCK = threading.Lock()


def _get_engine() -> Any:
    """Load the OCR engine once, lazily.

    Lazily because most uploads are text PDFs that never need OCR, and on the
    GPU profile this is one of the two slots that unloads to stay inside the
    T4's 16 GB (see config.ModelSlots.lazy_slots).
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _LOCK:
        if _ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR

            _ENGINE = RapidOCR()
            log.info("RapidOCR engine loaded")
    return _ENGINE


def release_engine() -> None:
    """Drop the engine to reclaim memory (GPU profile, lazy slots)."""
    global _ENGINE
    with _LOCK:
        _ENGINE = None


def _run(image: Any) -> list[tuple[list[Any], str, float]]:
    engine = _get_engine()
    result, _elapsed = engine(image)
    return result or []


def _order_lines(boxes: list[tuple[list[Any], str, float]]) -> str:
    """Reassemble detected boxes into reading order.

    RapidOCR returns boxes in detection order, which is not reading order. We
    group by vertical position into lines, then sort within each line by x.

    The RTL subtlety: Arabic *strings* are stored in logical order and rendered
    right-to-left by the display layer, so sorting boxes left-to-right by x and
    letting the renderer handle direction is correct. Reversing here would
    double-flip and produce scrambled text.
    """
    if not boxes:
        return ""

    entries: list[tuple[float, float, str]] = []
    heights: list[float] = []
    for box, text, confidence in boxes:
        if not text or confidence < MIN_BOX_CONFIDENCE:
            continue
        ys = [point[1] for point in box]
        xs = [point[0] for point in box]
        entries.append((sum(ys) / len(ys), min(xs), text.strip()))
        heights.append(max(ys) - min(ys))

    if not entries:
        return ""

    # Line tolerance scales with text size, so it works on both a phone photo
    # and a 300 dpi scan without tuning.
    median_height = sorted(heights)[len(heights) // 2] if heights else 12.0
    tolerance = max(median_height * 0.6, 6.0)

    entries.sort(key=lambda e: e[0])
    lines: list[list[tuple[float, float, str]]] = [[entries[0]]]
    for entry in entries[1:]:
        if abs(entry[0] - lines[-1][-1][0]) <= tolerance:
            lines[-1].append(entry)
        else:
            lines.append([entry])

    out: list[str] = []
    for line in lines:
        line.sort(key=lambda e: e[1])
        joined = " ".join(part for _, _, part in line if part)
        if joined.strip():
            out.append(joined)
    return "\n".join(out)


def ocr_image(path: Path) -> str:
    try:
        return _order_lines(_run(str(path)))
    except Exception:
        log.exception("OCR failed for %s", path.name)
        return ""


def ocr_pil_image(image: Any) -> str:
    """OCR an already-loaded PIL image (used for embedded PDF page scans)."""
    try:
        import numpy as np

        if image.mode != "RGB":
            image = image.convert("RGB")
        return _order_lines(_run(np.array(image)))
    except Exception:
        log.exception("OCR failed for embedded image")
        return ""
