"""Document parsing: PDF, DOCX, TXT and images (brief A.2/A.3, B.3).

Design notes that matter for retrieval quality:

**Scanned PDFs are detected, not assumed.** A PDF whose text layer yields almost
nothing per page is a scan, and running it through a text extractor produces an
empty document that indexes cleanly and answers nothing. We check per page and
fall back to OCR only where needed, which keeps the common case fast.

**Tables are preserved as Markdown, not flattened.** A telecom bill or tariff
sheet is mostly tabular. Flattening "Nitro 200 | 300 EGP | 200 GB" into a space
separated run makes the numbers unattachable to their labels, and the model then
quotes the wrong price. This is the single biggest quality difference in
document ingestion.

**Images are OCR'd to text, not embedded as images.** Visual retrieval would
need a second embedding space and a VLM; OCR into the existing text index reuses
the whole retrieval stack. Noted as Future Work in the plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..retrieval.normalize_ar import detect_language
from ..schemas import Language

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Below this many characters, a PDF page is treated as a scan and sent to OCR.
SCAN_THRESHOLD_CHARS = 60


class UnsupportedDocument(ValueError):
    pass


@dataclass
class ParsedPage:
    page: int
    text: str
    used_ocr: bool = False


@dataclass
class ParsedDocument:
    name: str
    pages: list[ParsedPage] = field(default_factory=list)
    language: Language = Language.UNKNOWN

    @property
    def used_ocr(self) -> bool:
        return any(page.used_ocr for page in self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


# --------------------------------------------------------------------------
def parse_document(path: Path, *, ocr_enabled: bool = True) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(f"unsupported file type: {suffix or '(none)'}")

    if suffix == ".pdf":
        document = _parse_pdf(path, ocr_enabled=ocr_enabled)
    elif suffix == ".docx":
        document = _parse_docx(path)
    elif suffix in IMAGE_SUFFIXES:
        document = _parse_image(path, ocr_enabled=ocr_enabled)
    else:
        document = _parse_text(path)

    document.language = detect_language(document.text[:4000])
    return document


def _parse_text(path: Path) -> ParsedDocument:
    # Arabic text files are frequently cp1256 rather than UTF-8; failing the
    # whole upload over an encoding guess would be a poor experience.
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        content = path.read_text(errors="replace")
    return ParsedDocument(name=path.name, pages=[ParsedPage(page=1, text=content)])


def _parse_docx(path: Path) -> ParsedDocument:
    import docx

    document = docx.Document(str(path))
    blocks: list[str] = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]
    for table in document.tables:
        blocks.append(_table_to_markdown(
            [[cell.text.strip() for cell in row.cells] for row in table.rows]
        ))
    # python-docx has no page concept; the whole document is one unit and the
    # chunker splits it on headings instead.
    return ParsedDocument(name=path.name, pages=[ParsedPage(page=1, text="\n\n".join(blocks))])


def _parse_pdf(path: Path, *, ocr_enabled: bool) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[ParsedPage] = []

    for number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            log.warning("text extraction failed on page %d of %s", number, path.name)
            text = ""

        used_ocr = False
        if len(text) < SCAN_THRESHOLD_CHARS and ocr_enabled:
            ocr_text = _ocr_pdf_page(path, number - 1)
            if len(ocr_text) > len(text):
                text, used_ocr = ocr_text, True

        if text.strip():
            pages.append(ParsedPage(page=number, text=text, used_ocr=used_ocr))

    return ParsedDocument(name=path.name, pages=pages)


def _parse_image(path: Path, *, ocr_enabled: bool) -> ParsedDocument:
    if not ocr_enabled:
        raise UnsupportedDocument("OCR is disabled, cannot read an image")
    from .ocr import ocr_image

    text = ocr_image(path)
    if not text.strip():
        # An empty result is reported rather than indexed: silently storing an
        # empty document would leave the user asking about a file the assistant
        # believes it has but cannot see.
        raise UnsupportedDocument("no readable text found in the image")
    return ParsedDocument(name=path.name, pages=[ParsedPage(page=1, text=text, used_ocr=True)])


def _ocr_pdf_page(path: Path, page_index: int) -> str:
    """Rasterise one PDF page and OCR it.

    pypdf cannot render, so we take the embedded images off the page — which is
    exactly what a scanned page consists of. This avoids a Poppler/pdf2image
    system dependency, which would break the pip-only install story.
    """
    try:
        from pypdf import PdfReader

        from .ocr import ocr_pil_image

        reader = PdfReader(str(path))
        page = reader.pages[page_index]
        texts: list[str] = []
        for image_file in page.images:
            try:
                import io

                from PIL import Image

                image = Image.open(io.BytesIO(image_file.data))
                texts.append(ocr_pil_image(image))
            except Exception:
                continue
        return "\n".join(t for t in texts if t.strip())
    except Exception:
        log.debug("OCR fallback failed for page %d of %s", page_index + 1, path.name)
        return ""


def _table_to_markdown(rows: list[list[str]]) -> str:
    """Render a table so label/value pairs survive chunking and embedding."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header, *body = padded
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)
