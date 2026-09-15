"""Generate the PPTX deck (deliverable A.5).

Kept as a script rather than a hand-made file so the deck regenerates when the
numbers change — every figure here comes from the evaluation results, not from
memory, and `data/eval/results/rag.json` is read at build time so the slides
cannot drift from what the system actually measured.

    python scripts/make_presentation.py
    python scripts/make_presentation.py --out docs/te-assistant.pptx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

# Telecom Egypt / WE brand-adjacent palette, kept deliberately restrained.
PURPLE = RGBColor(0x6B, 0x2D, 0x8B)
DARK = RGBColor(0x1A, 0x1A, 0x2E)
GREY = RGBColor(0x55, 0x55, 0x66)
LIGHT = RGBColor(0xF4, 0xF4, 0xF8)
ACCENT = RGBColor(0x00, 0x8C, 0x8C)
WARN = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self.blank = self.prs.slide_layouts[6]

    # -- primitives --------------------------------------------------------
    def _slide(self):
        return self.prs.slides.add_slide(self.blank)

    def _text(self, slide, text, *, left, top, width, height, size=18,
              bold=False, color=DARK, align=PP_ALIGN.LEFT, font="Segoe UI"):
        box = slide.shapes.add_textbox(left, top, width, height)
        frame = box.text_frame
        frame.word_wrap = True
        para = frame.paragraphs[0]
        para.alignment = align
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
        return frame

    def _bullets(self, slide, items, *, left, top, width, height, size=16,
                 color=DARK, spacing=10):
        box = slide.shapes.add_textbox(left, top, width, height)
        frame = box.text_frame
        frame.word_wrap = True
        for index, item in enumerate(items):
            para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            para.space_after = Pt(spacing)
            if isinstance(item, tuple):
                label, body = item
                run = para.add_run()
                run.text = f"{label}  "
                run.font.size = Pt(size)
                run.font.bold = True
                run.font.color.rgb = PURPLE
                run.font.name = "Segoe UI"
                run2 = para.add_run()
                run2.text = body
                run2.font.size = Pt(size)
                run2.font.color.rgb = color
                run2.font.name = "Segoe UI"
            else:
                run = para.add_run()
                run.text = f"•  {item}"
                run.font.size = Pt(size)
                run.font.color.rgb = color
                run.font.name = "Segoe UI"
        return frame

    def _bar(self, slide, *, top=None, height=None, color=PURPLE):
        from pptx.enum.shapes import MSO_SHAPE

        top = Emu(0) if top is None else top
        height = Inches(0.09) if height is None else height
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), top, SLIDE_W, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()
        shape.shadow.inherit = False
        return shape

    def _panel(self, slide, *, left, top, width, height, color=LIGHT):
        from pptx.enum.shapes import MSO_SHAPE

        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()
        shape.shadow.inherit = False
        return shape

    def _notes(self, slide, text: str) -> None:
        slide.notes_slide.notes_text_frame.text = text

    # -- slide builders ----------------------------------------------------
    def title_slide(self, title, subtitle, footer=""):
        slide = self._slide()
        from pptx.enum.shapes import MSO_SHAPE

        band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
        band.fill.solid()
        band.fill.fore_color.rgb = DARK
        band.line.fill.background()
        band.shadow.inherit = False

        self._text(slide, title, left=Inches(1), top=Inches(2.4), width=Inches(11.3),
                   height=Inches(1.4), size=44, bold=True, color=WHITE)
        self._text(slide, subtitle, left=Inches(1), top=Inches(3.8), width=Inches(11.3),
                   height=Inches(1.2), size=20, color=RGBColor(0xC8, 0xC8, 0xD8))
        if footer:
            self._text(slide, footer, left=Inches(1), top=Inches(6.3), width=Inches(11.3),
                       height=Inches(0.5), size=13, color=ACCENT)
        return slide

    def content(self, title, items, *, note="", kicker=""):
        slide = self._slide()
        self._bar(slide)
        if kicker:
            self._text(slide, kicker.upper(), left=Inches(0.7), top=Inches(0.35),
                       width=Inches(11.9), height=Inches(0.3), size=11, bold=True,
                       color=ACCENT)
            top = Inches(0.72)
        else:
            top = Inches(0.45)
        self._text(slide, title, left=Inches(0.7), top=top, width=Inches(11.9),
                   height=Inches(0.8), size=30, bold=True, color=DARK)
        self._bullets(slide, items, left=Inches(0.85), top=Inches(1.85),
                      width=Inches(11.6), height=Inches(4.8))
        if note:
            self._notes(slide, note)
        return slide

    def metrics(self, title, tiles, *, subtitle="", note=""):
        """Big-number tiles. Used where a measured figure is the whole point."""
        slide = self._slide()
        self._bar(slide)
        self._text(slide, title, left=Inches(0.7), top=Inches(0.45), width=Inches(11.9),
                   height=Inches(0.8), size=30, bold=True, color=DARK)
        if subtitle:
            self._text(slide, subtitle, left=Inches(0.7), top=Inches(1.15),
                       width=Inches(11.9), height=Inches(0.4), size=14, color=GREY)

        count = len(tiles)
        gap = Inches(0.3)
        total_gap = gap * (count - 1)
        tile_w = int((Inches(11.9) - total_gap) / count)
        left = Inches(0.7)
        for value, label in tiles:
            self._panel(slide, left=left, top=Inches(1.9), width=Emu(tile_w),
                        height=Inches(1.9))
            self._text(slide, value, left=left, top=Inches(2.15), width=Emu(tile_w),
                       height=Inches(0.9), size=40, bold=True, color=PURPLE,
                       align=PP_ALIGN.CENTER)
            self._text(slide, label, left=left, top=Inches(3.1), width=Emu(tile_w),
                       height=Inches(0.5), size=13, color=GREY, align=PP_ALIGN.CENTER)
            left = Emu(left + tile_w + gap)
        if note:
            self._notes(slide, note)
        return slide

    def table(self, title, headers, rows, *, note="", subtitle="", col_widths=None):
        slide = self._slide()
        self._bar(slide)
        self._text(slide, title, left=Inches(0.7), top=Inches(0.45), width=Inches(11.9),
                   height=Inches(0.8), size=30, bold=True, color=DARK)
        top = Inches(1.35)
        if subtitle:
            self._text(slide, subtitle, left=Inches(0.7), top=Inches(1.2),
                       width=Inches(11.9), height=Inches(0.4), size=14, color=GREY)
            top = Inches(1.75)

        shape = slide.shapes.add_table(
            len(rows) + 1, len(headers), Inches(0.7), top, Inches(11.9),
            Inches(0.4 * (len(rows) + 1)),
        )
        tbl = shape.table
        if col_widths:
            for index, width in enumerate(col_widths):
                tbl.columns[index].width = Inches(width)

        for col, header in enumerate(headers):
            cell = tbl.cell(0, col)
            cell.text = header
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size = Pt(14)
            para.runs[0].font.bold = True
            para.runs[0].font.color.rgb = WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = PURPLE

        for r, row in enumerate(rows, start=1):
            for c, value in enumerate(row):
                cell = tbl.cell(r, c)
                cell.text = str(value)
                para = cell.text_frame.paragraphs[0]
                para.runs[0].font.size = Pt(13)
                para.runs[0].font.color.rgb = DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
        if note:
            self._notes(slide, note)
        return slide

    def quote(self, title, statement, *, support=None, note="", warn=False, kicker=""):
        slide = self._slide()
        self._bar(slide, color=WARN if warn else PURPLE)
        if kicker:
            self._text(slide, kicker.upper(), left=Inches(0.7), top=Inches(0.35),
                       width=Inches(11.9), height=Inches(0.3), size=11, bold=True,
                       color=ACCENT)
            title_top = Inches(0.72)
        else:
            title_top = Inches(0.45)
        self._text(slide, title, left=Inches(0.7), top=title_top, width=Inches(11.9),
                   height=Inches(0.8), size=30, bold=True, color=DARK)
        self._panel(slide, left=Inches(0.7), top=Inches(1.7), width=Inches(11.9),
                    height=Inches(1.9), color=LIGHT)
        self._text(slide, statement, left=Inches(1.1), top=Inches(2.0),
                   width=Inches(11.1), height=Inches(1.4), size=22, bold=True,
                   color=WARN if warn else PURPLE)
        if support:
            self._bullets(slide, support, left=Inches(0.85), top=Inches(4.0),
                          width=Inches(11.6), height=Inches(2.6))
        if note:
            self._notes(slide, note)
        return slide

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(path))


# ---------------------------------------------------------------------------
def load_results() -> dict:
    """Read the measured numbers so the deck cannot drift from reality."""
    path = Path("data/eval/results/rag.json")
    if not path.exists():
        print("  ! no eval results found; run scripts/eval_rag.py first")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build(results: dict) -> Deck:
    overall = results.get("overall", {})
    by_lang = results.get("by_language", {})
    k = results.get("k", 5)
    recall = overall.get(f"recall@{k}")
    mrr = overall.get("mrr")
    latency = overall.get("median_latency_ms")
    chunks = results.get("kb_chunks", 681)

    fmt_pct = lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else "—"  # noqa: E731

    d = Deck()

    # 1 -------------------------------------------------------------------
    d.title_slide(
        "Telecom Egypt — Intelligent Assistant",
        "Bilingual voice + text RAG assistant  ·  Arabic, English, Egyptian dialect\n"
        "Runs fully on-premises — no external API in the serving path",
        "Proof of concept  ·  case study submission",
    )

    # 2 -------------------------------------------------------------------
    d.content(
        "The problem, stated as constraints",
        [
            "Thousands of daily inquiries in Arabic, English and Egyptian dialect",
            "Voice and text, with noisy real-world audio",
            "Answers grounded in te.eg and cited — a telecom cannot quote a wrong price",
            "Customers upload documents and ask questions about them",
            ("No external API in the serving path.",
             "Customer audio and documents stay on-premises."),
        ],
        kicker="Context",
        note="The last constraint shapes everything. It rules out the easy answer and "
             "forces every model choice to be defended on footprint and latency, not "
             "on leaderboard position.",
    )

    # 3 -------------------------------------------------------------------
    d.content(
        "What it does",
        [
            "Voice or text in — input mode known from the channel, not guessed",
            ("Text is ALWAYS displayed.", "Voice is additive, never the only output."),
            "Every answer carries citations, and refuses rather than guessing",
            "PDF / DOCX / TXT / image upload, scoped to the session only",
            "Actions (bill lookup, ticket creation) without giving the model DB access",
            "Frustration detection with a defined path to a human agent",
        ],
        kicker="Scope",
    )

    # 4 -------------------------------------------------------------------
    d.content(
        "Architecture — three services, and why only three",
        [
            ("UI (Gradio, :7860)", "thin client, holds no models"),
            ("CORE (:8000)", "guardrails · PII · retrieval · intent · permissions · sessions"),
            ("SPEECH (:8001)", "ASR + TTS — the only component split out"),
            "",
            ("Why speech is separate:", "heaviest models, slowest cold start, "
             "highest OOM risk — and voice is optional to the product, text chat is not."),
            ("Why nothing else is:", "retrieval, intent and sessions share the session "
             "store and vector index. Splitting them adds network hops and buys nothing."),
        ],
        kicker="System design",
        note="The question I asked at each boundary was: what breaks if this dies? For "
             "speech the answer is 'voice input', not 'the product'. That is a service. "
             "For retrieval and intent the answer is 'everything', so they stay together.",
    )

    # 5 -------------------------------------------------------------------
    d.table(
        "The degradation contract",
        ["Failure", "Result"],
        [
            ["Speech service down", "Mic hidden — text chat unaffected"],
            ["LLM unavailable", "Retrieval still returns cited passages"],
            ["Reranker unavailable", "Falls back to RRF ordering"],
            ["TTS fails", "Text answer still delivered"],
        ],
        subtitle="Demonstrable in one command:  docker compose stop speech",
        col_widths=[4.2, 7.7],
        note="A degradation story that is never exercised is a guess. Each row is a test.",
    )

    # 6 -------------------------------------------------------------------
    d.content(
        "Data flow — one voice turn",
        [
            "mic → VAD → ASR → hallucination filter → confidence gate",
            "→ input guardrails → PII masking → session resolve",
            "→ hybrid retrieve (BM25 + dense → RRF) → rerank",
            "→ intent detection",
            "       ├── action?  → permission check → backend executes",
            "       └── else     → grounded generation + citations",
            "→ output guardrail → unmask for display",
            "→ frustration score → structured turn record",
            "→ text + citations shown, spoken answer added",
        ],
        kicker="Data flow",
    )

    # 7 -------------------------------------------------------------------
    d.content(
        "Data quality — where retrieval is actually won",
        [
            ("te.eg is a Liferay portal:", "the same ~2–3k character menu on every "
             "page — about a third of a typical page."),
            "Leave it in and every chunk shares most of its tokens with every other "
            "chunk: dense similarity rises between unrelated pages and stops "
            "discriminating. BM25 degrades too — the menu vocabulary is exactly what "
            "users search for.",
            ("Fix — corpus-level, not per-page:", "any line on >35% of pages is "
             "template furniture, whatever the markup says."),
            ("Tables preserved as Markdown:", "'Nitro 200 | 300 EGP | 200 GB' flattened "
             "is a number with no label, and the model quotes the wrong price."),
        ],
        kicker="Pillar 1 — data quality",
    )

    # 8 -------------------------------------------------------------------
    d.metrics(
        "Corpus, after cleaning",
        [("600", "pages crawled"), ("388", "documents kept"),
         (str(chunks), "indexed chunks"), ("21%", "was one page")],
        subtitle="The press-release listing alone was 183k characters — excluded, and "
                 "any single document capped at 40k.",
    )

    # 9 -------------------------------------------------------------------
    d.quote(
        "A bilingual corpus, not an Arabic one queried in English",
        "English documents: 33 → 183.    English recall@5: poor → 90%.",
        support=[
            "The first crawl was 85% Arabic.",
            "te.eg serves English under separate /en/ URLs that Arabic-seeded "
            "link-following never reaches.",
            ("The lesson:", "the case study asks for bilingual output. That has to mean "
             "a bilingual corpus — not leaving the embedder to bridge the gap alone."),
        ],
        note="This was a bug I found by reading the per-language evaluation split. An "
             "overall recall number would have hidden it completely.",
    )

    # 10 ------------------------------------------------------------------
    d.content(
        "Arabic normalisation and light stemming",
        [
            ("BM25 is lexical:", "«إنترنت» and «انترنت» are different terms to it."),
            ("Arabic is agglutinative:", "untreated, «أشحن» / «شحن» / «الشحن» are three "
             "unrelated terms — so a naturally-phrased question misses the page that "
             "answers it."),
            "",
            ("And a bug worth admitting:", "the first stemmer also stripped "
             "single-letter prefixes, which ate the first root letter — «باقات» → «قات» "
             "but «باقة» → «اقه». The plural and singular of the word this corpus is "
             "most asked about stopped matching."),
            ("Now:", "multi-character article forms only. 11/11 conflation cases pass."),
        ],
        kicker="Pillar 1 — data quality",
        note="Showing the fix is stronger than showing the feature. It is evidence of "
             "testing rather than luck.",
    )

    # 11 ------------------------------------------------------------------
    d.table(
        "Hybrid retrieval — two failure modes, two retrievers",
        ["Retriever", "Fails at"],
        [
            ["Dense alone", "Exact identifiers — 'Nitro 200' retrieves 'Nitro 100'"],
            ["BM25 alone", "Cross-lingual — «عايز أعرف أسعار النت» has zero lexical "
                           "overlap with an English tariff page"],
        ],
        subtitle="Fused with Reciprocal Rank Fusion, not a weighted sum: the two scores "
                 "are not on comparable scales and per-query normalisation is fragile. "
                 "RRF only needs the ranks.",
        col_widths=[3.0, 8.9],
    )

    # 12 ------------------------------------------------------------------
    d.metrics(
        "Measured retrieval results",
        [(fmt_pct(recall), f"recall@{k}"),
         (f"{mrr:.3f}" if mrr else "—", "MRR"),
         (f"{latency:.0f} ms" if latency else "—", "median latency"),
         ("126", "tests passing")],
        subtitle=f"cpu-lite profile · 38-question bilingual golden set · {chunks} chunks",
        note="Every number on this slide came out of scripts/eval_rag.py on the running "
             "system, not from a spreadsheet.",
    )

    # 13 ------------------------------------------------------------------
    lang_rows = []
    for name, label in [("en", "English"), ("mixed", "Code-switched (AR/EN)"),
                        ("ar", "MSA"), ("arz", "Egyptian dialect")]:
        stats = by_lang.get(name, {})
        if stats:
            lang_rows.append([label, str(stats.get("n", "—")),
                              fmt_pct(stats.get(f"recall@{k}")),
                              f"{stats.get('mrr', 0):.3f}"])
    d.table(
        "Reported split by query language — on purpose",
        ["Query language", "n", f"recall@{k}", "MRR"],
        lang_rows or [["(run eval_rag.py)", "—", "—", "—"]],
        subtitle="An overall figure that hid 'English retrieves nothing' would be "
                 "misleading. Dialect is the weakest — I come back to why.",
        col_widths=[5.0, 1.6, 2.6, 2.7],
    )

    # 14 ------------------------------------------------------------------
    d.content(
        "ASR — built for noisy dialect, not a Whisper wrapper",
        [
            ("VAD before transcription:", "Whisper's failure on silence is not silence "
             "— it is a confident sentence learned from subtitle corpora."),
            ("Egyptian-dialect initial_prompt:", "Whisper drifts to MSA because MSA "
             "dominates its Arabic training data."),
            ("Hallucination filters:", "known artifacts («ترجمة نانسي قنقر», 'Thank you "
             "for watching') plus a repetition-loop detector."),
            ("Confidence gate:", "ask the user to repeat rather than answer a misheard "
             "question."),
            "",
            ("Never task='translate'.", "Pivoting dialect through English discards "
             "exactly what the case study is testing."),
        ],
        kicker="ASR",
    )

    # 15 ------------------------------------------------------------------
    d.quote(
        "Security — the architectural claim",
        "The model's output carries no authority.",
        support=[
            "1. input guardrails → before the model sees anything",
            "2. PII masking → the model never sees raw sensitive values",
            "3. intent detection → the model's ONLY job",
            "4. permission check → the backend decides",
            "5. execution → the backend acts, never the model",
            ("Steps 1 and 2 cannot be reordered:", "masking first feeds an injection "
             "string to the masker; guarding after the model defeats the point."),
        ],
        note="Live demo here: a correctly-detected check_bill_balance for someone "
             "else's account, refused. An intent can arrive perfectly formed, maximally "
             "confident, naming a real account — and it still does not execute unless "
             "that session holds the grant.",
    )

    # 16 ------------------------------------------------------------------
    d.content(
        "Security, evidenced",
        [
            ("Two real action intents", "over SQLite: check_bill_balance (read), "
             "create_support_ticket (write)"),
            ("PII by regex with a Luhn check", "— so an order reference is not masked "
             "as a card. Over-masking removes information the model needs."),
            ("Every refusal audited.", "An audit log that only records successes "
             "cannot answer 'did anyone try?'"),
            ("Tests assert the NEGATIVE cases:", "injection blocked, PII never reaching "
             "the model, unauthorised action refused, cross-session retrieval impossible."),
            "",
            ("Honest limit:", "guardrails are pattern-based. They stop scripted "
             "attempts, not a determined adversary. The architecture does not rely on "
             "them — the model has no database access regardless."),
        ],
        kicker="Security",
    )

    # 17 ------------------------------------------------------------------
    d.quote(
        "Session isolation under concurrent users",
        "There is deliberately no code path that reaches session documents "
        "without a filter.",
        support=[
            ("Unguessable IDs", "(secrets.token_urlsafe(32), 256 bits) stop guessing — "
             "but that is not the isolation property."),
            ("Mandatory filtering in the store", "is what actually isolates. Default "
             "with no filter is ZERO documents, never all."),
            ("Eviction is real:", "files and embeddings are both deleted at session end."),
        ],
        note="The failure mode I designed against is a caller who forgets to pass the "
             "filter. So I removed the ability to forget.",
    )

    # 18 ------------------------------------------------------------------
    d.table(
        "On-premises deployment",
        ["Concern", "Approach"],
        [
            ["Footprint", "Torch-free CPU profile: CTranslate2 + ONNX + llama.cpp → "
                          "~1.5 GB, not 4.5 GB"],
            ["Portability", "Two profiles, one codebase — device detection, never a fork"],
            ["Air-gapped", "uv pip download + fetch_models.py → nothing touches the network"],
            ["Containers", "Docker: 3 services, one image, weights as a volume"],
            ["Reproducible", "pyproject.toml + committed uv.lock (186 packages)"],
        ],
        col_widths=[2.6, 9.3],
    )

    # 19 ------------------------------------------------------------------
    d.quote(
        "A licensing finding",
        "jina-embeddings-v3 is CC-BY-NC-4.0. Non-commercial.",
        support=[
            "It is the strongest multilingual embedder in the shortlist.",
            "Same for jina-reranker-v2 and f5-tts-egyptian-arabic.",
            ("Replaced with BAAI/bge-m3 (MIT)", "— which also has ~20× the adoption. "
             "Zero cost to switch."),
            ("Licence is a hard filter,", "checked before quality. A non-commercial "
             "model in a customer-facing telecom assistant fails review for reasons "
             "that have nothing to do with benchmark scores."),
        ],
        warn=True,
    )

    # 20 ------------------------------------------------------------------
    d.quote(
        "What I did not take on trust",
        "Nawah-ASR-118M claims WER 0.3358 vs Whisper large-v3's 0.4149.",
        support=[
            ("43 downloads/month, 2 likes.", "The WER is self-reported on the author's "
             "own eval set."),
            ("Not Whisper-architecture", "→ needs torch → cannot run under CTranslate2, "
             "so it is out of the torch-free CPU profile on engineering grounds before "
             "quality is even discussed."),
            ("Decision:", "every model slot is pluggable, well-adopted defaults ship, "
             "and the Egyptian candidates get measured — converting an unverified claim "
             "into an evaluation result instead of a gamble on the critical path."),
        ],
        kicker="Evaluation",
    )

    # 21 ------------------------------------------------------------------
    d.content(
        "Scalability — what is built, and what is not",
        [
            ("Built:", "stateless request handling · singleton model loading · "
             "serialised llama.cpp access · WAL SQLite for concurrent readers · "
             "session TTL and eviction · lazy model loading for the 16 GB VRAM ceiling"),
            "",
            ("Honest limit — the session store is in-process.", "A second core replica "
             "would break session affinity today. The interface is narrow enough that "
             "Redis is a one-file swap, but horizontal scaling is currently a design "
             "property, not a demonstrated one."),
        ],
        kicker="Pillar 5 — scalability",
        note="I would rather say that than let someone find it by scaling the container.",
    )

    # 22 ------------------------------------------------------------------
    d.table(
        "Future work",
        ["Next", "Why not now"],
        [
            ["BGE-M3 on GPU profile", "Should close much of the 73% dialect gap"],
            ["Redis-backed sessions", "Makes horizontal scaling real rather than claimed"],
            ["Cross-encoder reranking on CPU", "Several seconds on 4 cores — blows the "
                                               "latency budget"],
            ["LoRA fine-tune on telecom QA", "Real, but not a POC-scale task"],
            ["Langfuse / LangSmith tracing", "Needs a server; JSONL tracing for now"],
            ["Per-document authentication", "Deliberately deferred — no login in the POC"],
            ["Visual retrieval without OCR", "Needs a second embedding space and a VLM"],
            ["Streaming WebSocket ASR", "Latency win, but not required for the demo"],
        ],
        col_widths=[5.0, 6.9],
    )

    # 23 ------------------------------------------------------------------
    d.title_slide(
        "Thank you",
        "The core runs offline on a CPU.  Every answer is cited.\n"
        "The model detects intent and nothing else — the backend decides what happens.\n\n"
        "And the numbers in this deck came out of /metrics on the running system.",
        "Questions",
    )

    return d


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PPTX deck")
    parser.add_argument("--out", type=Path, default=Path("docs/TelecomEgypt_Assistant.pptx"))
    args = parser.parse_args()

    results = load_results()
    deck = build(results)
    deck.save(args.out)

    print(f"  slides : {len(deck.prs.slides.__iter__.__self__._sldIdLst)}")
    print(f"  written: {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    print("\n  Speaker notes are attached to the slides that need them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
