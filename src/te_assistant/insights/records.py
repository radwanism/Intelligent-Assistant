"""Structured turn records and session insights — brief B.11.

A.1 promises the assistant turns customer interactions into "structured outputs
and actionable insights". Without this module that promise is invisible behind
the chat window, so every turn emits a `TurnRecord` and the UI surfaces an
aggregate view over them.

This is close to free: the intent and its slots already exist for B.4, the
frustration score already exists for B.6, and the citations already exist for
A.2. The only new work is writing them down in one shape and counting them.

Records are persisted as JSONL and **PII is masked before writing** — a trace
file that outlives the session must not become where customer phone numbers
end up (see `security/pii.scrub_for_logs`).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import TurnRecord

log = logging.getLogger(__name__)


class RecordStore:
    """Append-only turn log, in memory plus JSONL on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[TurnRecord] = []
        self._lock = threading.Lock()

    def add(self, record: TurnRecord) -> None:
        with self._lock:
            self._records.append(record)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")
            except Exception:
                # Telemetry must never break the request it is describing.
                log.exception("failed to persist turn record")

    def for_session(self, session_id: str) -> list[TurnRecord]:
        with self._lock:
            return [r for r in self._records if r.session_id == session_id]

    def all(self) -> list[TurnRecord]:
        with self._lock:
            return list(self._records)


@dataclass
class Insights:
    """Aggregate view. Shown in the UI and used in the presentation."""

    turns: int = 0
    voice_turns: int = 0
    text_turns: int = 0
    grounded_rate: float = 0.0
    escalation_rate: float = 0.0
    blocked_turns: int = 0
    actions_executed: int = 0
    actions_refused: int = 0
    top_intents: list[tuple[str, int]] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)
    unanswered: list[str] = field(default_factory=list)
    avg_total_ms: float = 0.0
    p95_total_ms: float = 0.0
    avg_stage_ms: dict[str, float] = field(default_factory=dict)


STAGES = ("asr_ms", "guardrail_ms", "retrieval_ms", "rerank_ms", "generation_ms", "tts_ms")


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank rather than interpolated because a POC's sample sizes are small
    and an interpolated p95 over nine turns invents precision that is not there.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(pct / 100 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return round(ordered[max(index, 0)], 1)


def summarize(records: list[TurnRecord]) -> Insights:
    if not records:
        return Insights()

    total = len(records)
    intents = Counter(r.intent.value for r in records)
    languages = Counter(r.language.value for r in records)
    totals = [r.timings.total_ms for r in records if r.timings.total_ms > 0]

    stage_means: dict[str, float] = {}
    for stage in STAGES:
        values = [
            getattr(r.timings, stage)
            for r in records
            if getattr(r.timings, stage) is not None
        ]
        if values:
            stage_means[stage.removesuffix("_ms")] = round(sum(values) / len(values), 1)

    # "Unanswered" is the actionable list: questions the knowledge base could not
    # cover. This is what a content team would actually use.
    unanswered = [
        r.question
        for r in records
        if r.resolution in ("no_answer", "refused") and r.question
    ]

    return Insights(
        turns=total,
        voice_turns=sum(1 for r in records if r.input_mode == "voice"),
        text_turns=sum(1 for r in records if r.input_mode == "text"),
        grounded_rate=round(sum(1 for r in records if r.grounded) / total, 3),
        escalation_rate=round(sum(1 for r in records if r.handed_off) / total, 3),
        blocked_turns=sum(1 for r in records if r.blocked_by_guardrail),
        actions_executed=sum(
            1 for r in records if r.action is not None and r.action.executed
        ),
        actions_refused=sum(
            1 for r in records if r.action is not None and not r.action.executed
        ),
        top_intents=intents.most_common(5),
        languages=dict(languages),
        unanswered=unanswered[-10:],
        avg_total_ms=round(sum(totals) / len(totals), 1) if totals else 0.0,
        p95_total_ms=_percentile(totals, 95),
        avg_stage_ms=stage_means,
    )


def render_markdown(insights: Insights) -> str:
    """Human-readable summary for the UI's insights tab."""
    if insights.turns == 0:
        return "_No turns yet in this session._"

    lines = [
        f"**Turns:** {insights.turns} "
        f"({insights.voice_turns} voice, {insights.text_turns} text)",
        f"**Grounded answers:** {insights.grounded_rate:.0%}",
        f"**Escalated to a human:** {insights.escalation_rate:.0%}",
        f"**Blocked by guardrails:** {insights.blocked_turns}",
        f"**Actions:** {insights.actions_executed} executed, "
        f"{insights.actions_refused} refused",
        f"**Latency:** avg {insights.avg_total_ms:.0f} ms, "
        f"p95 {insights.p95_total_ms:.0f} ms",
    ]

    if insights.avg_stage_ms:
        breakdown = ", ".join(
            f"{stage} {ms:.0f} ms" for stage, ms in insights.avg_stage_ms.items()
        )
        lines.append(f"**Per stage (avg):** {breakdown}")

    if insights.top_intents:
        intents = ", ".join(f"{name} ({count})" for name, count in insights.top_intents)
        lines.append(f"**Top intents:** {intents}")

    if insights.languages:
        langs = ", ".join(f"{lang} ({n})" for lang, n in insights.languages.items())
        lines.append(f"**Languages:** {langs}")

    if insights.unanswered:
        lines.append("\n**Questions the knowledge base could not answer:**")
        lines.extend(f"- {question}" for question in insights.unanswered)

    return "\n\n".join(lines)


def to_jsonl(records: list[TurnRecord]) -> str:
    return "\n".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) for r in records)
