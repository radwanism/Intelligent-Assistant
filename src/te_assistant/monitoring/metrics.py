"""Per-session monitoring: latency, WER and quality counters (brief B.10).

Mirrors the monitoring layer described in C.1 — "tracking Word Error Rate and
latency per session" — which is the right shape for this system because both
numbers are per-conversation properties, not per-process ones. A single global
average hides the session where ASR failed repeatedly on one caller's accent.

Deliberately in-process and dependency-free. Prometheus or Langfuse would be the
production answer and are named in Future Work; adding either here would mean a
server to run for a POC whose whole claim is that it installs and runs offline.

`/metrics` on the core service exposes a snapshot, so the numbers used in the
presentation come from the running system rather than a spreadsheet.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

# Bounded so a long-running demo cannot grow memory without limit.
MAX_SAMPLES_PER_STAGE = 2000


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile — see insights.records for why not interpolated."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(pct / 100 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return round(ordered[max(index, 0)], 1)


@dataclass
class StageStats:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def of(cls, samples: list[float]) -> StageStats:
        if not samples:
            return cls()
        return cls(
            count=len(samples),
            mean_ms=round(statistics.mean(samples), 1),
            p50_ms=percentile(samples, 50),
            p95_ms=percentile(samples, 95),
            max_ms=round(max(samples), 1),
        )


@dataclass
class SessionMetrics:
    """What we track for one conversation."""

    turns: int = 0
    voice_turns: int = 0
    asr_confidences: list[float] = field(default_factory=list)
    hallucinations_filtered: int = 0
    clarifications_requested: int = 0
    grounded: int = 0
    refused: int = 0
    blocked: int = 0
    escalated: int = 0
    actions_executed: int = 0
    actions_refused: int = 0
    total_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "voice_turns": self.voice_turns,
            "mean_asr_confidence": (
                round(statistics.mean(self.asr_confidences), 3)
                if self.asr_confidences
                else None
            ),
            "hallucinations_filtered": self.hallucinations_filtered,
            "clarifications_requested": self.clarifications_requested,
            "grounded_rate": round(self.grounded / self.turns, 3) if self.turns else 0.0,
            "refused": self.refused,
            "blocked_by_guardrail": self.blocked,
            "escalated": self.escalated,
            "actions": {
                "executed": self.actions_executed,
                "refused": self.actions_refused,
            },
            "latency": StageStats.of(self.total_ms).__dict__,
        }


class MetricsRegistry:
    """Thread-safe counters. Never raises into the request path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=MAX_SAMPLES_PER_STAGE)
        )
        self._sessions: dict[str, SessionMetrics] = defaultdict(SessionMetrics)
        self._wer_samples: list[tuple[str, float]] = []

    # -- recording ---------------------------------------------------------
    def record_turn(self, record: Any) -> None:
        """Ingest a `TurnRecord`. Typed loosely to avoid a circular import."""
        try:
            with self._lock:
                session = self._sessions[record.session_id]
                session.turns += 1
                if record.input_mode == "voice":
                    session.voice_turns += 1
                    if record.asr_confidence is not None:
                        session.asr_confidences.append(float(record.asr_confidence))

                if record.grounded:
                    session.grounded += 1
                if record.blocked_by_guardrail:
                    session.blocked += 1
                if record.handed_off:
                    session.escalated += 1
                if record.resolution == "refused":
                    session.refused += 1
                if record.action is not None:
                    if record.action.executed:
                        session.actions_executed += 1
                    else:
                        session.actions_refused += 1

                timings = record.timings
                if timings.total_ms:
                    session.total_ms.append(float(timings.total_ms))
                    self._stages["total"].append(float(timings.total_ms))
                for stage in (
                    "asr_ms", "guardrail_ms", "retrieval_ms",
                    "rerank_ms", "generation_ms", "tts_ms",
                ):
                    value = getattr(timings, stage, None)
                    if value:
                        self._stages[stage.removesuffix("_ms")].append(float(value))
        except Exception:
            # Monitoring must never break the turn it is measuring.
            pass

    def record_asr(
        self, session_id: str, *, filtered_hallucination: bool, needed_clarification: bool
    ) -> None:
        with self._lock:
            session = self._sessions[session_id]
            if filtered_hallucination:
                session.hallucinations_filtered += 1
            if needed_clarification:
                session.clarifications_requested += 1

    def record_wer(self, session_id: str, wer: float) -> None:
        """Only meaningful where a reference transcript exists (the eval set)."""
        with self._lock:
            self._wer_samples.append((session_id, float(wer)))

    # -- reading -----------------------------------------------------------
    def session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            metrics = self._sessions.get(session_id)
            return metrics.snapshot() if metrics else SessionMetrics().snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {
                name: StageStats.of(list(samples)).__dict__
                for name, samples in self._stages.items()
            }
            wer_values = [value for _, value in self._wer_samples]
            return {
                "sessions_tracked": len(self._sessions),
                "turns": sum(s.turns for s in self._sessions.values()),
                "voice_turns": sum(s.voice_turns for s in self._sessions.values()),
                "stages_ms": stages,
                "wer": {
                    "n": len(wer_values),
                    "mean": round(statistics.mean(wer_values), 4) if wer_values else None,
                },
                "hallucinations_filtered": sum(
                    s.hallucinations_filtered for s in self._sessions.values()
                ),
                "guardrail_blocks": sum(s.blocked for s in self._sessions.values()),
                "escalations": sum(s.escalated for s in self._sessions.values()),
            }

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self._sessions.clear()
            self._wer_samples.clear()


METRICS = MetricsRegistry()
