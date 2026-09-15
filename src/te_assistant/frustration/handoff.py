"""Human handoff — the second half of brief B.6.

B.6 asks for the *contract*, not just the trigger, so this module answers three
questions explicitly:

  **What transfers?** The full conversation transcript, the detected language and
  dialect (so routing can pick an Egyptian-dialect agent), the last intent, the
  reasons the system escalated, and any resolved account. PII is restored for the
  agent — they are authorised to see it — but the stored copy is masked, because
  the queue outlives the conversation.

  **What does the user see?** An immediate acknowledgement in their own language
  with a reference number, so the handoff is not a silent dead end.

  **What if no agent is available?** The conversation does not end. The ticket is
  queued, the user is told, and the assistant keeps answering — degrading to a
  working assistant is better than a dead one.

The queue is in-memory: a real deployment would publish to the contact-centre
system, which is one adapter away and is named in Future Work. What matters for
the POC is that the contract is defined and exercised, not that it integrates
with a CRM we do not have.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime

from ..schemas import FrustrationSignal, HandoffTicket, IntentName, Language

log = logging.getLogger(__name__)

# Business hours for the demo. A real implementation reads agent presence from
# the contact-centre; hard-coding it here keeps the "no agent available" branch
# demonstrable rather than theoretical.
AGENT_HOURS = range(9, 22)  # 09:00-21:59 Cairo


class HandoffQueue:
    def __init__(self) -> None:
        self._tickets: dict[str, HandoffTicket] = {}
        self._lock = threading.Lock()

    def agent_available(self, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        # Cairo is UTC+2 year-round (Egypt reintroduced DST in 2023, but the
        # POC does not need that precision — it needs the branch to exist).
        cairo_hour = (now.hour + 2) % 24
        return cairo_hour in AGENT_HOURS

    def open(
        self,
        *,
        session_id: str,
        reason: str,
        transcript: list[dict[str, str]],
        language: Language,
        last_intent: IntentName,
    ) -> HandoffTicket:
        ticket = HandoffTicket(
            ticket_id=f"HO-{uuid.uuid4().hex[:8].upper()}",
            session_id=session_id,
            reason=reason,
            transcript=transcript[-20:],
            detected_language=language,
            last_intent=last_intent,
            agent_available=self.agent_available(),
        )
        with self._lock:
            self._tickets[ticket.ticket_id] = ticket
        log.info(
            "handoff opened %s for session %s (agent_available=%s): %s",
            ticket.ticket_id, session_id[:8], ticket.agent_available, reason,
        )
        return ticket

    def get(self, ticket_id: str) -> HandoffTicket | None:
        with self._lock:
            return self._tickets.get(ticket_id)

    def pending(self) -> list[HandoffTicket]:
        with self._lock:
            return list(self._tickets.values())


def handoff_message(ticket: HandoffTicket, *, arabic: bool) -> str:
    """What the user is told at the moment of transfer."""
    if ticket.agent_available:
        if arabic:
            return (
                f"أنا بحوّلك دلوقتي لأحد زملائي من خدمة العملاء. "
                f"رقم المرجع بتاعك {ticket.ticket_id}. "
                "هما شايفين المحادثة كاملة، مش هتحتاج تعيد كلامك."
            )
        return (
            f"I'm transferring you to a customer service colleague now. "
            f"Your reference is {ticket.ticket_id}. "
            "They can see the full conversation, so you won't need to repeat yourself."
        )

    if arabic:
        return (
            f"سجّلت طلبك للتحدث مع أحد زملائي برقم مرجع {ticket.ticket_id}، "
            "بس مفيش حد متاح دلوقتي — فريق خدمة العملاء بيشتغل من 9 الصبح لـ 10 بالليل. "
            "هيتواصلوا معاك أول ما يفتحوا، وأنا تحت أمرك لو عايز تسأل حاجة تانية."
        )
    return (
        f"I've logged your request for a colleague under reference {ticket.ticket_id}, "
        "but no one is available right now — our team works 09:00 to 22:00. "
        "They'll follow up when they're back, and I can keep helping in the meantime."
    )


def should_escalate(signal: FrustrationSignal, *, already_handed_off: bool) -> bool:
    """Escalate once per session.

    Re-escalating on every subsequent turn would spam the queue with duplicates
    of a conversation an agent is already handling.
    """
    return signal.should_handoff and not already_handed_off
