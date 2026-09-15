"""Action execution — the "backend does the work" half of B.4.

Control flow for any action intent is fixed and has no shortcut:

    guardrails -> PII mask -> LLM detects intent -> PermissionChecker -> here

`execute` refuses to run unless it is handed an already-allowed
`PermissionDecision`. It does not call the checker itself: taking the decision as
a required argument means a caller cannot skip the check by forgetting to make
it, only by actively forging a decision object, which is visible in review.

Handlers receive the resolved `customer_id` from the decision — never the one
the model put in the slots. The model's slot value was a claim; the decision is
the adjudicated fact.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..schemas import ActionOutcome, Intent, IntentName
from ..security.permissions import PermissionDecision
from .db import ActionDB

log = logging.getLogger(__name__)

Handler = Callable[["ActionContext"], dict[str, Any]]


class ActionContext:
    def __init__(
        self, *, db: ActionDB, session_id: str, customer_id: str, slots: dict[str, Any]
    ) -> None:
        self.db = db
        self.session_id = session_id
        self.customer_id = customer_id
        self.slots = slots


def _check_bill_balance(ctx: ActionContext) -> dict[str, Any]:
    customer = ctx.db.customer_by_id(ctx.customer_id)
    if customer is None:
        raise LookupError("account not found")
    balance = float(customer["balance_egp"])
    return {
        "plan": customer["plan"],
        "balance_egp": round(balance, 2),
        "bill_due_date": customer["bill_due_date"],
        # Negative balance means money owed — resolve it here rather than
        # leaving the model to infer a sign convention.
        "status": "due" if balance < 0 else "settled",
        "amount_due_egp": round(abs(balance), 2) if balance < 0 else 0.0,
    }


def _create_support_ticket(ctx: ActionContext) -> dict[str, Any]:
    subject = str(ctx.slots.get("subject") or "Customer support request").strip()[:200]
    body = str(ctx.slots.get("body") or ctx.slots.get("description") or "").strip()[:2000]
    if not body:
        body = subject
    return ctx.db.create_ticket(
        customer_id=ctx.customer_id,
        session_id=ctx.session_id,
        subject=subject,
        body=body,
    )


HANDLERS: dict[IntentName, Handler] = {
    IntentName.CHECK_BILL_BALANCE: _check_bill_balance,
    IntentName.CREATE_SUPPORT_TICKET: _create_support_ticket,
}


class ActionRegistry:
    def __init__(self, db: ActionDB) -> None:
        self.db = db

    def execute(
        self, *, intent: Intent, session_id: str, decision: PermissionDecision
    ) -> ActionOutcome:
        """Run an action, but only on an allowed decision."""
        if not intent.name.is_action:
            return ActionOutcome(executed=False, intent=intent.name, reason="not an action")

        if not decision.allowed:
            self.db.audit(
                session_id=session_id,
                intent=intent.name.value,
                allowed=False,
                reason=decision.reason,
            )
            log.info(
                "action refused: intent=%s session=%s reason=%s",
                intent.name.value, session_id[:8], decision.reason,
            )
            return ActionOutcome(executed=False, intent=intent.name, reason=decision.reason)

        if not decision.customer_id:
            self.db.audit(
                session_id=session_id,
                intent=intent.name.value,
                allowed=False,
                reason="allowed decision carried no customer",
            )
            return ActionOutcome(
                executed=False, intent=intent.name, reason="unresolved account"
            )

        handler = HANDLERS.get(intent.name)
        if handler is None:
            return ActionOutcome(
                executed=False, intent=intent.name, reason="no handler registered"
            )

        ctx = ActionContext(
            db=self.db,
            session_id=session_id,
            customer_id=decision.customer_id,
            slots=intent.slots or {},
        )
        try:
            result = handler(ctx)
        except Exception as exc:
            # A handler failure must not surface as a successful action, and the
            # exception text must not reach the user — it can carry schema detail.
            log.exception("action handler failed: %s", intent.name.value)
            self.db.audit(
                session_id=session_id,
                intent=intent.name.value,
                allowed=True,
                reason=f"handler error: {type(exc).__name__}",
            )
            return ActionOutcome(
                executed=False, intent=intent.name, reason="action could not be completed"
            )

        self.db.audit(session_id=session_id, intent=intent.name.value, allowed=True)
        return ActionOutcome(executed=True, intent=intent.name, result=result)


def describe_result(outcome: ActionOutcome, *, arabic: bool) -> str:
    """Render an action result deterministically.

    Not generated by the model. Account balances and ticket numbers are exactly
    the kind of value that must not pass through a component capable of
    paraphrasing them, so the template is code.
    """
    if not outcome.executed or not outcome.result:
        return ""

    result = outcome.result
    if outcome.intent is IntentName.CHECK_BILL_BALANCE:
        if arabic:
            if result["status"] == "due":
                return (
                    f"باقتك الحالية: {result['plan']}. "
                    f"المستحق عليك {result['amount_due_egp']} جنيه، "
                    f"وآخر موعد للسداد {result['bill_due_date']}."
                )
            return (
                f"باقتك الحالية: {result['plan']}. "
                f"رصيدك {result['balance_egp']} جنيه ومفيش مستحقات."
            )
        if result["status"] == "due":
            return (
                f"Your plan is {result['plan']}. "
                f"You owe EGP {result['amount_due_egp']}, due {result['bill_due_date']}."
            )
        return (
            f"Your plan is {result['plan']}. "
            f"Your balance is EGP {result['balance_egp']} with nothing outstanding."
        )

    if outcome.intent is IntentName.CREATE_SUPPORT_TICKET:
        if arabic:
            return f"تم فتح تذكرة دعم برقم {result['ticket_id']}. هنتواصل معاك قريب."
        return (
            f"Support ticket {result['ticket_id']} has been opened. "
            "Our team will follow up shortly."
        )

    return ""
