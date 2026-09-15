"""Permission checks — step 4 of the B.4 chain.

The architectural claim being enforced here is narrow and worth stating plainly:

    The LLM decides *what the user appears to want*.
    This module decides *whether they are allowed to have it*.

Those are different questions, answered by different components, and the second
one never consults the model. An intent arriving with `confidence=0.99` and
`customer_id="cust-1001"` carries no authority whatsoever — the model can be
talked into emitting anything, and that is fine, because emitting is not doing.

The demo makes this visible: a prompt that successfully steers intent detection
toward `create_support_ticket` for someone else's account still gets refused
here, and the refusal is recorded in the audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..actions.db import ActionDB
from ..schemas import IntentName

log = logging.getLogger(__name__)

# Which permission scope each action intent requires.
REQUIRED_SCOPE: dict[IntentName, str] = {
    IntentName.CHECK_BILL_BALANCE: "billing:read",
    IntentName.CREATE_SUPPORT_TICKET: "tickets:write",
}


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str = ""
    customer_id: str | None = None

    @classmethod
    def deny(cls, reason: str) -> PermissionDecision:
        return cls(allowed=False, reason=reason)


class PermissionChecker:
    def __init__(self, db: ActionDB) -> None:
        self.db = db

    def check(
        self, *, session_id: str, intent: IntentName, slots: dict[str, object]
    ) -> PermissionDecision:
        """Decide whether this session may perform this action."""
        if not intent.is_action:
            return PermissionDecision(allowed=True)

        if not session_id:
            return PermissionDecision.deny("no session")

        scope = REQUIRED_SCOPE.get(intent)
        if scope is None:
            # An action intent with no declared scope is a programming error, and
            # the safe reading of an unknown action is "not permitted".
            return PermissionDecision.deny(f"no scope defined for {intent.value}")

        customer_id = self._resolve_customer(session_id, slots)
        if customer_id is None:
            return PermissionDecision.deny(
                "could not resolve the account this request refers to"
            )

        if not self.db.has_permission(session_id, customer_id, scope):
            # Deliberately the same message whether the grant is missing or the
            # customer belongs to someone else: a differentiated error would let
            # a caller probe which accounts exist.
            return PermissionDecision(
                allowed=False,
                reason=f"session is not authorised for {scope} on this account",
                customer_id=customer_id,
            )

        return PermissionDecision(allowed=True, customer_id=customer_id)

    def _resolve_customer(
        self, session_id: str, slots: dict[str, object]
    ) -> str | None:
        """Work out which account is meant — without trusting the model.

        A `customer_id` in the slots is treated as a *claim*, never as proof: it
        is only accepted if this session already holds a grant for it. That is
        what stops "check the balance for cust-1002" from working just because
        the model dutifully filled in the slot.
        """
        granted = set(self.db.customers_for_session(session_id))

        claimed = slots.get("customer_id")
        if isinstance(claimed, str) and claimed:
            return claimed if claimed in granted else None

        msisdn = slots.get("msisdn")
        if isinstance(msisdn, str) and msisdn:
            customer = self.db.customer_by_msisdn(msisdn.strip())
            if customer and customer["customer_id"] in granted:
                return str(customer["customer_id"])
            return None

        # No explicit target: fall back to the session's own account, but only
        # when there is exactly one, so "my bill" is never ambiguous.
        if len(granted) == 1:
            return next(iter(granted))
        return None
