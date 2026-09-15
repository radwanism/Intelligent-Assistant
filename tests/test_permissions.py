"""Permission and action tests — the core claim of brief B.4.

The property under test: **the model's output carries no authority.** An intent
can arrive perfectly formed, maximally confident, and naming a real account, and
it still does not execute unless this session holds the grant.
"""

from __future__ import annotations

import pytest

from te_assistant.actions.db import ActionDB
from te_assistant.actions.registry import ActionRegistry, describe_result
from te_assistant.schemas import Intent, IntentName
from te_assistant.security.permissions import PermissionChecker


@pytest.fixture()
def db(tmp_path):
    database = ActionDB(tmp_path / "test.db")
    database.seed()
    return database


@pytest.fixture()
def checker(db):
    return PermissionChecker(db)


@pytest.fixture()
def registry(db):
    return ActionRegistry(db)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
def test_action_refused_without_any_grant(checker) -> None:
    decision = checker.check(
        session_id="session-a", intent=IntentName.CHECK_BILL_BALANCE, slots={}
    )
    assert not decision.allowed


def test_confident_intent_does_not_grant_authority(db, checker, registry) -> None:
    """A maximally confident intent naming a real account is still refused."""
    intent = Intent(
        name=IntentName.CHECK_BILL_BALANCE,
        confidence=1.0,
        slots={"customer_id": "cust-1001"},
    )
    decision = checker.check(
        session_id="attacker", intent=intent.name, slots=intent.slots
    )
    assert not decision.allowed

    outcome = registry.execute(
        intent=intent, session_id="attacker", decision=decision
    )
    assert not outcome.executed
    assert outcome.result is None


def test_cannot_access_another_customers_account(db, checker) -> None:
    """Session is granted cust-1001; asking for cust-1002 must fail."""
    db.grant("session-a", "cust-1001", "billing:read")
    decision = checker.check(
        session_id="session-a",
        intent=IntentName.CHECK_BILL_BALANCE,
        slots={"customer_id": "cust-1002"},
    )
    assert not decision.allowed


def test_msisdn_lookup_respects_grants(db, checker) -> None:
    """Knowing someone's phone number is not authorisation to read their bill."""
    db.grant("session-a", "cust-1001", "billing:read")
    decision = checker.check(
        session_id="session-a",
        intent=IntentName.CHECK_BILL_BALANCE,
        slots={"msisdn": "01198765432"},  # belongs to cust-1002
    )
    assert not decision.allowed


def test_scopes_do_not_leak_across_actions(db, checker) -> None:
    """A read grant must not authorise a write."""
    db.grant("session-a", "cust-1001", "billing:read")
    write = checker.check(
        session_id="session-a", intent=IntentName.CREATE_SUPPORT_TICKET, slots={}
    )
    assert not write.allowed


def test_ambiguous_target_is_refused(db, checker) -> None:
    """With two granted accounts, 'my bill' is ambiguous and must not guess."""
    db.grant("session-a", "cust-1001", "billing:read")
    db.grant("session-a", "cust-1002", "billing:read")
    decision = checker.check(
        session_id="session-a", intent=IntentName.CHECK_BILL_BALANCE, slots={}
    )
    assert not decision.allowed


def test_registry_refuses_to_execute_on_a_denied_decision(db, registry, checker) -> None:
    intent = Intent(name=IntentName.CREATE_SUPPORT_TICKET, confidence=0.99)
    decision = checker.check(
        session_id="session-a", intent=intent.name, slots={}
    )
    outcome = registry.execute(intent=intent, session_id="session-a", decision=decision)
    assert not outcome.executed
    assert db.tickets_for("cust-1001") == []


# --------------------------------------------------------------------------
# The happy path still has to work
# --------------------------------------------------------------------------
def test_granted_read_succeeds(db, checker, registry) -> None:
    db.grant("session-a", "cust-1001", "billing:read")
    intent = Intent(name=IntentName.CHECK_BILL_BALANCE, confidence=0.9)
    decision = checker.check(session_id="session-a", intent=intent.name, slots={})
    assert decision.allowed

    outcome = registry.execute(intent=intent, session_id="session-a", decision=decision)
    assert outcome.executed
    assert outcome.result["plan"] == "WE Nitro 200"


def test_granted_write_creates_exactly_one_ticket(db, checker, registry) -> None:
    db.grant("session-b", "cust-1002", "tickets:write")
    intent = Intent(
        name=IntentName.CREATE_SUPPORT_TICKET,
        confidence=0.9,
        slots={"subject": "No internet", "body": "Down since yesterday"},
    )
    decision = checker.check(session_id="session-b", intent=intent.name, slots=intent.slots)
    outcome = registry.execute(intent=intent, session_id="session-b", decision=decision)

    assert outcome.executed
    tickets = db.tickets_for("cust-1002")
    assert len(tickets) == 1
    assert tickets[0]["subject"] == "No internet"


def test_negative_balance_is_reported_as_amount_due(db, checker, registry) -> None:
    """cust-1002 has a negative balance; the wording must not say 'settled'."""
    db.grant("session-b", "cust-1002", "billing:read")
    intent = Intent(name=IntentName.CHECK_BILL_BALANCE, confidence=0.9)
    decision = checker.check(session_id="session-b", intent=intent.name, slots={})
    outcome = registry.execute(intent=intent, session_id="session-b", decision=decision)

    assert outcome.result["status"] == "due"
    assert outcome.result["amount_due_egp"] == 42.0
    assert "42.0" in describe_result(outcome, arabic=False)


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def test_refusals_are_audited(db, checker, registry) -> None:
    """An audit log that only records successes cannot answer 'did anyone try?'"""
    intent = Intent(name=IntentName.CREATE_SUPPORT_TICKET, confidence=0.99)
    decision = checker.check(session_id="attacker", intent=intent.name, slots={})
    registry.execute(intent=intent, session_id="attacker", decision=decision)

    trail = db.audit_trail("attacker")
    assert len(trail) == 1
    assert trail[0]["allowed"] == 0
    assert trail[0]["intent"] == "create_support_ticket"


def test_non_action_intents_bypass_permission_checks(checker) -> None:
    decision = checker.check(
        session_id="session-a", intent=IntentName.ANSWER_QUESTION, slots={}
    )
    assert decision.allowed
