"""The database the model never touches.

Brief B.4 is explicit: the LLM must not connect to or act on the database. So
this module is reachable only from `actions/registry.py`, which is itself only
reachable after a permission check. Nothing in `llm/` imports anything here, and
that boundary is the architecture — not a convention.

Every query is parameterised. That is ordinary practice, but it matters more
than usual here because part of the input did originate from a language model:
the intent slots. Slot values are treated as untrusted user data throughout.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   TEXT PRIMARY KEY,
    msisdn        TEXT UNIQUE NOT NULL,
    full_name     TEXT NOT NULL,
    plan          TEXT NOT NULL,
    balance_egp   REAL NOT NULL DEFAULT 0,
    bill_due_date TEXT
);

CREATE TABLE IF NOT EXISTS permissions (
    session_id  TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    scope       TEXT NOT NULL,
    PRIMARY KEY (session_id, customer_id, scope)
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_audit (
    audit_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    intent      TEXT NOT NULL,
    allowed     INTEGER NOT NULL,
    reason      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);
"""

# A tiny fixture set so the B.4 chain is demonstrable end to end. Two customers
# with different permission grants is the minimum needed to show that the
# backend — not the model — decides who may do what.
SEED_CUSTOMERS = [
    ("cust-1001", "01012345678", "Mona Adel", "WE Nitro 200", 213.50, "2026-10-05"),
    ("cust-1002", "01198765432", "Karim Fouad", "WE Home 100", -42.00, "2026-09-28"),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ActionDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        # Concurrent sessions read while one writes; WAL keeps readers unblocked.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def seed(self) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO customers "
                "(customer_id, msisdn, full_name, plan, balance_egp, bill_due_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                SEED_CUSTOMERS,
            )
        log.info("seeded %d demo customers", len(SEED_CUSTOMERS))

    # -- reads -------------------------------------------------------------
    def customer_by_msisdn(self, msisdn: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE msisdn = ?", (msisdn,)
            ).fetchone()
        return dict(row) if row else None

    def customer_by_id(self, customer_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            ).fetchone()
        return dict(row) if row else None

    # -- permissions -------------------------------------------------------
    def grant(self, session_id: str, customer_id: str, scope: str) -> None:
        """Bind a session to a customer and a scope.

        In production this is what an authenticated login would establish. In
        the POC the demo harness grants it explicitly, which keeps the
        permission check real while leaving authentication itself as future work.
        """
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO permissions (session_id, customer_id, scope) "
                "VALUES (?, ?, ?)",
                (session_id, customer_id, scope),
            )

    def has_permission(self, session_id: str, customer_id: str, scope: str) -> bool:
        if not session_id or not customer_id or not scope:
            return False
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM permissions "
                "WHERE session_id = ? AND customer_id = ? AND scope = ?",
                (session_id, customer_id, scope),
            ).fetchone()
        return row is not None

    def customers_for_session(self, session_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT customer_id FROM permissions WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return [row["customer_id"] for row in rows]

    # -- writes ------------------------------------------------------------
    def create_ticket(
        self, *, customer_id: str, session_id: str, subject: str, body: str
    ) -> dict[str, Any]:
        ticket_id = f"TKT-{uuid.uuid4().hex[:10].upper()}"
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tickets "
                "(ticket_id, customer_id, session_id, subject, body, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'open', ?)",
                (ticket_id, customer_id, session_id, subject, body, _now()),
            )
        return {"ticket_id": ticket_id, "status": "open", "customer_id": customer_id}

    def tickets_for(self, customer_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE customer_id = ? ORDER BY created_at DESC",
                (customer_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- audit -------------------------------------------------------------
    def audit(self, *, session_id: str, intent: str, allowed: bool, reason: str = "") -> None:
        """Record every action decision, refusals included.

        Refusals are the interesting half: an audit log that only records
        successes cannot answer "did anyone try?".
        """
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO action_audit "
                "(audit_id, session_id, intent, allowed, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, session_id, intent, int(allowed), reason, _now()),
            )

    def audit_trail(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM action_audit WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]
