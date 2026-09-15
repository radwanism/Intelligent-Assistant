"""Input guardrails — step 1 of the B.4 chain.

These run *before the request reaches the model at all*. That ordering is the
whole point: a guardrail implemented as a system-prompt instruction is not a
guardrail, because the attack it defends against is one that rewrites the
model's instructions.

Scope is deliberately narrow. This is pattern matching, not a classifier:

  * On CPU, a guardrail model would cost more than the answer itself.
  * Pattern matching is auditable and testable — `tests/test_guardrails.py`
    asserts each rule, which is what makes this a security control rather
    than a vibe.

Its limits are stated honestly in the README: this stops the common,
scripted attempts. A determined adversary with paraphrase budget gets past
regex, which is why the architecture never relies on the guardrail alone —
the LLM has no database access regardless of what it is persuaded to say
(B.4 steps 3-4). Defence in depth is the actual control; this is the cheap
outer layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import GuardrailVerdict

MAX_INPUT_CHARS = 4000


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    detail: str


def _rule(name: str, pattern: str, detail: str) -> Rule:
    return Rule(name, re.compile(pattern, re.IGNORECASE | re.UNICODE), detail)


# Instruction-override attempts, English and Arabic.
INJECTION_RULES: tuple[Rule, ...] = (
    _rule(
        "ignore_instructions",
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
        r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
        r"\b(instruction|prompt|rule|direction|context)",
        "attempt to override system instructions",
    ),
    _rule(
        "ignore_instructions_ar",
        r"(تجاهل|انسى|تخطى|الغي)\s*(كل\s*)?(التعليمات|الاوامر|الأوامر|التوجيهات)",
        "attempt to override system instructions (Arabic)",
    ),
    _rule(
        "reveal_prompt",
        r"\b(reveal|show|print|repeat|output|display|dump)\b[^.\n]{0,30}"
        r"\b(system\s*prompt|initial\s*instruction|your\s*instruction|prompt above)",
        "attempt to extract the system prompt",
    ),
    _rule(
        "reveal_prompt_ar",
        r"(اظهر|أظهر|اطبع|اعرض|كرر)\s*[^.\n]{0,30}(التعليمات|البرومبت|النظام)",
        "attempt to extract the system prompt (Arabic)",
    ),
    _rule(
        "role_override",
        r"\b(you are now|from now on you|act as|pretend to be|roleplay as)\b"
        r"[^.\n]{0,40}\b(dan|developer mode|admin|root|unrestricted|jailbroken)",
        "role-override / jailbreak persona",
    ),
    _rule(
        "fake_system_turn",
        r"(^|\n)\s*(system|assistant)\s*[:>]\s*",
        "forged conversation turn in user input",
    ),
    _rule(
        "delimiter_injection",
        r"(<\|im_start\|>|<\|im_end\|>|\[/?INST\]|<<SYS>>|###\s*system)",
        "chat-template delimiter injection",
    ),
)

# Attempts to get the model to act on data directly — the exact thing B.4
# architecturally forbids. Flagged separately so the demo can show that the
# refusal is by design, not by luck.
DB_ACTION_RULES: tuple[Rule, ...] = (
    _rule(
        "sql_injection",
        r"\b(drop\s+table|delete\s+from|truncate\s+table|update\s+\w+\s+set|"
        r"insert\s+into|union\s+select|;\s*--)\b",
        "SQL embedded in user input",
    ),
    _rule(
        "direct_db_command",
        r"\b(execute|run|perform)\b[^.\n]{0,25}\b(query|sql|command|script)\b"
        r"[^.\n]{0,25}\b(database|db|table)\b",
        "request for the model to operate the database directly",
    ),
    _rule(
        "privilege_escalation",
        r"\b(as|become|switch to)\b\s+(an?\s+)?(admin|administrator|superuser|root)\b"
        r"|\bbypass\b[^.\n]{0,20}\b(permission|auth|check)",
        "privilege-escalation request",
    ),
)

ALL_RULES: tuple[Rule, ...] = (*INJECTION_RULES, *DB_ACTION_RULES)


def check_input(text: str) -> GuardrailVerdict:
    """Run before anything else touches the request."""
    if text is None or not text.strip():
        return GuardrailVerdict(allowed=False, rule="empty_input", detail="empty message")

    if len(text) > MAX_INPUT_CHARS:
        # Oversized input is both a context-stuffing vector and a CPU-budget
        # risk on the cpu-lite profile.
        return GuardrailVerdict(
            allowed=False,
            rule="input_too_long",
            detail=f"input exceeds {MAX_INPUT_CHARS} characters",
        )

    for rule in ALL_RULES:
        if rule.pattern.search(text):
            return GuardrailVerdict(allowed=False, rule=rule.name, detail=rule.detail)

    return GuardrailVerdict(allowed=True)


# --------------------------------------------------------------------------
# Output side
# --------------------------------------------------------------------------
# Never let a masked placeholder reach the user as-is, and never let the model
# echo a system prompt fragment back out.
LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|im_(start|end)\|>"),
    re.compile(r"(?i)\byou are the telecom egypt\b"),
)


def check_output(text: str) -> GuardrailVerdict:
    for pattern in LEAK_PATTERNS:
        if pattern.search(text):
            return GuardrailVerdict(
                allowed=False, rule="output_leak", detail="model echoed prompt scaffolding"
            )
    return GuardrailVerdict(allowed=True)


REFUSAL_AR = (
    "معلش، مش هقدر أساعد في الطلب ده. "
    "أقدر أجاوبك على أي استفسار عن خدمات وباقات المصرية للاتصالات."
)
REFUSAL_EN = (
    "Sorry, I can't help with that request. "
    "I can answer questions about Telecom Egypt services and plans."
)


def refusal_message(is_arabic: bool) -> str:
    return REFUSAL_AR if is_arabic else REFUSAL_EN
