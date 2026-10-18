"""Checks whether a drafted reminder actually names the real amount and due
date, rather than trusting the model to have gotten it right. Shared by
scripts/agent_demo.py and app.py so both retry the same way.
"""

from __future__ import annotations

FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]  # fmt: skip


def due_date_variants(due_date_iso: str) -> list[str]:
    """Acceptable ways of writing this due date: '31 juillet', '31/07/2026', '2026-07-31'."""
    y, m, d = due_date_iso.split("-")
    return [f"{int(d)} {FRENCH_MONTHS[int(m) - 1]}", f"{d}/{m}/{y}", due_date_iso]


def reminder_is_valid(text: str, amount_display: str, due_date_iso: str) -> bool:
    text_norm = text.lower()
    has_amount = amount_display.lower() in text_norm
    has_date = any(v.lower() in text_norm for v in due_date_variants(due_date_iso))
    return has_amount and has_date
