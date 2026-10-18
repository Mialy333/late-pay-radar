"""Checks whether a drafted reminder actually names the real amount and due
date, rather than trusting the model to have gotten it right. Shared by
scripts/agent_demo.py and app.py so both retry the same way.
"""

from __future__ import annotations

from late_pay_radar.scoring import format_amount

FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]  # fmt: skip


def due_date_variants(due_date_iso: str) -> list[str]:
    """Acceptable ways of writing this due date: '31 juillet', '31/07/2026', '2026-07-31'."""
    y, m, d = due_date_iso.split("-")
    return [f"{int(d)} {FRENCH_MONTHS[int(m) - 1]}", f"{d}/{m}/{y}", due_date_iso]


def amount_variants(amount: float, currency: str) -> list[str]:
    """Both language-specific formats are accepted -- French '3 250,00 €' or
    English '€3,250.00' -- since the reminder may be written in either."""
    return [format_amount(amount, currency, "French"), format_amount(amount, currency, "English")]


def reminder_is_valid(text: str, amount: float, currency: str, due_date_iso: str) -> bool:
    text_norm = text.lower()
    has_amount = any(v.lower() in text_norm for v in amount_variants(amount, currency))
    has_date = any(v.lower() in text_norm for v in due_date_variants(due_date_iso))
    return has_amount and has_date
