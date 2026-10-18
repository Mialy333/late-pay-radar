"""Run 2 demo prompts through the late-pay-radar agent (one invoice, one
reminder each), logging tool calls, arguments, total latency, and whether a
reminder-validation retry happened.

Ranking across many invoices is NOT demoed here -- that's a code-computed
table for the UI, not something the agent does (see late_pay_radar/agent.py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from late_pay_radar.agent import build_agent
from late_pay_radar.reminder_check import reminder_is_valid
from late_pay_radar.tools import CALL_LOG, list_open_invoices


def pick_barnet_invoice_id() -> str:
    """Find a real open Barnet invoice id for the second demo prompt."""
    open_invoices = list_open_invoices()
    CALL_LOG.clear()  # this lookup is setup, not part of a logged demo run
    barnet = next((inv for inv in open_invoices if inv["invoice_id"].startswith("BARNET-")), None)
    return barnet["invoice_id"] if barnet else open_invoices[0]["invoice_id"]


def get_invoice_result_from_log() -> dict | None:
    return next((c["result"] for c in CALL_LOG if c["tool"] == "get_invoice_risk" and c.get("result")), None)


def run(agent, prompt: str) -> None:
    CALL_LOG.clear()
    start = time.perf_counter()
    result = agent(prompt)
    text = str(result)

    invoice = get_invoice_result_from_log()
    retried = False
    still_invalid = False
    if invoice is not None:
        valid = reminder_is_valid(text, invoice["amount_display"], invoice["due_date"])
        if not valid:
            followup = (
                f"Your reminder must include the amount {invoice['amount_display']} "
                f"and the due date {invoice['due_date']}. Rewrite only the reminder."
            )
            result = agent(followup)
            text = str(result)
            retried = True
            still_invalid = not reminder_is_valid(text, invoice["amount_display"], invoice["due_date"])

    total_seconds = time.perf_counter() - start

    print(f"\n{'=' * 70}\nPROMPT: {prompt}\n{'=' * 70}")
    print(text)

    notes = dict.fromkeys(n for call in CALL_LOG for n in call.get("training_data_notes", []))
    for note in notes:
        print(f"\nNote: {note}")
    if still_invalid:
        print("\nWARNING: reminder may still be missing the exact amount or due date after 1 retry.")

    print("\n--- run log ---")
    for call in CALL_LOG:
        print(f"  tool={call['tool']}  args={call['args']}  seconds={call['seconds']}")
    print(f"  retried: {retried}")
    print(f"  total latency: {total_seconds:.2f}s")


def main() -> None:
    agent = build_agent()
    barnet_id = pick_barnet_invoice_id()

    prompts = [
        "Explain the risk on invoice 2026-001 and draft a reminder in French",
        f"Explain the risk on invoice {barnet_id} and draft a reminder in English",
    ]
    for prompt in prompts:
        run(agent, prompt)


if __name__ == "__main__":
    main()
