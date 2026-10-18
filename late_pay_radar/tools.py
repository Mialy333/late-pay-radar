"""Strands tools that expose live invoice risk scoring to the agent.

Both tools rebuild features from the underlying CSVs on every call (the
datasets are small, so this is cheap) and hybrid-score via
`late_pay_radar.scoring`. Every call is appended to `CALL_LOG` (tool name,
arguments, seconds) so scripts/agent_demo.py can report what the agent
actually did, not just what it says it did.

`data/my_invoice.csv` (the user's own tracked invoice) is scored "as of its
issue date": any `paid_date` already recorded in that file is deliberately
ignored for scoring, so the demo reflects what the risk engine would have
said at the time, not what we now know with hindsight.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from strands import tool

from late_pay_radar.features import build_features
from late_pay_radar.scoring import rank_open_invoices, score_invoices

BARNET_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "barnet_invoices.csv"
MY_INVOICE_PATH = Path(__file__).resolve().parent.parent / "data" / "my_invoice.csv"

CALL_LOG: list[dict] = []

# get_invoice_risk looks up invoices from whichever dataframe this currently
# points at. Defaults to the combined Barnet + my_invoice.csv set (CLI demo
# behavior); app.py swaps this to whatever data source is selected in the UI
# (the Barnet demo, or the user's uploaded CSV, kept in memory only).
_active_loader: Callable[[], pd.DataFrame] = lambda: _load_combined()  # noqa: E731


def set_active_loader(loader: Callable[[], pd.DataFrame]) -> None:
    global _active_loader
    _active_loader = loader


def _collect_notes(result: list[dict] | dict | None) -> list[str]:
    """Pull any non-null training_data_note(s) out of a tool result, for
    deterministic disclosure by the caller (see scripts/agent_demo.py) --
    never rely on the model to repeat this itself."""
    if result is None:
        return []
    rows = result if isinstance(result, list) else [result]
    return [r["training_data_note"] for r in rows if r.get("training_data_note")]


def _load_combined() -> pd.DataFrame:
    frames = [pd.read_csv(BARNET_DATA_PATH, parse_dates=["issue_date", "due_date", "paid_date"])]
    if MY_INVOICE_PATH.exists():
        mine = pd.read_csv(MY_INVOICE_PATH, parse_dates=["issue_date", "due_date", "paid_date"])
        mine["paid_date"] = pd.NaT  # score as of issue date, ignore any outcome already on file
        frames.append(mine)
    return pd.concat(frames, ignore_index=True)


@tool
def list_open_invoices(top_n: int = 5) -> list[dict]:
    """List the top N open (not yet paid) invoices by risk, highest first.

    Sorting and selection happen HERE, not in the agent's own reasoning: this
    always returns at most `top_n` items, already ranked, so the agent never
    needs to sort or pick from a long list itself.

    Args:
        top_n: Maximum number of invoices to return (default 5).

    Returns a list of dicts, each with: rank (1 = highest risk), invoice_id,
    risk (0-1), source, confidence, client_invoice_count,
    client_past_late_ratio, amount, currency, amount_display, due_date,
    due_status, as_of_date, training_data_note. Ties in risk are broken by
    amount, highest first.
    """
    start = time.perf_counter()
    scored = rank_open_invoices(_load_combined())[:top_n]
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    CALL_LOG.append(
        {
            "tool": "list_open_invoices",
            "args": {"top_n": top_n},
            "seconds": round(time.perf_counter() - start, 3),
            "training_data_notes": _collect_notes(scored),
            "result": scored,
        }
    )
    return scored


@tool
def get_invoice_risk(invoice_id: str) -> dict | None:
    """Get the late-payment risk score for one specific invoice by its id.

    Args:
        invoice_id: The invoice's id, e.g. "BARNET-00123" or "2026-001".

    Returns a dict (risk 0-1, source, confidence, client_invoice_count,
    client_past_late_ratio, amount, currency, amount_display, due_date,
    due_status, as_of_date, training_data_note), or None if not found.
    """
    start = time.perf_counter()
    features = build_features(_active_loader())
    scored = score_invoices(features)
    result = next((r for r in scored if r["invoice_id"] == invoice_id), None)
    CALL_LOG.append(
        {
            "tool": "get_invoice_risk",
            "args": {"invoice_id": invoice_id},
            "seconds": round(time.perf_counter() - start, 3),
            "training_data_notes": _collect_notes(result),
            "result": result,
        }
    )
    return result
