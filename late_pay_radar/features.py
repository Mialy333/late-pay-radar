"""Feature engineering for the late-payment backtest.

Every feature here is computable AT ISSUE TIME: amount, payment terms, and
this client's history from invoices issued strictly EARLIER than the one
being scored. Nothing derives from this invoice's own `paid_date`, so a
model trained on these features can't cheat by peeking at the outcome it's
trying to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LATE_THRESHOLD_DAYS = 7

# Base features, available for every dataset (synthetic or Barnet): amount is
# log-transformed since invoice values are right-skewed, and weekday/month
# capture simple seasonality. All of these come straight from amount_eur /
# issue_date / due_date, so they're known at issue time by construction.
FEATURE_COLUMNS = [
    "log_amount",
    "payment_terms_days",
    "client_past_late_ratio",
    "client_past_avg_delay",
    "client_invoice_count",
    "month",
    "issue_weekday",
]

# Only present for datasets that carry these columns (currently Barnet only);
# callers should check `df.columns` before adding these to their feature list.
OPTIONAL_CATEGORICAL_COLUMNS = ["supplier_type", "invoice_source"]


def _label(due_date: pd.Timestamp, paid_date: pd.Timestamp) -> float:
    """1 = paid more than LATE_THRESHOLD_DAYS late, 0 = on time, NaN = still unpaid."""
    if pd.isna(paid_date):
        return float("nan")
    return float((paid_date - due_date).days > LATE_THRESHOLD_DAYS)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per invoice with issue-time features plus the label.

    `client_past_late_ratio` / `client_past_avg_delay` are computed only from
    that client's already-RESOLVED (paid) invoices issued earlier, so they
    never use information that wasn't actually available yet at issue time.
    Invoices still unpaid get `label = NaN` (no ground truth yet) but still
    count toward `client_invoice_count` for later invoices.
    """
    df = df.sort_values(["client", "issue_date"]).reset_index(drop=True)
    optional_cols = [c for c in OPTIONAL_CATEGORICAL_COLUMNS if c in df.columns]

    history: dict[str, list[tuple[int, int]]] = {}  # client -> [(delay_days, was_late), ...]
    prior_count: dict[str, int] = {}

    rows = []
    for _, inv in df.iterrows():
        client = inv["client"]
        past = history.setdefault(client, [])
        n_prior = prior_count.get(client, 0)

        if past:
            past_late_ratio = sum(late for _, late in past) / len(past)
            past_avg_delay = sum(delay for delay, _ in past) / len(past)
        else:
            # no resolved invoice for this client yet: neutral prior
            past_late_ratio = 0.0
            past_avg_delay = 0.0

        row = {
            "invoice_id": inv["invoice_id"],
            "client": client,
            "issue_date": inv["issue_date"],
            "due_date": inv["due_date"],
            "paid_date": inv["paid_date"],
            "amount_eur": inv["amount_eur"],
            "log_amount": float(np.log(inv["amount_eur"])),
            "payment_terms_days": (inv["due_date"] - inv["issue_date"]).days,
            "client_past_late_ratio": past_late_ratio,
            "client_past_avg_delay": past_avg_delay,
            "client_invoice_count": n_prior,
            "month": inv["issue_date"].month,
            "issue_weekday": inv["issue_date"].weekday(),
            "label": _label(inv["due_date"], inv["paid_date"]),
        }
        for col in optional_cols:
            row[col] = inv[col]
        rows.append(row)

        prior_count[client] = n_prior + 1
        if pd.notna(inv["paid_date"]):
            delay_days = (inv["paid_date"] - inv["due_date"]).days
            past.append((delay_days, int(delay_days > LATE_THRESHOLD_DAYS)))

    return pd.DataFrame(rows)
