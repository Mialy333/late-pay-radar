"""Adapter: map the Barnet Council "Financial Invoices Late Payments" open
dataset onto this project's invoice schema (invoice_id, client, amount_eur,
issue_date, due_date, paid_date, supplier_type, invoice_source).

Source: London Borough of Barnet, https://open.barnet.gov.uk/dataset/2g58n
Licence: Open Government Licence v3.0 (OGL) — contains public sector
information licensed under the Open Government Licence v3.0.

Caveat: `amount_eur` is a straight passthrough of Barnet's "Invoice Amount",
which is actually GBP, not EUR. The column is named for schema compatibility
with the synthetic dataset only — no currency conversion is applied.

We read: Supplier Name, Supplier Type, Invoice Source, Invoice Number,
Invoice Amount, Invoice Date, Due Date (Last), Payment Date (Last) — all
known at (or before) issue time. Every other column in the source file is
dropped, and why:

- `Fully Paid`, `Payment Status Flag`, `Paid Within Terms (Due Date)`,
  `Paid Within Terms (Rcvd Date)`, `Days in Dispute (on Holds)`,
  `Days that Interest is Due (Rcvd Date)`, `Notional Interest Due
  (Rcvd Date)`, `Due Date to Payment Date (Rcvd Date)`, `Total Paid` —
  these all describe what actually happened to the invoice (whether/when it
  was paid, disputed, charged interest). Using any of them as a feature
  would leak the label we're trying to predict.
- `Pay Band (from Received Date)` — a bucketed version of actual payment
  timing, i.e. a discretised outcome. Same leakage problem.
- `Terms Name` — a free-text duplicate of the due date already encoded in
  `Due Date (Last)`; `payment_terms_days` is derived directly from
  `Due Date (Last) - Invoice Date` instead, which is exact and doesn't need
  parsing "Immediate" / "30 Days" / etc.
- `Invoice Received Date`, `Invoice Creation Date` — administrative
  timestamps, not the invoice's issue date; `Invoice Received Date` is empty
  for every row in the files we use, so it carries no information anyway.
- `Supplier Number` — a second identifier for the same client as
  `Supplier Name`; redundant.
- `Check LE and BU`, `Business Unit` — internal Barnet accounting fields
  unrelated to payment behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = [
    "Supplier Name",
    "Supplier Type",
    "Invoice Source",
    "Invoice Number",
    "Invoice Amount",
    "Invoice Date",
    "Due Date (Last)",
    "Payment Date (Last)",
]


def _load_one(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, usecols=SOURCE_COLUMNS)
    raw = raw[raw["Invoice Amount"] > 0]  # drop credit notes / zero-value corrections: not real invoices
    raw = raw[raw["Due Date (Last)"] >= raw["Invoice Date"]]  # drop rows with a bad/inconsistent due date
    return raw


def load_barnet_invoices(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate the given Barnet period files into our schema."""
    raw = pd.concat([_load_one(Path(p)) for p in paths], ignore_index=True)
    raw = raw.sort_values("Invoice Date").reset_index(drop=True)

    return pd.DataFrame(
        {
            "invoice_id": [f"BARNET-{i + 1:05d}" for i in range(len(raw))],
            "client": raw["Supplier Name"],
            "amount_eur": raw["Invoice Amount"],  # GBP, see module docstring
            "issue_date": raw["Invoice Date"],
            "due_date": raw["Due Date (Last)"],
            "paid_date": raw["Payment Date (Last)"],  # NaT = still open/unpaid
            "supplier_type": raw["Supplier Type"],
            "invoice_source": raw["Invoice Source"],
        }
    )
