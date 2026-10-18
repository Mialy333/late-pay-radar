"""Hybrid risk scoring: route each invoice to whichever signal the backtest
showed is actually better for it (see scripts/cold_start_analysis.py):

- fewer than 3 prior invoices for this client ("cold start"): there's too
  little client history to trust `client_past_late_ratio`, so fall back to
  TabPFN's predicted probability, flagged as low confidence.
- 3 or more prior invoices: `client_past_late_ratio` itself, flagged as
  normal confidence -- the backtest found it edges out TabPFN once a client
  has real history.

TabPFN is trained ONCE, on every resolved Barnet invoice, and cached for the
life of the process (`get_model()` is an `lru_cache`d singleton) so repeated
scoring calls don't keep re-fitting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tabpfn import TabPFNClassifier

from late_pay_radar.features import FEATURE_COLUMNS, OPTIONAL_CATEGORICAL_COLUMNS, build_features

COLD_START_MAX_PRIOR_INVOICES = 3
BARNET_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "barnet_invoices.csv"

# End of the most recent Barnet report period we loaded (scripts/build_barnet_dataset.py
# uses the 01.12.25-31.05.26 file) -- Barnet's own "open invoices" are only open as of
# THAT date, not today, since the data hasn't been refreshed since.
BARNET_REPORT_END = date(2026, 5, 31)

# TabPFN is trained ONLY on Barnet Council data (see get_model()) -- flag that
# explicitly whenever we fall back to it, so the agent can warn the user this
# is public-sector data, not necessarily representative of their own clients.
TRAINING_DATA_NOTE = (
    "Model trained on UK public-sector supplier invoices (Barnet Council, 77% paid late); "
    "transfer to other clients is uncertain."
)


def _currency_for(invoice_id: str) -> str:
    """Barnet invoices are real GBP; anything else is the user's own (EUR)."""
    return "GBP" if invoice_id.startswith("BARNET-") else "EUR"


def _format_amount(amount: float, currency: str) -> str:
    """Locale-appropriate display string, tied to currency (not the reminder's language)."""
    plain = f"{amount:,.2f}"  # e.g. "3,738.89"
    if currency == "EUR":
        integer_part, decimal_part = plain.split(".")
        return f"{integer_part.replace(',', ' ')},{decimal_part} €"  # "3 738,89 €"
    if currency == "GBP":
        return f"£{plain}"  # "£3,738.89"
    return f"{plain} {currency}"


def _due_status(due_date: date, as_of: date) -> str:
    delta = (due_date - as_of).days
    if delta >= 0:
        return f"not yet due, due in {delta} days"
    return f"overdue by {-delta} days"


def _as_of_for(invoice_id: str, issue_date: date) -> date:
    """The reference date 'now' for this invoice's due_status.

    Barnet invoices: the Barnet report itself is stale (last refreshed
    2026-05-31), so "open as of today" would overstate how overdue they are
    by however long it's been since that report -- use the report's own end
    date instead. Any other invoice (the user's own data/my_invoice.csv): we
    don't have a live "today" for demo data either, so score it as if it had
    just been issued -- as_of = its own issue_date.
    """
    if invoice_id.startswith("BARNET-"):
        return BARNET_REPORT_END
    return issue_date


@dataclass
class ScoringModel:
    clf: TabPFNClassifier
    feature_columns: list[str]
    categories: dict[str, dict[str, int]]  # column -> {value: code}, fit on training data only


def _encode_categories(df: pd.DataFrame, categories: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Add a `{col}_code` column for every trained categorical feature.

    Must produce these columns even when `col` is entirely absent from `df`
    (e.g. a dataset -- an upload, or my_invoice.csv alone -- that never had
    supplier_type/invoice_source at all): the model's feature_columns always
    includes them, so every row gets the "unseen category" fallback code
    rather than a missing-column error.
    """
    df = df.copy()
    for col, mapping in categories.items():
        unseen_code = len(mapping)
        code_col = f"{col}_code"
        if col in df.columns:
            df[code_col] = df[col].map(mapping).fillna(unseen_code).astype(int)
        else:
            df[code_col] = unseen_code
    return df


@lru_cache(maxsize=1)
def get_model() -> ScoringModel:
    """Train TabPFN once on every resolved Barnet invoice, then cache it."""
    raw = pd.read_csv(BARNET_DATA_PATH, parse_dates=["issue_date", "due_date", "paid_date"])
    features = build_features(raw)
    resolved = features[features["label"].notna()].copy()
    resolved["label"] = resolved["label"].astype(int)

    categories: dict[str, dict[str, int]] = {}
    feature_columns = list(FEATURE_COLUMNS)
    for col in OPTIONAL_CATEGORICAL_COLUMNS:
        if col not in resolved.columns:
            continue
        categories[col] = {v: i for i, v in enumerate(sorted(resolved[col].dropna().unique()))}
        feature_columns.append(f"{col}_code")

    resolved = _encode_categories(resolved, categories)

    clf = TabPFNClassifier(random_state=42)
    clf.fit(resolved[feature_columns], resolved["label"])
    return ScoringModel(clf=clf, feature_columns=feature_columns, categories=categories)


def score_invoices(features_df: pd.DataFrame) -> list[dict]:
    """Hybrid-score every row of a dataframe already run through `build_features()`.

    Returns one dict per invoice: invoice_id, risk (0-1), source, confidence,
    client_invoice_count, client_past_late_ratio, amount, currency,
    amount_display, due_date, due_status, as_of_date, training_data_note.
    """
    model = get_model()
    encoded = _encode_categories(features_df, model.categories)
    cold_mask = encoded["client_invoice_count"] < COLD_START_MAX_PRIOR_INVOICES

    risk = encoded["client_past_late_ratio"].astype(float).to_numpy().copy()
    if cold_mask.any():
        cold_X = encoded.loc[cold_mask, model.feature_columns]
        risk[cold_mask.to_numpy()] = model.clf.predict_proba(cold_X)[:, 1]

    rows = []
    for i, (_, inv) in enumerate(encoded.iterrows()):
        is_cold = bool(cold_mask.iloc[i])
        invoice_id = inv["invoice_id"]
        currency = _currency_for(invoice_id)
        amount = float(inv["amount_eur"])
        due_date = inv["due_date"].date()
        as_of = _as_of_for(invoice_id, inv["issue_date"].date())
        rows.append(
            {
                "invoice_id": invoice_id,
                "risk": round(float(risk[i]), 3),
                "source": "TabPFN (new client)" if is_cold else "client history",
                "confidence": "low" if is_cold else "normal",
                "client_invoice_count": int(inv["client_invoice_count"]),
                "client_past_late_ratio": round(float(inv["client_past_late_ratio"]), 3),
                "amount": amount,
                "currency": currency,
                "amount_display": _format_amount(amount, currency),
                "due_date": due_date.isoformat(),
                "due_status": _due_status(due_date, as_of),
                "as_of_date": as_of.isoformat(),
                "training_data_note": TRAINING_DATA_NOTE if is_cold else None,
            }
        )
    return rows


def rank_open_invoices(raw_df: pd.DataFrame) -> list[dict]:
    """Score every open (unpaid) invoice in a raw invoice dataframe, ranked by
    risk then amount (both highest first). Dataset-agnostic: works for
    Barnet, the user's own invoices, or any dataframe matching our schema.
    """
    features = build_features(raw_df)
    open_invoices = features[features["paid_date"].isna()]
    return sorted(score_invoices(open_invoices), key=lambda r: (r["risk"], r["amount"]), reverse=True)
