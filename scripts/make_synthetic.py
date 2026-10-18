"""Generate a small synthetic invoice dataset for the late-payment backtest.

This data is entirely synthetic (seeded random numbers) and must never be
presented as real invoice or client data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_invoices.csv"
AS_OF = pd.Timestamp("2026-10-16")  # "today": anything that would be paid after this is still open
N_INVOICES = 80
PAYMENT_TERMS_OPTIONS = [15, 30, 45]
LATE_THRESHOLD_DAYS = 7

# Each client has its own payment habit: how often it pays late, how late
# when it does, and how often an invoice is simply never paid (write-off).
CLIENTS = {
    "Acme Corp":         {"late_prob": 0.05, "delay_mean": 0,  "delay_std": 2,  "unpaid_rate": 0.00},
    "Globex":            {"late_prob": 0.15, "delay_mean": 1,  "delay_std": 3,  "unpaid_rate": 0.00},
    "Initech":           {"late_prob": 0.30, "delay_mean": 5,  "delay_std": 6,  "unpaid_rate": 0.02},
    "Umbrella Ltd":      {"late_prob": 0.55, "delay_mean": 10, "delay_std": 8,  "unpaid_rate": 0.05},
    "Hooli":             {"late_prob": 0.75, "delay_mean": 20, "delay_std": 10, "unpaid_rate": 0.08},
    "Soylent":           {"late_prob": 0.40, "delay_mean": 7,  "delay_std": 12, "unpaid_rate": 0.03},
    "Stark Industries":  {"late_prob": 0.25, "delay_mean": 3,  "delay_std": 5,  "unpaid_rate": 0.00},
    "Wayne Enterprises": {"late_prob": 0.65, "delay_mean": 15, "delay_std": 9,  "unpaid_rate": 0.06},
}


def _simulate_delay(rng: np.random.Generator, profile: dict) -> int:
    """Days between due_date and paid_date (negative = paid early)."""
    if rng.random() < profile["late_prob"]:
        # late: keep the delay comfortably past the 7-day threshold
        mean = profile["delay_mean"] + LATE_THRESHOLD_DAYS
        return max(LATE_THRESHOLD_DAYS + 1, round(rng.normal(mean, profile["delay_std"])))
    # on time: small delay, occasionally a few days early
    return min(LATE_THRESHOLD_DAYS, round(rng.normal(0, 2)))


def make_synthetic_invoices(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    client_names = list(CLIENTS.keys())

    issue_dates = AS_OF - pd.to_timedelta(rng.integers(0, 24 * 30, size=N_INVOICES), unit="D")
    clients = rng.choice(client_names, size=N_INVOICES)
    terms = rng.choice(PAYMENT_TERMS_OPTIONS, size=N_INVOICES)
    amounts = np.round(rng.uniform(250, 6000, size=N_INVOICES), 2)

    rows = []
    for i in range(N_INVOICES):
        client = clients[i]
        profile = CLIENTS[client]
        issue_date = issue_dates[i]
        due_date = issue_date + pd.Timedelta(days=int(terms[i]))

        if rng.random() < profile["unpaid_rate"]:
            paid_date = pd.NaT  # written off: never paid
        else:
            delay = _simulate_delay(rng, profile)
            candidate_paid = due_date + pd.Timedelta(days=int(delay))
            candidate_paid = max(candidate_paid, issue_date + pd.Timedelta(days=1))
            paid_date = candidate_paid if candidate_paid <= AS_OF else pd.NaT  # still open today

        rows.append(
            {
                "invoice_id": f"INV-{i + 1:04d}",
                "client": client,
                "amount_eur": amounts[i],
                "issue_date": issue_date.date().isoformat(),
                "due_date": due_date.date().isoformat(),
                "paid_date": paid_date.date().isoformat() if pd.notna(paid_date) else "",
            }
        )

    return pd.DataFrame(rows).sort_values("issue_date").reset_index(drop=True)


def main() -> None:
    df = make_synthetic_invoices()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} synthetic invoices to {OUTPUT_PATH}")
    print(f"  still open (unpaid): {(df['paid_date'] == '').sum()}")


if __name__ == "__main__":
    main()
