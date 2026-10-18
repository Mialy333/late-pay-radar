"""Build data/barnet_invoices.csv from the raw Barnet Council period files in
data/raw/, then print row count and % late as a sanity gate before anyone
runs a backtest on it.

Source: London Borough of Barnet, https://open.barnet.gov.uk/dataset/2g58n
Licence: Open Government Licence v3.0 (OGL).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from late_pay_radar.adapters.barnet import load_barnet_invoices
from late_pay_radar.features import LATE_THRESHOLD_DAYS

RAW_FILES = [
    "data/raw/barnet_invoices_2025-06_2025-11.xlsx",
    "data/raw/barnet_invoices_2025-12_2026-05.xlsx",
]
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "barnet_invoices.csv"

MIN_LATE_RATE = 0.10
MAX_LATE_RATE = 0.90


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    df = load_barnet_invoices([root / f for f in RAW_FILES])

    resolved = df[df["paid_date"].notna()]
    late = (resolved["paid_date"] - resolved["due_date"]).dt.days > LATE_THRESHOLD_DAYS
    late_rate = late.mean()

    print(f"rows: {len(df)} total, {len(resolved)} resolved (paid), {len(df) - len(resolved)} still open")
    print(f"% late (>{LATE_THRESHOLD_DAYS}d after due date, among resolved): {late_rate:.1%}")

    if not (MIN_LATE_RATE <= late_rate <= MAX_LATE_RATE):
        print(
            f"\nSTOP: late rate {late_rate:.1%} is outside the sane [10%, 90%] band — "
            "label looks degenerate for this dataset/period, not writing the CSV."
        )
        sys.exit(1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} invoices to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
