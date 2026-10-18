"""Does TabPFN's edge over the naive baseline hold up for cold-start clients?

Splits the Barnet test set in two by `client_invoice_count` (prior invoices
for that client seen before this one): (a) fewer than 3 — cold start, little
or no history to go on — and (b) 3 or more. Reuses run_backtest() as-is, so
the model, split and decision threshold are identical to the main backtest;
only the test-set slicing for this report is new.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import roc_auc_score

from scripts.backtest import run_backtest

COLD_START_MAX_PRIOR_INVOICES = 3
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "barnet_invoices.csv"


def report_subset(name: str, y_true, test_proba, naive_score) -> None:
    late_rate = y_true.mean()
    print(f"\n{name}: n={len(y_true)}  % late={late_rate:.1%}")
    if len(set(y_true)) < 2:
        print("  ROC-AUC undefined (only one class present in this subset)")
        return
    print(f"  TabPFN ROC-AUC:      {roc_auc_score(y_true, test_proba):.3f}")
    print(f"  Naive score ROC-AUC: {roc_auc_score(y_true, naive_score):.3f}")


def main() -> None:
    r = run_backtest(DATA_PATH)  # same model/split/threshold as scripts/backtest.py — unchanged
    test, y_test = r["test"], r["y_test"]

    cold_mask = test["client_invoice_count"] < COLD_START_MAX_PRIOR_INVOICES
    warm_mask = ~cold_mask

    report_subset(
        f"(a) Cold start (<{COLD_START_MAX_PRIOR_INVOICES} prior invoices)",
        y_test[cold_mask],
        r["test_proba"][cold_mask.values],
        r["naive_score"][cold_mask.values],
    )
    report_subset(
        f"(b) Warm ({COLD_START_MAX_PRIOR_INVOICES}+ prior invoices)",
        y_test[warm_mask],
        r["test_proba"][warm_mask.values],
        r["naive_score"][warm_mask.values],
    )


if __name__ == "__main__":
    main()
