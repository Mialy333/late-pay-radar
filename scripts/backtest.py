"""Backtest: do issue-time features beat simple baselines at predicting which
invoices will be paid more than 7 days late?

Time-based split: the oldest ~70% of RESOLVED (paid) invoices train the
model, the most recent resolved invoices test it. Invoices still unpaid have
no ground truth yet and are excluded from both splits.

TabPFN's decision threshold is chosen on a VALIDATION slice carved out of the
last 20% of the training set (maximizing balanced accuracy) — never on the
test set. ROC-AUC is threshold-independent and uses predict_proba directly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Project root isn't on sys.path when this script is run directly
# (e.g. `uv run python scripts/backtest.py`); add it so `late_pay_radar` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()  # must happen before `import tabpfn` so TABPFN_TOKEN is already in the environment

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tabpfn import TabPFNClassifier

from late_pay_radar.features import FEATURE_COLUMNS, OPTIONAL_CATEGORICAL_COLUMNS, build_features

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_invoices.csv"
TRAIN_FRACTION = 0.7
VAL_FRACTION_OF_TRAIN = 0.2
MAX_TRAIN_ROWS = 3000  # TabPFN is a forward-pass-only model; cap train size to keep CPU runs fast


def load_invoices(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path, parse_dates=["issue_date", "due_date", "paid_date"])


def naive_baseline_preds(resolved: pd.DataFrame) -> pd.Series:
    """Predict 'late' iff this client's previous (by issue_date) invoice was late.

    Only looks at the client's prior invoice, same information cutoff as the
    model: no peeking at the invoice being scored.
    """
    resolved = resolved.sort_values(["client", "issue_date"])
    preds = {}
    last_late: dict[str, int] = {}
    for _, inv in resolved.iterrows():
        client = inv["client"]
        preds[inv["invoice_id"]] = last_late.get(client, 0)
        last_late[client] = int(inv["label"])
    return pd.Series(preds)


def add_categorical_codes(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Encode any OPTIONAL_CATEGORICAL_COLUMNS present as integer codes.

    The code mapping is fit on `train` only; categories never seen in train
    map to a single "unseen" code in `test` (no vocabulary leaks from test).
    Mutates `train`/`test` in place and returns the extended feature list.
    """
    feature_columns = list(feature_columns)
    for col in OPTIONAL_CATEGORICAL_COLUMNS:
        if col not in train.columns:
            continue
        categories = {v: i for i, v in enumerate(sorted(train[col].dropna().unique()))}
        unseen_code = len(categories)
        code_col = f"{col}_code"
        train[code_col] = train[col].map(categories).fillna(unseen_code).astype(int)
        test[code_col] = test[col].map(categories).fillna(unseen_code).astype(int)
        feature_columns.append(code_col)
    return feature_columns


def best_threshold_by_balanced_accuracy(y_val, val_proba) -> float:
    """Scan candidate thresholds on the VALIDATION slice only, maximizing balanced accuracy."""
    candidates = np.unique(np.concatenate([[0.0, 1.0], val_proba]))
    best_t, best_score = 0.5, -1.0
    for t in candidates:
        score = balanced_accuracy_score(y_val, (val_proba >= t).astype(int))
        if score > best_score:
            best_t, best_score = t, score
    return best_t


def report(name: str, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{name}")
    print(f"  precision={precision_score(y_true, y_pred, zero_division=0):.2f}", end="  ")
    print(f"recall={recall_score(y_true, y_pred, zero_division=0):.2f}", end="  ")
    print(f"balanced_acc={balanced_accuracy_score(y_true, y_pred):.2f}")
    print("  confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(f"    {cm.tolist()}")


def summary_row(name: str, y_true, y_pred, proba) -> dict:
    auc = roc_auc_score(y_true, proba) if proba is not None and len(set(y_true)) > 1 else float("nan")
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def run_backtest(data_path: Path) -> dict:
    """Run the full backtest and return its artifacts (no printing).

    Reused by scripts/backtest.py's own CLI output and by one-off analyses
    (e.g. cold-start slicing) that need the exact same model/threshold/split
    without re-deriving or changing any of them.
    """
    raw = load_invoices(data_path)
    features = build_features(raw)

    resolved = features[features["label"].notna()].copy()
    resolved["label"] = resolved["label"].astype(int)
    resolved = resolved.sort_values("issue_date").reset_index(drop=True)

    split_idx = int(len(resolved) * TRAIN_FRACTION)
    train, test = resolved.iloc[:split_idx].copy(), resolved.iloc[split_idx:].copy()

    if len(train) > MAX_TRAIN_ROWS:
        print(f"train has {len(train)} rows, capping to the {MAX_TRAIN_ROWS} most recent for speed")
        train = train.iloc[-MAX_TRAIN_ROWS:].copy()

    feature_columns = add_categorical_codes(train, test, FEATURE_COLUMNS)

    # Validation slice = last 20% of (the already time-sorted) training set.
    # Used ONLY to pick TabPFN's decision threshold — never touches test.
    val_size = max(1, int(len(train) * VAL_FRACTION_OF_TRAIN))
    fit_sub, val = train.iloc[:-val_size], train.iloc[-val_size:]
    split_sizes = f"train: {len(train)} ({len(fit_sub)} fit + {len(val)} validation), test (most recent): {len(test)}"

    X_fit, y_fit = fit_sub[feature_columns], fit_sub["label"]
    X_val, y_val = val[feature_columns], val["label"]
    X_train, y_train = train[feature_columns], train["label"]
    X_test, y_test = test[feature_columns], test["label"]

    # 1) Fit on the fit-only slice to pick a threshold from validation performance.
    threshold_clf = TabPFNClassifier(random_state=42)
    threshold_clf.fit(X_fit, y_fit)
    val_proba = threshold_clf.predict_proba(X_val)[:, 1]
    threshold = best_threshold_by_balanced_accuracy(y_val, val_proba)

    # 2) Refit on the full (capped) training set for the final model, apply that threshold to test.
    clf = TabPFNClassifier(random_state=42)
    fit_start = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    test_proba = clf.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - predict_start
    tabpfn_preds = (test_proba >= threshold).astype(int)

    # Naive baseline: binary rule for precision/recall, client_past_late_ratio as a
    # continuous risk score (what a freelancer would actually rank invoices by) for ROC-AUC.
    naive_preds_all = naive_baseline_preds(resolved)
    naive_preds = naive_preds_all.loc[test["invoice_id"]].values
    naive_score = test["client_past_late_ratio"].values

    # Majority-class baseline: always predict whatever label was most common in training.
    majority_label = int(y_train.mode().iloc[0])
    majority_preds = np.full(len(y_test), majority_label)

    return {
        "split_sizes": split_sizes,
        "threshold": threshold,
        "test": test,
        "y_test": y_test,
        "test_proba": test_proba,
        "tabpfn_preds": tabpfn_preds,
        "naive_preds": naive_preds,
        "naive_score": naive_score,
        "majority_preds": majority_preds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV with invoice_id, client, amount_eur, issue_date, due_date, paid_date (default: synthetic data)",
    )
    args = parser.parse_args()

    r = run_backtest(args.data)
    print(r["split_sizes"])
    print(f"chosen decision threshold (from validation, max balanced accuracy): {r['threshold']:.2f}")

    report("TabPFN", r["y_test"], r["tabpfn_preds"])
    print(f"  fit={r['fit_seconds']:.2f}s  predict={r['predict_seconds']:.2f}s")

    report("Naive baseline (late iff client's previous invoice was late)", r["y_test"], r["naive_preds"])

    majority_label = int(r["majority_preds"][0])
    report(f"Majority class (always predict {majority_label})", r["y_test"], r["majority_preds"])

    summary = pd.DataFrame(
        [
            summary_row("Majority class", r["y_test"], r["majority_preds"], None),
            summary_row("Naive baseline", r["y_test"], r["naive_preds"], r["naive_score"]),
            summary_row("TabPFN", r["y_test"], r["tabpfn_preds"], r["test_proba"]),
        ]
    ).set_index("model")
    print("\nSummary:")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
