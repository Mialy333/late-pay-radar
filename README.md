# late-pay-radar

Backtest spike: can issue-time-only features beat a naive "late iff the
client's previous invoice was late" baseline at predicting invoices that
will be paid more than 7 days late?

## Data

### Synthetic (default)

`uv run python scripts/make_synthetic.py` generates `data/synthetic_invoices.csv`
— 80 fully synthetic invoices across 8 clients with different seeded payment
habits. Never real data.

### Barnet Council (real, open data)

`uv run python scripts/build_barnet_dataset.py` builds `data/barnet_invoices.csv`
from the two most recent period files under `data/raw/` (gitignored — re-download
from the source below if you need them).

- **Source:** London Borough of Barnet, "Financial Invoices Late Payments" —
  <https://open.barnet.gov.uk/dataset/2g58n>
- **Licence:** Open Government Licence v3.0 (OGL). Contains public sector
  information licensed under the Open Government Licence v3.0.
- **Attribution:** Data provided by the London Borough of Barnet under the
  Open Government Licence v3.0.
- **Caveat:** `amount_eur` in this dataset is actually GBP (Barnet's "Invoice
  Amount"), kept under that column name only for schema compatibility with
  the synthetic data — see `late_pay_radar/adapters/barnet.py`.

## Running the backtest

```bash
uv run python scripts/backtest.py                           # synthetic data
uv run python scripts/backtest.py --data data/barnet_invoices.csv  # Barnet data
```

Compares three models on a time-based split (oldest ~70% of resolved
invoices train, most recent test; training capped at 3,000 rows, keeping the
most recent, for speed):

- **Majority class** — always predicts whichever label was most common in training.
- **Naive baseline** — predicts "late" iff the client's previous invoice was late;
  for ROC-AUC, its score is `client_past_late_ratio` (the ranking signal a
  freelancer would actually use to triage invoices).
- **TabPFN** — fit on issue-time features (log amount, payment terms, client
  history, month/weekday, and for Barnet: supplier type, invoice source).
  Its decision threshold is chosen on a validation slice carved from the
  last 20% of the training set (maximizing balanced accuracy) — the test set
  is never used for tuning. ROC-AUC uses `predict_proba` directly and so is
  threshold-independent.

Prints a confusion matrix per model plus a summary table (accuracy, balanced
accuracy, ROC-AUC, precision, recall).

Requires a `TABPFN_TOKEN` in `.env` (see `.env.example`) — get one at
<https://ux.priorlabs.ai/account> after accepting the TabPFN licence.
