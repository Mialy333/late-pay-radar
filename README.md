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

## Agent

A small Strands Agents agent, backed by a local Gemma model via Ollama
(`gemma4:e2b`, same setup as the Week 1 project), that explains ONE
invoice's risk and drafts ONE reminder for it. Ranking across many invoices
is deliberately NOT the agent's job — `list_open_invoices()` sorts and caps
to `top_n` in code (for a future code-computed UI table), so the small
model never has to sort or pick from a long list itself.

- `late_pay_radar/scoring.py` — hybrid router: fewer than 3 prior invoices
  for a client → TabPFN's predicted probability (confidence "low"); 3+ →
  `client_past_late_ratio` directly (confidence "normal"). Matches what
  `scripts/cold_start_analysis.py` found actually ranks better in each case.
  TabPFN is trained once on resolved Barnet invoices and cached in-process.
  Every result also carries `currency`, a locale-formatted `amount_display`,
  `due_status` ("not yet due, due in N days" / "overdue by N days") and
  `as_of_date` — computed from the invoice's own issue date (for
  `data/my_invoice.csv`) or the Barnet report's own cutoff date (for Barnet
  invoices), never from today's real date — plus a `training_data_note`
  whenever the TabPFN fallback fires.
- `late_pay_radar/tools.py` — two Strands tools: `list_open_invoices(top_n=5)`
  (pre-sorted by risk, tie-broken by amount, with an explicit `rank`) and
  `get_invoice_risk(invoice_id)`. Only the latter is given to the agent.
  Both combine open Barnet invoices with the user's own
  `data/my_invoice.csv` (scored as of its issue date, ignoring any
  `paid_date` already on file).
- `late_pay_radar/agent.py` — system prompt: explain risk in 2-3 plain
  sentences using only tool-returned numbers; draft a friendly pre-due
  reminder or a firm overdue one depending on `due_status`, using
  `amount_display` as-is. Never invent figures.

```bash
ollama pull gemma4:e2b   # if not already present
uv run python scripts/agent_demo.py
```

Runs 2 prompts (one invoice in French, one in English) and, per run: logs
tools called, their arguments, and total latency; deterministically appends
`training_data_note` as a `Note:` line when present (never left to the
model); and validates each drafted reminder actually contains the real
amount and due date, sending one corrective follow-up if not (logged as
`retried: True/False`, with a warning if still invalid after that retry).

## UI

```bash
uv run streamlit run app.py
```

One page (`app.py`): pick a data source (the Barnet demo, or your own data —
upload a CSV, or one or more **text-based** PDF invoices), see the full
ranked table (rank, amount, due status, risk %, source, confidence —
computed entirely in code, no LLM; low-confidence rows highlighted), then
pick one invoice and a language to get an explanation and a validated,
copyable reminder from the agent.

**PDF import** (`late_pay_radar/pdf_import.py`): text is extracted locally
with `pdfplumber`; scanned/image-only PDFs (no extractable text) are
rejected with a clear message rather than guessed at. The extracted text is
sent to the same local Gemma model to pull out `invoice_id, client, amount,
currency, issue_date, due_date, paid_date` as JSON; results are validated
(dates parse, amount > 0, due_date ≥ issue_date), with one retry on
failure and only the still-invalid fields blanked (not the whole record) if
that retry doesn't fix it. Nothing is scored until the user reviews the
extracted fields in an editable table and clicks **Confirm** — and nothing
is ever written to disk, for either CSVs or PDFs; everything stays in the
in-memory upload buffer / Streamlit session state.

**Privacy note:** checked whether `tabpfn` makes any network call at
inference time now that its weights are cached — it doesn't. The package
has no telemetry code path at all; its only network calls (model download +
one-time license check) are gated behind a local-file-exists check, so once
weights are cached they never run. Verified empirically by forcing all
network access to fail (bogus proxy + `HF_HUB_OFFLINE=1`) and confirming
`fit`/`predict` still succeed. Separately, Streamlit's own anonymous usage
telemetry is disabled via `.streamlit/config.toml`
(`browser.gatherUsageStats = false`) — unrelated to invoice data, but
disabled anyway since the app claims to run entirely on your machine.
