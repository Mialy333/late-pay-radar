# late-pay-radar

Ranks your open invoices by late-payment risk and drafts the right
reminder — entirely on your own machine (TabPFN + a local Gemma model via
Ollama). Demo video: <https://youtu.be/sccG2NXpBWw>

## Quick start

```bash
git clone https://github.com/Mialy333/late-pay-radar && cd late-pay-radar
uv sync
ollama pull gemma4:e2b
uv run python scripts/download_barnet.py && uv run python scripts/build_barnet_dataset.py
uv run streamlit run app.py
```

Before that last command, you need a `TABPFN_TOKEN` in `.env` — see the next section.

## TabPFN account (one-time)

The risk model (TabPFN) needs a one-time license acceptance before it will
run locally, even for the demo data:

1. Create a free account at <https://ux.priorlabs.ai>.
2. Accept the TabPFN-3.5 license (non-commercial use) on the Licenses tab.
3. Copy your API key from <https://ux.priorlabs.ai/account>.
4. `cp .env.example .env` and put the key in as `TABPFN_TOKEN=<your key>`.

This is a one-time step: once accepted, it's cached locally and no further
account/network interaction is needed (see "Why local?" below).

## Why local?

- **TabPFN**: trains and predicts entirely on your machine. Its only
  network calls are the one-time model download + license check above,
  gated behind a local-file-exists check — once its weights are cached,
  it never touches the network again. Verified empirically, not just by
  reading the source: ran `fit`/`predict` with all network access forced
  to fail (a bogus proxy + `HF_HUB_OFFLINE=1`) and got the identical
  result, same speed, as with network available.
- **Gemma**: runs entirely through your local Ollama install
  (`gemma4:e2b`) — no API key, no cloud call.
- **Your invoice data**: uploaded CSVs and PDFs are read from the
  in-memory upload buffer and never written to disk.
- **Streamlit's own anonymous usage telemetry** is disabled via
  `.streamlit/config.toml` (`browser.gatherUsageStats = false`) —
  unrelated to invoice data, but disabled anyway since this app claims to
  run entirely on your machine.

## Known limitations

- TabPFN (used for brand-new clients with no payment history) is trained
  on UK public-sector supplier invoices (Barnet Council, 77% paid late) —
  not representative of most freelance clients. Treat its score as
  indicative, not calibrated.
- The "TabPFN beats client history for cold-start clients" finding (see
  `scripts/cold_start_analysis.py`) is from one backtest on one dataset —
  indicative, not a guarantee it holds for your own clients.
- PDF import extracts a `currency` field for your review, but it isn't
  currently wired into `amount_display` formatting, which still infers
  GBP-vs-EUR from the invoice ID prefix (a holdover from the Barnet-only
  design). Imported invoices always display in EUR formatting regardless
  of the PDF's stated currency.
- Scanned (image-only) PDFs aren't supported — only text-based PDFs.
- Each "Explain & draft reminder" click takes roughly 10-15s on a MacBook
  (two local Gemma calls, plus a validation retry when needed).

## Data attribution

- **Barnet Council invoices**: London Borough of Barnet, "Financial
  Invoices Late Payments" — <https://open.barnet.gov.uk/dataset/2g58n>,
  licensed under the Open Government Licence v3.0 (OGL). Contains public
  sector information licensed under the Open Government Licence v3.0.
- **Synthetic invoices**: seeded random data, never real.

## License

MIT (see `LICENSE`) for the code in this repository. The Barnet Council
data under `data/` is licensed separately under the OGL v3.0 above.

## Data

### Synthetic (default)

`uv run python scripts/make_synthetic.py` generates `data/synthetic_invoices.csv`
— 80 fully synthetic invoices across 8 clients with different seeded payment
habits. Never real data.

### Barnet Council (real, open data)

```bash
uv run python scripts/download_barnet.py      # fetches the 2 raw period files into data/raw/ (gitignored)
uv run python scripts/build_barnet_dataset.py  # builds data/barnet_invoices.csv from them
```

See "Data attribution" above for source/license. `data/barnet_invoices.csv`
itself is committed (so the demo works out of the box), but `data/raw/`
isn't — these two scripts are what make the pipeline reproducible from
scratch.

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

Requires a `TABPFN_TOKEN` in `.env` — see "TabPFN account" above.

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
in-memory upload buffer / Streamlit session state. See "Why local?" at the
top for the privacy/network story.
