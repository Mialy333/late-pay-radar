"""Late Pay Radar -- one-page Streamlit UI.

The ranked table is computed entirely in code (no LLM). The agent is only
invoked per-invoice, on demand, to explain risk and draft a reminder.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from late_pay_radar.agent import build_agent
from late_pay_radar.pdf_import import (
    EXTRACTION_FIELDS,
    SCANNED_PDF_MESSAGE,
    extract_invoice,
    extract_pdf_text,
    has_errors,
    has_extractable_text,
)
from late_pay_radar.reminder_check import reminder_is_valid
from late_pay_radar.scoring import rank_open_invoices
from late_pay_radar.tools import CALL_LOG, set_active_loader

BARNET_DATA_PATH = Path(__file__).resolve().parent / "data" / "barnet_invoices.csv"
EXPECTED_COLUMNS = ["invoice_id", "client", "amount_eur", "issue_date", "due_date", "paid_date"]

st.set_page_config(page_title="Late Pay Radar", page_icon="📡")

st.title("Late Pay Radar")
st.caption(
    "Ranks your open invoices by late-payment risk and drafts the right reminder. "
    "Runs on your machine: TabPFN + Gemma via Ollama."
)

source = st.radio(
    "Data",
    ["Demo: Barnet Council open invoices (OGL v3.0)", "My invoices (upload CSV or PDF)"],
)

raw_df: pd.DataFrame | None = None

if source.startswith("Demo"):
    raw_df = pd.read_csv(BARNET_DATA_PATH, parse_dates=["issue_date", "due_date", "paid_date"])
    set_active_loader(lambda: raw_df)
else:
    st.markdown(
        "**CSV expected columns:** `invoice_id, client, amount_eur, issue_date, due_date, paid_date` "
        "(dates as `YYYY-MM-DD`; leave `paid_date` empty for invoices not yet paid). "
        "Or upload one or more **text-based** PDF invoices to extract fields automatically "
        "(scanned/image-only PDFs aren't supported)."
    )
    st.dataframe(
        pd.DataFrame(
            [{"invoice_id": "INV-001", "client": "Acme Corp", "amount_eur": 1500.0,
              "issue_date": "2026-01-01", "due_date": "2026-01-31", "paid_date": ""}]
        ),
        hide_index=True,
    )
    uploaded_files = st.file_uploader(
        "Upload CSV or PDF invoice(s)", type=["csv", "pdf"], accept_multiple_files=True
    ) or []
    csv_files = [f for f in uploaded_files if f.name.lower().endswith(".csv")]
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]

    if csv_files:
        try:
            # Read straight from the in-memory upload buffer -- never written to disk.
            df = pd.read_csv(csv_files[0])
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing:
                st.error(
                    f"Missing column(s): {', '.join(missing)}. "
                    f"Expected: {', '.join(EXPECTED_COLUMNS)}."
                )
            else:
                df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
                df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
                df["paid_date"] = pd.to_datetime(df["paid_date"], errors="coerce")
                df["amount_eur"] = pd.to_numeric(df["amount_eur"], errors="coerce")
                bad_dates = df["issue_date"].isna() | df["due_date"].isna()
                if bad_dates.any():
                    st.error(
                        f"Could not parse issue_date/due_date on {int(bad_dates.sum())} row(s). "
                        "Use YYYY-MM-DD."
                    )
                elif df["amount_eur"].isna().any():
                    st.error("amount_eur must be numeric on every row.")
                else:
                    raw_df = df
                    set_active_loader(lambda: raw_df)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read that CSV: {exc}")

    elif pdf_files:
        # Only re-run extraction (LLM calls) when the uploaded PDF set actually changes --
        # not on every Streamlit rerun (e.g. clicking Confirm).
        pdf_signature = tuple((f.name, f.size) for f in pdf_files)
        if st.session_state.get("pdf_signature") != pdf_signature:
            st.session_state["pdf_signature"] = pdf_signature
            st.session_state.pop("confirmed_pdf_df", None)
            rows = []
            for i, f in enumerate(pdf_files):
                text = extract_pdf_text(f)  # read from the upload buffer, never saved to disk
                if not has_extractable_text(text):
                    st.error(f"{f.name}: {SCANNED_PDF_MESSAGE}")
                    continue
                with st.spinner(f"Extracting {f.name} with Gemma..."):
                    data, errors, retried = extract_invoice(text)
                row = {field: data.get(field) for field in EXTRACTION_FIELDS}
                if not row.get("invoice_id"):
                    row["invoice_id"] = f"PDF-{i + 1:03d}"
                row["source_file"] = f.name
                row["needs_review"] = has_errors(errors)
                rows.append(row)
            st.session_state["pdf_rows"] = rows

        rows = st.session_state.get("pdf_rows", [])
        if rows:
            if any(r["needs_review"] for r in rows):
                st.warning(
                    "Some fields couldn't be reliably extracted -- please check and fix "
                    "them below (see the needs_review column) before confirming."
                )
            edited = st.data_editor(
                pd.DataFrame(rows),
                num_rows="dynamic",
                use_container_width=True,
                key="pdf_editor",
                hide_index=True,
            )
            if st.button("Confirm"):
                df = edited.rename(columns={"amount": "amount_eur"}).copy()
                df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
                df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
                df["paid_date"] = pd.to_datetime(df["paid_date"], errors="coerce")
                df["amount_eur"] = pd.to_numeric(df["amount_eur"], errors="coerce")
                bad = df["issue_date"].isna() | df["due_date"].isna() | df["amount_eur"].isna()
                if bad.any():
                    st.error(
                        f"{int(bad.sum())} row(s) still have missing/invalid issue_date, "
                        "due_date, or amount -- please fix before confirming."
                    )
                else:
                    st.session_state["confirmed_pdf_df"] = df[EXPECTED_COLUMNS].copy()
                    st.success("Confirmed.")

    if "confirmed_pdf_df" in st.session_state:
        raw_df = st.session_state["confirmed_pdf_df"]
        set_active_loader(lambda: raw_df)

if raw_df is not None and len(raw_df) > 0:
    scored = rank_open_invoices(raw_df)

    if not scored:
        st.info("No open invoices in this dataset.")
    else:
        st.caption(f"As of {scored[0]['as_of_date']}")

        table = pd.DataFrame(
            [
                {
                    "rank": i + 1,
                    "invoice": r["invoice_id"],
                    "amount": r["amount_display"],
                    "due_status": r["due_status"],
                    "risk": f"{r['risk'] * 100:.0f}%",
                    "source": r["source"],
                    "confidence": r["confidence"],
                }
                for i, r in enumerate(scored)
            ]
        )

        def _highlight_low_confidence(row: pd.Series) -> list[str]:
            style = "background-color: #fff3cd" if row["confidence"] == "low" else ""
            return [style] * len(row)

        st.dataframe(
            table.style.apply(_highlight_low_confidence, axis=1),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Explain & draft reminder")
        invoice_ids = [r["invoice_id"] for r in scored]
        selected_id = st.selectbox("Invoice", invoice_ids)
        language = st.radio("Language", ["Français", "English"], horizontal=True)

        if st.button("Explain & draft reminder"):
            lang_word = "French" if language == "Français" else "English"
            with st.spinner("Asking the agent..."):
                agent = build_agent()
                CALL_LOG.clear()
                start = time.perf_counter()

                explanation = str(agent(f"Explain the risk on invoice {selected_id}."))
                reminder = str(agent(f"Now draft a reminder for this invoice in {lang_word}."))

                invoice = next(
                    (c["result"] for c in CALL_LOG if c["tool"] == "get_invoice_risk" and c.get("result")),
                    None,
                )
                retried = False
                still_invalid = False
                if invoice is not None and not reminder_is_valid(
                    reminder, invoice["amount_display"], invoice["due_date"]
                ):
                    followup = (
                        f"Your reminder must include the amount {invoice['amount_display']} "
                        f"and the due date {invoice['due_date']}. Rewrite only the reminder."
                    )
                    reminder = str(agent(followup))
                    retried = True
                    still_invalid = not reminder_is_valid(
                        reminder, invoice["amount_display"], invoice["due_date"]
                    )

                total_seconds = time.perf_counter() - start

            st.markdown("**Explanation**")
            st.write(explanation)

            if invoice is not None and invoice.get("training_data_note"):
                st.info(invoice["training_data_note"])
            if still_invalid:
                st.warning("Reminder may still be missing the exact amount or due date after 1 retry.")

            st.markdown("**Reminder** (copyable)")
            st.code(reminder, language=None)

            tools_called = ", ".join(c["tool"] for c in CALL_LOG)
            st.caption(
                f"Tools called: {tools_called or 'none'} · Retried: {retried} · "
                f"Latency: {total_seconds:.1f}s"
            )
