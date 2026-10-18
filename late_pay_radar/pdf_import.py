"""Import an invoice from a text-based PDF: extract text locally with
pdfplumber, then ask the local Gemma model (same one the agent uses, via
Ollama) to pull out structured fields as JSON. Nothing here ever touches
disk -- PDFs are read from the in-memory upload buffer, and extraction
results are only held in memory until the user confirms them in the UI.

Scanned (image-only) PDFs have no extractable text and are rejected with a
clear message rather than silently producing garbage.
"""

from __future__ import annotations

import json
from typing import Any, BinaryIO

import ollama
import pdfplumber

from late_pay_radar.agent import MODEL_ID, OLLAMA_HOST

MIN_EXTRACTABLE_CHARS = 20
SCANNED_PDF_MESSAGE = "Scanned PDFs aren't supported yet, please use a text PDF or CSV."

EXTRACTION_FIELDS = ["invoice_id", "client", "amount", "currency", "issue_date", "due_date", "paid_date"]

EXTRACTION_SYSTEM_PROMPT = """You extract structured data from invoice text.

Return ONLY a JSON object with exactly these keys: invoice_id, client,
amount, currency, issue_date, due_date, paid_date.

Rules:
- Dates must be ISO format YYYY-MM-DD.
- amount must be a plain number: no currency symbol, no thousands separator.
- currency must be a 3-letter code (e.g. EUR, GBP, USD).
- paid_date is null if the invoice is unpaid or the date isn't stated.
- Any other field you truly cannot find is null. Never invent a value.
- Return ONLY the JSON object -- no explanation, no markdown fences.
"""


def extract_pdf_text(file: BinaryIO) -> str:
    """Extract all text from a PDF file-like object. Never writes to disk."""
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def has_extractable_text(text: str) -> bool:
    return len(text.strip()) >= MIN_EXTRACTABLE_CHARS


def _parse_date(value: Any) -> str | None:
    if not value or not isinstance(value, str):
        return None
    try:
        import datetime

        return datetime.date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        return None


def validate_extracted(data: dict) -> dict[str, str | None]:
    """Per-field validation errors (None = that field is fine)."""
    errors: dict[str, str | None] = dict.fromkeys(EXTRACTION_FIELDS)

    issue_date = _parse_date(data.get("issue_date"))
    errors["issue_date"] = None if issue_date else "could not parse as YYYY-MM-DD"

    due_date = _parse_date(data.get("due_date"))
    errors["due_date"] = None if due_date else "could not parse as YYYY-MM-DD"

    try:
        amount = float(data.get("amount"))
        errors["amount"] = None if amount > 0 else "must be greater than 0"
    except (TypeError, ValueError):
        errors["amount"] = "not a valid number"

    if errors["issue_date"] is None and errors["due_date"] is None and due_date < issue_date:  # type: ignore[operator]
        errors["due_date"] = "due_date is before issue_date"

    return errors


def has_errors(errors: dict[str, str | None]) -> bool:
    return any(v is not None for v in errors.values())


def blank_invalid_fields(data: dict, errors: dict[str, str | None]) -> dict:
    cleaned = dict(data)
    for field, err in errors.items():
        if err is not None:
            cleaned[field] = None
    return cleaned


def _ask_gemma_for_json(pdf_text: str, retry_note: str | None = None) -> dict:
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Invoice text:\n\n{pdf_text}"},
    ]
    if retry_note:
        messages.append({"role": "user", "content": retry_note})
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(model=MODEL_ID, messages=messages, format="json")
    return json.loads(response["message"]["content"])


def extract_invoice(pdf_text: str) -> tuple[dict, dict[str, str | None], bool]:
    """Extract + validate, retrying once on failure, then falling back to
    blanking only the fields that didn't validate (never the whole record).

    Returns (data, field_errors, retried).
    """
    try:
        data = _ask_gemma_for_json(pdf_text)
    except Exception:  # noqa: BLE001
        data = dict.fromkeys(EXTRACTION_FIELDS)

    errors = validate_extracted(data)
    if not has_errors(errors):
        return data, errors, False

    retry_note = "Your previous JSON had problems: " + "; ".join(
        f"{field} ({err})" for field, err in errors.items() if err
    )
    retry_note += ". Return corrected JSON only, with the same keys."
    try:
        data = _ask_gemma_for_json(pdf_text, retry_note=retry_note)
    except Exception:  # noqa: BLE001
        pass
    errors = validate_extracted(data)
    return blank_invalid_fields(data, errors), errors, True
