"""Download the 2 most recent Barnet Council period files into data/raw/
(gitignored -- not checked in, so a fresh clone needs this step before
scripts/build_barnet_dataset.py can run).

Source: London Borough of Barnet, "Financial Invoices Late Payments" --
https://open.barnet.gov.uk/dataset/2g58n
Licence: Open Government Licence v3.0 (OGL). No login required -- these
are direct public download links.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = {
    "barnet_invoices_2025-06_2025-11.xlsx": (
        "https://open.barnet.gov.uk/download/2g58n/w3g/"
        "Late%20Payment%20of%20Finance%20Invoice%2001.06.25%20-%2030.11.25.xlsx"
    ),
    "barnet_invoices_2025-12_2026-05.xlsx": (
        "https://open.barnet.gov.uk/download/2g58n/dyw/"
        "Late%20Payment%20of%20Finance%20Invoices%20from%2001.12.25%20-31.05.26.xlsx"
    ),
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"already have {dest}, skipping")
            continue
        print(f"downloading {filename} ...")
        try:
            # The server rejects urllib's default User-Agent (403); a plain browser-like one works.
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response, open(dest, "wb") as f:  # noqa: S310
                f.write(response.read())
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED to download {filename}: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
