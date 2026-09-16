"""Render docs/report.html to docs/ground-score-report.pdf.

Chrome's headless print is the whole toolchain here. The alternatives all pull
in a dependency (weasyprint, reportlab, a LaTeX install) that `make reproduce`
would then have to carry, and the report is not part of the reproduction path.

The HTML source is kept out of the repository (it is gitignored); the committed
PDF is the deliverable. Every number in it is read off a committed file in
results/, so the PDF can be checked against the repo without the source.

    python scripts/build_report.py
    python scripts/build_report.py --browser "C:/path/to/chrome.exe"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "report.html"
OUTPUT = ROOT / "docs" / "ground-score-report.pdf"

# Chromium-family binaries, in the order worth trying. Edge is on every Windows
# box and prints identically, which saves a reviewer installing anything.
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    for name in ("google-chrome", "chromium", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in CANDIDATES:
        if Path(path).exists():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default=None, help="path to a Chrome/Edge binary")
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"missing {SOURCE}")
        return 1

    browser = find_browser(args.browser)
    if not browser:
        print("No Chrome or Edge found. Pass --browser, or open docs/report.html")
        print("in any browser and print to PDF (A4, no headers/footers).")
        return 1

    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={OUTPUT}", SOURCE.as_uri()],
        check=True,
    )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
