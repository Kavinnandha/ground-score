"""Fetch the Kaggle 'Customer Support on Twitter' dataset into data/raw/.

Only needed for `make full`. `make reproduce` runs off the committed subsample
in data/processed/ and never touches this.

Verified 2026-09: this dataset is served by the Kaggle API *without*
authentication, so a reviewer needs no Kaggle account to re-run the full
pipeline. Token auth is still attempted as a fallback in case that changes:
KAGGLE_API_TOKEN (bearer) or KAGGLE_USERNAME + KAGGLE_KEY (basic), read from
the environment or from .env.

Uses the REST API directly rather than the `kaggle` CLI package, which
authenticates at module import time and fails noisily when no credentials exist.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
TARGET = RAW_DIR / "twcs.csv"

DATASET = "thoughtvector/customer-support-on-twitter"
API_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"

MANUAL = f"""
Automatic download failed.

Option A -- manual download:
  1. https://www.kaggle.com/datasets/{DATASET}
  2. "Download" (~500 MB zip), extract twcs.csv
  3. Place it at: {TARGET}

Option B -- credentials, if Kaggle has started requiring them:
  1. https://www.kaggle.com/settings/account -> API -> "Create New Token"
  2. Put KAGGLE_USERNAME and KAGGLE_KEY in .env (see .env.example)
  3. Re-run: python scripts/download_data.py

Option C -- skip it entirely:
  `make reproduce` does not need this file. It replays the committed
  subsample in data/processed/ and the committed caches in cache/.
"""


def load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def auth_headers() -> list[tuple[str, dict[str, str]]]:
    """Auth strategies to try, in order. Anonymous first -- it works today."""
    load_dotenv()
    strategies: list[tuple[str, dict[str, str]]] = [("anonymous", {})]

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        strategies.append(("bearer token", {"Authorization": f"Bearer {token}"}))

    user, key = os.environ.get("KAGGLE_USERNAME", "").strip(), os.environ.get("KAGGLE_KEY", "").strip()
    if not (user and key):
        for candidate in (
            Path.home() / ".kaggle" / "kaggle.json",
            Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json",
        ):
            try:
                if candidate.is_file():
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    user, key = data.get("username", ""), data.get("key", "")
                    break
            except (OSError, json.JSONDecodeError):
                continue
    if user and key:
        token64 = base64.b64encode(f"{user}:{key}".encode()).decode()
        strategies.append(("basic auth", {"Authorization": f"Basic {token64}"}))

    return strategies


def download(headers: dict[str, str], label: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "customer-support-on-twitter.zip"
    request = urllib.request.Request(API_URL, headers=headers)

    print(f"Downloading {DATASET} ({label})...", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, zip_path.open("wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            elif done % (50 << 20) < (1 << 20):
                print(f"\r  {done / 1e6:7.1f} MB", end="", flush=True)
    print()

    # A Kaggle auth failure returns an HTML page with HTTP 200, not an error.
    with zip_path.open("rb") as fh:
        if fh.read(4) != b"PK\x03\x04":
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"{label}: server returned a non-zip payload (likely an auth wall)")
    return zip_path


def main() -> int:
    if TARGET.exists():
        print(f"Already present: {TARGET} ({TARGET.stat().st_size / 1e6:.0f} MB)")
        return 0

    zip_path = None
    for label, headers in auth_headers():
        try:
            zip_path = download(headers, label)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
            print(f"  {label} failed: {exc}")
    if zip_path is None:
        print(MANUAL)
        return 1

    print("Extracting twcs.csv...")
    with zipfile.ZipFile(zip_path) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith("twcs.csv")), None)
        if name is None:
            print(f"twcs.csv not found in archive. Contents: {zf.namelist()}")
            return 1
        with zf.open(name) as src, TARGET.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    zip_path.unlink(missing_ok=True)
    print(f"Ready: {TARGET} ({TARGET.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
