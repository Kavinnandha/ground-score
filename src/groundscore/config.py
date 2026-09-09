"""Single place that resolves config and canonical paths.

Every module reads paths from here rather than recomputing `parents[2]`, so
moving a file never silently splits the pipeline across two directories.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "configs"
BRAND_CONFIG = CONFIG_DIR / "brand.yaml"
THRESHOLDS_CONFIG = CONFIG_DIR / "thresholds.yaml"
TAXONOMY_PATH = REPO_ROOT / "taxonomy" / "intents.yaml"

DATA_DIR = REPO_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "twcs.csv"
THREADS_PATH = DATA_DIR / "processed" / "threads.jsonl"
GOLDEN_DIR = DATA_DIR / "golden"
GOLDEN_PATH = GOLDEN_DIR / "golden_v1.jsonl"
GOLDEN_CANDIDATES_PATH = GOLDEN_DIR / "candidates.jsonl"
HUMAN_JUDGE_PATH = GOLDEN_DIR / "human_reply_scores.jsonl"

RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache"

SPLIT_HISTORY = "history"
SPLIT_GOLDEN_POOL = "golden_pool"


@lru_cache(maxsize=None)
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def brand_config() -> dict[str, Any]:
    return load_yaml(BRAND_CONFIG)


def brand() -> str:
    return brand_config()["brand"]


def thresholds() -> dict[str, Any]:
    """Routing thresholds. Absent until tuned on the dev split.

    Defaults are deliberately conservative (escalate more) so that a run made
    before tuning cannot silently look better than the tuned one.
    """
    try:
        return load_yaml(THRESHOLDS_CONFIG)
    except FileNotFoundError:
        return {"tau_confidence": 0.9, "tau_similarity": 0.75, "tuned": False}
