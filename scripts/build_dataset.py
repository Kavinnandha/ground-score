"""Build the committed working corpus for the chosen brand.

Produces data/processed/threads.jsonl -- the file `make reproduce` reads. The
raw 516MB CSV is never needed again after this runs.

The split is the important part. Threads are assigned to `history` (retrieval
corpus) or `golden_pool` (evaluation) by a stable hash of the thread id, so:
  * the assignment is identical on every machine and every run,
  * it does not shift when the sample size or brand list changes,
  * and no evaluation thread can ever enter the retrieval index, which would
    let the agent retrieve the exact conversation it is being scored on.

tests/test_pipeline.py asserts that last property directly.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundscore import config  # noqa: E402
from groundscore.cleaning import (  # noqa: E402
    dedupe_key, is_probably_english, is_usable_customer_message, stable_bucket,
)
from groundscore.ingest import build_threads  # noqa: E402


def main() -> int:
    cfg = config.brand_config()
    brand = cfg["brand"]
    corpus_cfg, split_cfg = cfg["corpus"], cfg["split"]
    seed = corpus_cfg["seed"]
    golden_buckets = split_cfg["golden_pool_buckets"]

    if not config.RAW_CSV.exists():
        print(f"Missing {config.RAW_CSV}. Run: python scripts/download_data.py")
        return 1

    print(f"Building corpus for {brand} (cap {corpus_cfg['max_threads']} threads)...")
    raw = list(build_threads(
        [brand], config.RAW_CSV,
        max_threads_per_brand=corpus_cfg["max_threads"], seed=seed,
    ))
    print(f"  {len(raw)} raw threads")

    kept: list[dict] = []
    seen_keys: set[str] = set()
    dropped = Counter()

    for thread in raw:
        msg = thread["customer_msg"]
        if not is_usable_customer_message(msg):
            dropped["unusable_customer_message"] += 1
            continue
        # Language filter. The Latin-script check above does NOT catch this:
        # Spanish, French, German and Portuguese are all Latin-script, and
        # clustering the corpus without this filter produced four clusters that
        # were languages rather than intents. Multilingual support is explicitly
        # out of scope (see REPORT.md), so this traffic is removed rather than
        # served badly.
        if not is_probably_english(msg):
            dropped["not_english"] += 1
            continue
        if not thread["brand_replies"] or not thread["brand_replies"][0]["text"].strip():
            dropped["empty_brand_reply"] += 1
            continue
        key = dedupe_key(msg)
        if key in seen_keys:
            dropped["near_duplicate"] += 1
            continue
        seen_keys.add(key)

        bucket = stable_bucket(f"split:{thread['thread_id']}", 100)
        thread["split"] = (
            config.SPLIT_GOLDEN_POOL if bucket < golden_buckets else config.SPLIT_HISTORY
        )
        thread["bucket"] = bucket
        kept.append(thread)

    kept.sort(key=lambda t: int(t["thread_id"]))  # deterministic file order

    config.THREADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.THREADS_PATH.open("w", encoding="utf-8") as fh:
        for thread in kept:
            fh.write(json.dumps(thread, ensure_ascii=False) + "\n")

    splits = Counter(t["split"] for t in kept)
    size_mb = config.THREADS_PATH.stat().st_size / 1e6
    print(f"  dropped: {dict(dropped)}")
    print(f"  kept {len(kept)} threads -> {config.THREADS_PATH} ({size_mb:.1f} MB)")
    print(f"  split: {dict(splits)}")

    summary = {
        "brand": brand,
        "raw_threads": len(raw),
        "kept_threads": len(kept),
        "dropped": dict(dropped),
        "splits": dict(splits),
        "seed": seed,
        "file_mb": round(size_mb, 2),
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
