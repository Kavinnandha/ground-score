"""Measure candidate brands so the brand choice is evidence-based.

Why bother profiling instead of just picking the biggest brand
--------------------------------------------------------------
The agent's second job is "draft a reply grounded in how that brand has
historically resolved similar issues". That task is only meaningful if the
brand's historical replies actually *contain* resolutions. Several of the
largest brands in this dataset answer nearly every public tweet with some
variant of "please DM us" -- for those, a perfectly-imitating agent learns to
emit deflections, and every reply-quality metric becomes a measure of how well
the model reproduces a non-answer.

So the deciding statistic is the handoff rate, not the volume. The other
columns guard against picking a low-handoff brand that is useless for other
reasons (too few threads, one-line replies, no topical variety).

A correction worth recording: the first version of this profiler measured only
DM-style deflection and reported AmazonHelp at 0.008. That undercounted by 13x,
because AmazonHelp rarely says "DM us" and instead routes people to a contact
page. Both forms are now measured separately, and `handoff_rate` -- the union --
is the column the brand decision rests on.

Columns
-------
threads              usable (customer opener -> brand reply) threads in sample
dm_deflection_rate   share of FIRST replies pushing the customer into DMs
link_handoff_rate    share routing to a contact page or phone line instead
handoff_rate         union of the two: "not resolved in this channel"
median_reply_chars   length of the first brand reply
substantive_rate     share of first replies that are non-handoff AND >=80
                     chars -- a proxy for "contains an actual instruction"
multi_turn_rate      share of threads where the customer replied again, i.e.
                     the brand's answer did not end the conversation
lexical_diversity    distinct-token / total-token ratio over customer openers,
                     a cheap proxy for intent variety (a brand whose inbound is
                     all "where is my order" makes for a trivial taxonomy)
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from .cleaning import is_deflection, is_handoff, is_link_handoff, is_usable_customer_message
from .ingest import RAW_CSV, build_threads

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

# Chosen from the outbound-volume ranking: the five highest-volume brands plus
# two mid-volume support accounts whose domains (games, streaming) plausibly
# have troubleshooting rather than logistics answers.
CANDIDATES = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "XboxSupport",
    "hulu_support",
]

SAMPLE_PER_BRAND = 4000


def profile(
    brands: list[str] | None = None,
    csv_path: Path = RAW_CSV,
    sample_per_brand: int = SAMPLE_PER_BRAND,
) -> list[dict]:
    brands = brands or CANDIDATES
    buckets: dict[str, list[dict]] = {b.lower(): [] for b in brands}

    for thread in build_threads(brands, csv_path, max_threads_per_brand=sample_per_brand):
        buckets[thread["brand"]].append(thread)

    rows = []
    for brand in brands:
        threads = buckets[brand.lower()]
        usable = [t for t in threads if is_usable_customer_message(t["customer_msg"])]
        if not usable:
            rows.append({"brand": brand, "threads": 0})
            continue

        first_replies = [t["brand_replies"][0]["text"] for t in usable]
        deflections = [is_deflection(r) for r in first_replies]
        handoffs = [is_handoff(r) for r in first_replies]
        lengths = [len(r) for r in first_replies]

        tokens = [w for t in usable for w in t["customer_msg"].lower().split()]
        counts = Counter(tokens)

        rows.append({
            "brand": brand,
            "threads": len(usable),
            "dm_deflection_rate": round(sum(deflections) / len(usable), 3),
            "link_handoff_rate": round(
                sum(1 for r in first_replies if is_link_handoff(r)) / len(usable), 3),
            "handoff_rate": round(sum(handoffs) / len(usable), 3),
            "median_reply_chars": int(statistics.median(lengths)),
            "substantive_rate": round(
                sum(1 for r, h in zip(first_replies, handoffs) if not h and len(r) >= 80)
                / len(usable), 3),
            "multi_turn_rate": round(sum(1 for t in usable if t["customer_turns"] > 1) / len(usable), 3),
            "median_replies_per_thread": statistics.median([len(t["brand_replies"]) for t in usable]),
            "lexical_diversity": round(len(counts) / max(len(tokens), 1), 4),
        })
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = ["brand", "threads", "dm_deflection_rate", "link_handoff_rate", "handoff_rate",
            "substantive_rate", "median_reply_chars", "multi_turn_rate", "lexical_diversity"]
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    body = [
        "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        for r in sorted(rows, key=lambda r: -r.get("substantive_rate", 0))
    ]
    return "\n".join([head, rule, *body])


def main() -> None:
    rows = profile()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "brand_profile.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    table = to_markdown(rows)
    (RESULTS_DIR / "brand_profile.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
