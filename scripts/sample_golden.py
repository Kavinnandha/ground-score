"""Sample the 200 golden-set candidates and pre-fill weak labels for review.

Sampling design (mirrored in data/golden/LABELING_NOTES.md)
-----------------------------------------------------------
Three strata, because one sampling scheme cannot serve the three things the
golden set has to measure:

  proportional (60%)  Drawn in proportion to cluster size. This stratum, and
                      only this stratum, estimates performance on real traffic.
  rare (25%)          Inverse-frequency draw over clusters. Without it, macro-F1
                      on tail classes would be estimated from two or three
                      examples and would be pure noise.
  adversarial (15%)   Hand-picked hard cases, selected by heuristics that have
                      nothing to do with the model: very short messages, all
                      caps, emoji-only, multi-question, rage markers.

The strata are recorded per example and scored SEPARATELY. Pooling them and
reporting one number would be misleading in both directions -- the rare stratum
drags accuracy down relative to real traffic, and the proportional stratum
hides the tail. The report gives the traffic-weighted number as the headline
and the others alongside it.

Stratification is by CLUSTER, not by predicted intent: see the note in
discover_intents.py on why using the classifier here would be circular.

Weak labels are pre-filled so a human adjudicates rather than types. The
override rate is recorded and reported -- if the human accepted 98% of the weak
labels, the golden set is closer to a copy of the model than to ground truth,
and the reader deserves to know that.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundscore import config, embed, retrieve, taxonomy  # noqa: E402
from groundscore.classify import classify  # noqa: E402
from groundscore.cleaning import stable_bucket  # noqa: E402

SEED = 42

# Adversarial heuristics. Deliberately model-independent -- these are surface
# properties of the text, so the hard slice is not "cases this model finds hard"
# (which would flatter it) but "cases that are objectively underspecified".
ADVERSARIAL_TESTS: list[tuple[str, "callable"]] = [
    ("very_short", lambda m: len(m) < 40),
    ("all_caps", lambda m: len(m) > 20 and sum(c.isupper() for c in m if c.isalpha())
                           / max(sum(c.isalpha() for c in m), 1) > 0.7),
    ("emoji_heavy", lambda m: sum(1 for c in m if ord(c) > 0x2100) >= 3),
    ("multi_question", lambda m: m.count("?") >= 2),
    ("rage_markers", lambda m: bool(re.search(r"(!{3,}|\bwtf\b|\bffs\b|\bdisgrace\b|"
                                              r"\bnever again\b|\bworst\b)", m, re.I))),
    ("multi_intent", lambda m: bool(re.search(r"\b(and also|as well as|plus,|second(?:ly)?,)\b", m, re.I))),
    ("sarcasm_marker", lambda m: bool(re.search(r"(thanks a lot|great job|well done|"
                                                r"brilliant|fantastic)\W*$", m, re.I))),
]


def adversarial_tags(message: str) -> list[str]:
    return [name for name, test in ADVERSARIAL_TESTS if test(message)]


def assign_clusters(messages: list[str], backend: str = "auto") -> np.ndarray:
    path = config.CACHE_DIR / "cluster_centroids.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m groundscore.discover_intents"
        )
    with np.load(path) as data:
        centroids = data["centroids"].astype(np.float32)
    vectors = embed.embed(messages, backend=backend, fit_corpus=messages)
    if vectors.shape[1] != centroids.shape[1]:
        raise RuntimeError(
            "embedding dimension does not match saved centroids -- the cluster "
            "model and the embedding backend are out of sync; re-run discover_intents"
        )
    return np.argmax(vectors @ centroids.T, axis=1)


def stratified_sample(pool: list[dict], clusters: np.ndarray, cfg: dict) -> list[dict]:
    target = cfg["target_size"]
    n_adv = int(round(target * cfg["adversarial_share"]))
    n_rare = int(round(target * cfg["rare_intent_share"]))
    n_prop = target - n_adv - n_rare

    # Deterministic ordering: stable hash, not RNG state, so the sample does not
    # move when the pool is rebuilt or the code is re-run on another machine.
    def rank(item: dict) -> int:
        return stable_bucket(f"golden:{SEED}:{item['thread_id']}", 10**9)

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for item, cluster in zip(pool, clusters):
        item = dict(item)
        item["cluster"] = int(cluster)
        item["adversarial_tags"] = adversarial_tags(item["customer_msg"])
        by_cluster[int(cluster)].append(item)
    for items in by_cluster.values():
        items.sort(key=rank)

    chosen: dict[str, dict] = {}

    # 1. Adversarial: prefer messages tripping the most heuristics.
    adversarial = sorted(
        (i for items in by_cluster.values() for i in items if i["adversarial_tags"]),
        key=lambda i: (-len(i["adversarial_tags"]), rank(i)),
    )
    for item in adversarial[:n_adv]:
        chosen[item["thread_id"]] = {**item, "stratum": "adversarial"}

    # 2. Rare: round-robin from smallest clusters upward.
    sizes = Counter({c: len(items) for c, items in by_cluster.items()})
    for cluster, _ in sorted(sizes.items(), key=lambda kv: kv[1]):
        if len([v for v in chosen.values() if v["stratum"] == "rare"]) >= n_rare:
            break
        for item in by_cluster[cluster]:
            if item["thread_id"] in chosen:
                continue
            chosen[item["thread_id"]] = {**item, "stratum": "rare"}
            break

    while len([v for v in chosen.values() if v["stratum"] == "rare"]) < n_rare:
        pick = next(
            (i for cluster, _ in sorted(sizes.items(), key=lambda kv: kv[1])
             for i in by_cluster[cluster] if i["thread_id"] not in chosen),
            None,
        )
        if pick is None:
            break
        chosen[pick["thread_id"]] = {**pick, "stratum": "rare"}

    # 3. Proportional: fill the rest in proportion to cluster mass.
    total = sum(sizes.values())
    quota = {c: max(1, round(n_prop * size / total)) for c, size in sizes.items()}
    for cluster in sorted(quota, key=lambda c: -sizes[c]):
        taken = 0
        for item in by_cluster[cluster]:
            if len([v for v in chosen.values() if v["stratum"] == "proportional"]) >= n_prop:
                break
            if item["thread_id"] in chosen or taken >= quota[cluster]:
                continue
            chosen[item["thread_id"]] = {**item, "stratum": "proportional"}
            taken += 1

    return sorted(chosen.values(), key=lambda i: (i["stratum"], rank(i)))


def main() -> int:
    cfg = config.brand_config()["golden"]
    pool = retrieve.golden_pool()
    if not pool:
        print("Empty golden pool. Run: python scripts/build_dataset.py")
        return 1
    print(f"golden pool: {len(pool)} threads")

    messages = [t["customer_msg"] for t in pool]
    clusters = assign_clusters(messages)
    sample = stratified_sample(pool, clusters, cfg)
    print(f"sampled {len(sample)}: {dict(Counter(s['stratum'] for s in sample))}")

    print("pre-filling weak labels (this calls the classifier)...")
    retriever = retrieve.load_retriever()
    brand = config.brand()
    candidates = []
    for i, item in enumerate(sample, start=1):
        neighbours = retriever.search(item["customer_msg"], k=5)
        weak = classify(item["customer_msg"], neighbours, brand)
        candidates.append({
            "thread_id": item["thread_id"],
            "customer_msg": item["customer_msg"],
            "customer_msg_raw": item["customer_msg_raw"],
            "brand_replies": [r["text"] for r in item["brand_replies"]][:3],
            "cluster": item["cluster"],
            "stratum": item["stratum"],
            "adversarial_tags": item["adversarial_tags"],
            "weak_intent": weak.intent,
            "weak_confidence": round(weak.confidence, 3),
            "weak_rationale": weak.rationale,
            "neighbours": [e.as_dict() for e in neighbours[:3]],
            # Dev/test assignment is fixed HERE, before any label is written, so
            # it cannot be chosen after seeing which split flatters the results.
            "split": "dev" if stable_bucket(f"devtest:{item['thread_id']}", 100)
                     < int(cfg["dev_share"] * 100) else "test",
        })
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    config.GOLDEN_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.GOLDEN_CANDIDATES_PATH.open("w", encoding="utf-8") as fh:
        for row in candidates:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    splits = Counter(c["split"] for c in candidates)
    print(f"wrote {config.GOLDEN_CANDIDATES_PATH}")
    print(f"  dev/test: {dict(splits)}")
    print(f"  weak-label distribution: {dict(Counter(c['weak_intent'] for c in candidates))}")
    print(f"  taxonomy classes: {len(taxonomy.names())}")
    print("\nNext: python tools/label_cli.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
