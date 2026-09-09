"""Derive a candidate intent taxonomy from the data, not from imagination.

Three steps, deliberately kept separate so each is auditable:

  1. Cluster the brand's inbound messages in embedding space. Sweep k and pick
     by silhouette. This step has no LLM in it and no human in it.
  2. Ask the LLM to name and define each cluster from a sample of its members.
     The LLM sees only cluster contents -- it is describing, not inventing.
  3. A HUMAN merges, splits and renames the candidates into the final taxonomy
     (taxonomy/intents.yaml), and writes the boundary rules.

Step 3 is where the judgment lives, and it is not automated. An LLM asked to
"produce a taxonomy" in one shot yields plausible-sounding classes that do not
match the actual distribution -- the classic failure where the taxonomy has a
"billing" class the brand never receives and no class for the 8% of traffic
that is people quoting the brand's own ad copy at them. Clustering first
anchors the taxonomy to real frequency mass.

Output of this module is a DRAFT (taxonomy/intents.draft.yaml). It is not used
by the agent until a human has reviewed it into intents.yaml.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
import yaml

from . import config, embed, llm
from .ingest import read_jsonl

K_RANGE = range(8, 25)
SILHOUETTE_SAMPLE = 3000
EXEMPLARS_PER_CLUSTER = 25
SEED = 42

CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "definition": {"type": "string"},
        "is_coherent": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["name", "definition", "is_coherent", "notes"],
}

SUMMARISE_PROMPT = """You are analysing customer support messages sent to {brand} on Twitter.

Below are {n} messages that an unsupervised clustering algorithm grouped together.
They were grouped by embedding similarity, so the grouping may be wrong.

Messages:
{messages}

Produce:
- name: a short snake_case intent label describing what these customers WANT.
  Name the customer's goal, not the topic. Prefer "delivery_status" over
  "packages". Prefer "refund_request" over "money".
- definition: one sentence a human annotator could apply consistently.
- is_coherent: false if these messages do not actually share a single intent.
  Be strict. A cluster of generic complaints with no common ask is NOT coherent.
- notes: if incoherent, say what distinct groups you see. If coherent, note any
  boundary that an annotator might get wrong.

Return JSON only."""


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    share: float
    name: str
    definition: str
    is_coherent: bool
    notes: str
    exemplars: list[str]

    def as_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "share": round(self.share, 4),
            "name": self.name,
            "definition": self.definition,
            "is_coherent": self.is_coherent,
            "notes": self.notes,
            "exemplars": self.exemplars[:8],
        }


def choose_k(vectors: np.ndarray, k_range=K_RANGE, seed: int = SEED) -> tuple[int, list[dict]]:
    """Sweep k, score by silhouette on a fixed subsample.

    Silhouette on all 9k points for 17 values of k is needlessly slow and the
    ranking is stable on a subsample, so a fixed-seed subsample is scored
    instead. The subsample is the same for every k, so the comparison is fair.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=min(SILHOUETTE_SAMPLE, len(vectors)), replace=False)
    sample = vectors[idx]

    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(vectors)
        score = float(silhouette_score(sample, model.labels_[idx], metric="cosine"))
        scores.append({"k": k, "silhouette": round(score, 4),
                       "inertia": round(float(model.inertia_), 2)})
        print(f"  k={k:2d}  silhouette={score:.4f}", flush=True)

    best = max(scores, key=lambda s: s["silhouette"])
    return best["k"], scores


def summarise_cluster(brand: str, messages: list[str]) -> dict:
    sample = messages[:EXEMPLARS_PER_CLUSTER]
    prompt = SUMMARISE_PROMPT.format(
        brand=brand,
        n=len(sample),
        messages="\n".join(f"- {m}" for m in sample),
    )
    return llm.complete_json(
        prompt,
        schema=CLUSTER_SCHEMA,
        default={"name": "unknown", "definition": "", "is_coherent": False,
                 "notes": "LLM output unparseable"},
    )


def discover(k: int | None = None, backend: str = "auto") -> dict:
    threads = [t for t in read_jsonl(config.THREADS_PATH)
               if t.get("split") == config.SPLIT_HISTORY]
    messages = [t["customer_msg"] for t in threads]
    brand = config.brand()

    print(f"Embedding {len(messages)} messages...")
    vectors = embed.embed(messages, backend=backend, fit_corpus=messages)

    if k is None:
        print("Sweeping k by silhouette...")
        k, sweep = choose_k(vectors)
        print(f"  chose k={k}")
    else:
        sweep = []

    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(vectors)
    labels = model.labels_
    counts = Counter(labels.tolist())

    # Order cluster members by distance to centroid so the LLM sees the most
    # typical messages, not a random draw that over-represents the fringe.
    summaries = []
    for cluster_id in sorted(counts, key=lambda c: -counts[c]):
        member_idx = np.flatnonzero(labels == cluster_id)
        centroid = model.cluster_centers_[cluster_id]
        order = member_idx[np.argsort(-(vectors[member_idx] @ centroid))]
        ordered_messages = [messages[i] for i in order]

        print(f"  summarising cluster {cluster_id} (n={counts[cluster_id]})...", flush=True)
        result = summarise_cluster(brand, ordered_messages)
        summaries.append(ClusterSummary(
            cluster_id=int(cluster_id),
            size=int(counts[cluster_id]),
            share=counts[cluster_id] / len(messages),
            name=result.get("name", "unknown"),
            definition=result.get("definition", ""),
            is_coherent=bool(result.get("is_coherent", False)),
            notes=result.get("notes", ""),
            exemplars=ordered_messages[:8],
        ))

    # Persist centroids so the golden-set sampler can stratify by cluster
    # WITHOUT consulting the classifier. Stratifying on predicted intent would
    # be circular: any intent the classifier never predicts would never be
    # sampled, and the evaluation would be blind to exactly the classes the
    # model is worst at.
    np.savez_compressed(
        config.CACHE_DIR / "cluster_centroids.npz",
        centroids=model.cluster_centers_.astype(np.float32),
        cluster_ids=np.array(sorted(counts), dtype=np.int32),
    )

    return {
        "brand": brand,
        "k": k,
        "n_messages": len(messages),
        "silhouette_sweep": sweep,
        "clusters": [s.as_dict() for s in summaries],
    }


def write_draft(result: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "intent_clusters.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    draft = {
        "_note": (
            "DRAFT produced by src/groundscore/discover_intents.py. "
            "Not used by the agent. A human merges this into intents.yaml."
        ),
        "brand": result["brand"],
        "clusters": [
            {
                "name": c["name"],
                "definition": c["definition"],
                "share": c["share"],
                "size": c["size"],
                "is_coherent": c["is_coherent"],
                "notes": c["notes"],
                "exemplars": c["exemplars"],
            }
            for c in result["clusters"]
        ],
    }
    path = config.TAXONOMY_PATH.parent / "intents.draft.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    write_draft(discover())


if __name__ == "__main__":
    main()
