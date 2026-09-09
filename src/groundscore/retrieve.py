"""Retrieval over the brand's historical (customer message -> reply) pairs.

This is what "grounded in how that brand has historically resolved similar
issues" means operationally: for an incoming message, pull the k most similar
past customer messages and hand the drafter what the brand actually replied.

Two properties matter more than retrieval quality itself:

1. The index is built ONLY from threads in the `history` split. Golden-pool
   threads are excluded at corpus-build time, so the agent can never retrieve
   the exact conversation it is being evaluated on. tests/test_pipeline.py
   asserts this; without it every reply metric would be inflated by leakage.

2. The top similarity score is returned to the caller, because it is a routing
   feature. A message with no similar precedent is a message the agent has
   nothing to ground a reply in, and that is a reason to escalate rather than
   to improvise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config, embed
from .cleaning import is_deflection
from .ingest import read_jsonl


@dataclass
class Exemplar:
    thread_id: str
    customer_msg: str
    brand_reply: str
    similarity: float
    is_deflection: bool

    def as_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "customer_msg": self.customer_msg,
            "brand_reply": self.brand_reply,
            "similarity": round(self.similarity, 4),
            "is_deflection": self.is_deflection,
        }


def default_backend() -> str:
    """Embedding backend from configs/brand.yaml (see the note there)."""
    return config.brand_config()["retrieval"].get("backend", "tfidf")


class Retriever:
    def __init__(self, threads: list[dict], *, backend: str | None = None):
        backend = backend or default_backend()
        self.threads = threads
        self.backend = backend
        self.messages = [t["customer_msg"] for t in threads]
        self.replies = [t["brand_replies"][0]["text"] for t in threads]
        self.ids = [t["thread_id"] for t in threads]

        self._tfidf = None
        if backend == "tfidf":
            self._tfidf = embed.TfidfSvdEmbedder().fit(self.messages)
            self.matrix = self._tfidf.transform(self.messages)
        else:
            self.matrix = embed.embed(self.messages, backend=backend, fit_corpus=self.messages)
            if backend == "auto" and self.matrix.shape[1] != embed.EMBED_DIM:
                # embed() silently fell back to TF-IDF; keep the fitted model so
                # queries are projected into the same space as the index.
                self._tfidf = embed.TfidfSvdEmbedder().fit(self.messages)
                self.matrix = self._tfidf.transform(self.messages)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._tfidf is not None:
            return self._tfidf.transform(texts)
        return embed.embed(texts, backend="gemini")

    def search(self, query: str, k: int = 5) -> list[Exemplar]:
        return self.search_many([query], k=k)[0]

    def search_many(self, queries: list[str], k: int = 5) -> list[list[Exemplar]]:
        if not queries:
            return []
        # Vectors are L2-normalised, so a dot product IS cosine similarity.
        sims = self._encode(queries) @ self.matrix.T
        out = []
        for row in sims:
            top = np.argpartition(-row, min(k, len(row) - 1))[:k]
            top = top[np.argsort(-row[top])]
            out.append([
                Exemplar(
                    thread_id=self.ids[i],
                    customer_msg=self.messages[i],
                    brand_reply=self.replies[i],
                    similarity=float(row[i]),
                    is_deflection=is_deflection(self.replies[i]),
                )
                for i in top
            ])
        return out


@lru_cache(maxsize=4)
def load_retriever(backend: str | None = None) -> Retriever:
    """History-split retriever. Cached: rebuilding it re-reads 10k threads."""
    threads = [
        t for t in read_jsonl(config.THREADS_PATH)
        if t.get("split") == config.SPLIT_HISTORY
    ]
    if not threads:
        raise RuntimeError(
            f"No history-split threads in {config.THREADS_PATH}. "
            "Run: python scripts/build_dataset.py"
        )
    return Retriever(threads, backend=backend)


def golden_pool() -> list[dict]:
    return [
        t for t in read_jsonl(config.THREADS_PATH)
        if t.get("split") == config.SPLIT_GOLDEN_POOL
    ]
