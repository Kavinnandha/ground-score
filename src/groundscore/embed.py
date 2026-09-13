"""Text embeddings, cached to disk so offline replay works.

Backend choice
--------------
`sentence-transformers/all-MiniLM` would be the obvious default. It is
unavailable here: the dev machine runs Python 3.14, for which no PyTorch wheels
exist, and pinning an older interpreter would complicate setup for every
reviewer.

Three backends are implemented:

  ollama  nomic-embed-text running locally. THE DEFAULT. 768-dim, ~324
          texts/minute warm, no quota, no key, no network.
  gemini  gemini-embedding-001. Implemented and working, but the free tier
          throttled it to a few hundred texts before stalling, so it could not
          embed a 10k-message corpus in practice.
  tfidf   char-ngram TF-IDF + TruncatedSVD, pure scikit-learn. Keyless fallback
          that needs no model at all. Genuinely worse at paraphrase ("can't log
          in" vs "password reset loop" stay far apart), so it is a fallback and
          not the default.

Vectors are L2-normalised, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import llm, providers

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB_CACHE_PATH = REPO_ROOT / "cache" / "emb_cache.npz"

# 768 rather than the native 3072: it keeps the committed cache ~4x smaller for
# a marginal retrieval-quality cost on short tweets, and the repo has to stay
# clonable. Documented in DECISIONS.md.
# nomic-embed-text is natively 768-dim, which is also what the Gemini path was
# truncated to, so the two backends produce caches of the same shape.
EMBED_DIM = 768
# The free tier meters embeddings at 100 *texts* per minute (the quota is named
# "requests" but a batched call is charged per text). Relying on 429s to pace
# the run does not work: a large request arrives mid-window, gets rejected, and
# on retry collides with the next window's traffic, so the job can stall for a
# long time making no progress. Small requests plus deliberate pacing keep the
# run moving steadily instead.
_BATCH = 20 if llm.provider() == "gemini" else 64
_TEXTS_PER_MINUTE = 90   # Gemini free tier only; local has no quota
_last_request_at = 0.0


def _text_key(text: str, model: str, dim: int) -> str:
    return hashlib.sha256(f"{model}|{dim}|{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Flat key -> vector store persisted as a single .npz."""

    def __init__(self, path: Path = EMB_CACHE_PATH):
        self.path = path
        self.keys: dict[str, int] = {}
        self.matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self._dirty = False
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        with np.load(self.path, allow_pickle=False) as data:
            key_list = [str(k) for k in data["keys"]]
            self.matrix = data["vectors"].astype(np.float32)
        self.keys = {k: i for i, k in enumerate(key_list)}

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.keys.items(), key=lambda kv: kv[1])
        # float16 on disk: these are L2-normalised unit vectors, so every
        # component is in [-1, 1] where fp16 has ~3 decimal digits of precision.
        # That is far below the noise floor of cosine *ranking* over 9k short
        # texts, and it halves a cache that has to be committed to the repo.
        np.savez_compressed(
            self.path,
            keys=np.array([k for k, _ in ordered]),
            vectors=self.matrix.astype(np.float16),
        )
        self._dirty = False

    def get(self, key: str) -> np.ndarray | None:
        idx = self.keys.get(key)
        return None if idx is None else self.matrix[idx]

    def put_many(self, keys: list[str], vectors: np.ndarray) -> None:
        if not keys:
            return
        start = self.matrix.shape[0]
        self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)]) if start else vectors.astype(np.float32)
        for offset, key in enumerate(keys):
            self.keys[key] = start + offset
        self._dirty = True

    def __len__(self) -> int:
        return len(self.keys)


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def _retry_delay_seconds(message: str, fallback: float = 30.0) -> float:
    """Pull Google's own 'Please retry in 29.2s' hint out of a 429 body.

    Obeying the server's stated delay beats guessing at the quota shape: the
    free tier meters embeddings at 100 requests/minute, but a batched call
    appears to be charged per text rather than per request, so the effective
    ceiling is not something the client can compute up front.
    """
    match = re.search(r"[Pp]lease retry in ([0-9.]+)s", message)
    if match:
        return float(match.group(1)) + 2.0
    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", message)
    if match:
        return float(match.group(1)) + 2.0
    return fallback


def _embed_chunk(texts: list[str], model: str, dim: int, max_attempts: int = 10) -> np.ndarray:
    """Embed one request's worth of texts.

    Local (Ollama) has no quota, so it just retries transport hiccups. The
    Gemini path additionally has to survive quota rejections, and obeys the
    server's own stated retry delay rather than guessing at the window shape.
    """
    if llm.provider() == "ollama":
        for attempt in range(max_attempts):
            try:
                vectors = providers.embed(model, texts)
                return _l2_normalise(np.array(vectors, dtype=np.float32))
            except providers.ProviderError as exc:
                if attempt == max_attempts - 1:
                    raise
                print(f"    ollama embed retry ({exc})", flush=True)
                time.sleep(min(3.0 * (2 ** attempt), 30.0))

    from google.genai import types

    config = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=dim)

    for attempt in range(max_attempts):
        try:
            resp = llm._get_client().models.embed_content(
                model=model, contents=texts, config=config
            )
            return _l2_normalise(np.array([e.values for e in resp.embeddings], dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 - classified below
            message = str(exc)
            is_quota = "429" in message or "RESOURCE_EXHAUSTED" in message
            is_network = isinstance(exc, (OSError, ConnectionError)) or any(
                s in type(exc).__name__ for s in ("Connect", "Timeout", "Transport", "Remote")
            ) or "getaddrinfo" in message
            if attempt == max_attempts - 1 or not (is_quota or is_network):
                raise
            delay = _retry_delay_seconds(message) if is_quota else min(5.0 * (2 ** attempt), 120.0)
            reason = "quota" if is_quota else f"network ({type(exc).__name__})"
            print(f"    {reason}, sleeping {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def embed_model(texts: list[str], *, model: str = llm.MODEL_EMBED, dim: int = EMBED_DIM,
                cache: EmbeddingCache | None = None,
                on_progress: Callable[[int], None] | None = None) -> np.ndarray:
    """Embed via the configured provider, serving hits from the cache.

    Raises llm.OfflineCacheMiss when texts are uncached and the backend is not
    reachable -- same contract as llm.complete(), for the same reason.
    """
    cache = cache if cache is not None else EmbeddingCache()
    keys = [_text_key(t, model, dim) for t in texts]

    missing_idx = [i for i, k in enumerate(keys) if cache.get(k) is None]
    if missing_idx:
        missing = f"{len(missing_idx)} of {len(texts)} texts are not in the embedding cache"
        if llm.provider() == "ollama":
            if not providers.available():
                raise llm.OfflineCacheMiss(
                    f"{missing}, and Ollama is not reachable at {providers.OLLAMA_HOST}.\n"
                    "  Start the Ollama app and rerun, or use backend='tfidf'."
                )
        elif not llm.have_key():
            raise llm.OfflineCacheMiss(
                f"{missing} and GEMINI_API_KEY is unset.\n"
                "  Set a key and run 'make full' to regenerate, or use backend='tfidf'."
            )
        # Deduplicate before spending API calls: threads repeat boilerplate.
        uniq_texts = list(dict.fromkeys(texts[i] for i in missing_idx))

        # Persist after EVERY request, not at the end. This run is rate limited
        # to roughly 100 texts/minute, so a failure near the end of a 10k-text
        # job would otherwise throw away an hour of quota.
        for start in range(0, len(uniq_texts), _BATCH):
            chunk = uniq_texts[start : start + _BATCH]
            vectors = _embed_chunk(chunk, model, dim)
            cache.put_many([_text_key(t, model, dim) for t in chunk], vectors)
            cache.save()
            if on_progress:
                on_progress(len(chunk))

    return np.vstack([cache.get(k) for k in keys]).astype(np.float32)


class TfidfSvdEmbedder:
    """Keyless fallback. Must be fit() on the corpus before transform()."""

    def __init__(self, dim: int = 256, seed: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectoriser = TfidfVectorizer(
            analyzer="char_wb",  # char n-grams survive tweet typos far better than words
            ngram_range=(3, 5),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=seed)
        self._fitted = False

    def fit(self, corpus: list[str]) -> "TfidfSvdEmbedder":
        self.svd.fit(self.vectoriser.fit_transform(corpus))
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedder.fit() must be called before transform()")
        return _l2_normalise(self.svd.transform(self.vectoriser.transform(texts)))


def embed(texts: list[str], *, backend: str = "auto", fit_corpus: list[str] | None = None) -> np.ndarray:
    """Embed `texts`.

    backend='auto'   -> Gemini (cache or API); falls back to TF-IDF only if the
                        cache cannot serve the request AND no key is available.
    backend='gemini' -> Gemini, hard-fail on an offline miss.
    backend='tfidf'  -> keyless fallback; `fit_corpus` defaults to `texts`.
    """
    if backend == "tfidf":
        return TfidfSvdEmbedder().fit(fit_corpus or texts).transform(texts)
    if backend in ("gemini", "ollama", "model"):
        return embed_model(texts)
    if backend != "auto":
        raise ValueError(f"unknown embedding backend: {backend!r}")
    try:
        return embed_model(texts)
    except llm.OfflineCacheMiss:
        return TfidfSvdEmbedder().fit(fit_corpus or texts).transform(texts)


# Back-compat alias: earlier revisions called this embed_gemini.
embed_gemini = embed_model
