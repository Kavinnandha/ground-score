"""Text embeddings, cached to disk so offline replay works.

Backend choice, and why it is not sentence-transformers
-------------------------------------------------------
The obvious default for this task would be `sentence-transformers/all-MiniLM`.
It is unavailable here: the dev machine runs Python 3.14, for which no PyTorch
wheels exist. Rather than pin an older interpreter, embeddings come from the
Gemini embeddings API and are cached to a committed .npz, so the retrieval
index is byte-identical for a reviewer with no API key.

A pure-sklearn TF-IDF + SVD backend is kept as a keyless fallback. It is
genuinely worse at short-text semantic similarity (it matches surface tokens,
so "can't log in" and "password reset loop" stay far apart), which is why it is
a fallback and not the default -- but it makes every code path runnable by
someone with no key and no cache at all.

Vectors are L2-normalised, so cosine similarity is a plain dot product. Gemini
embeddings truncated below their native 3072 dimensions must be renormalised;
that is done here rather than at call sites.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import llm

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB_CACHE_PATH = REPO_ROOT / "cache" / "emb_cache.npz"

# 768 rather than the native 3072: it keeps the committed cache ~4x smaller for
# a marginal retrieval-quality cost on short tweets, and the repo has to stay
# clonable. Documented in DECISIONS.md.
EMBED_DIM = 768
_BATCH = 100  # Gemini embed_content batch ceiling


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
        np.savez_compressed(
            self.path,
            keys=np.array([k for k, _ in ordered]),
            vectors=self.matrix,
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


def _embed_gemini_uncached(texts: list[str], model: str, dim: int) -> np.ndarray:
    from google.genai import types

    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        chunk = texts[i : i + _BATCH]
        resp = llm._get_client().models.embed_content(
            model=model,
            contents=chunk,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=dim,
            ),
        )
        out.extend(e.values for e in resp.embeddings)
    return _l2_normalise(np.array(out, dtype=np.float32))


def embed_gemini(texts: list[str], *, model: str = llm.MODEL_EMBED, dim: int = EMBED_DIM,
                 cache: EmbeddingCache | None = None) -> np.ndarray:
    """Embed via Gemini, serving hits from the committed cache.

    Raises llm.OfflineCacheMiss when texts are uncached and no key is set --
    same contract as llm.complete(), for the same reason.
    """
    cache = cache if cache is not None else EmbeddingCache()
    keys = [_text_key(t, model, dim) for t in texts]

    missing_idx = [i for i, k in enumerate(keys) if cache.get(k) is None]
    if missing_idx:
        if not llm.have_key():
            raise llm.OfflineCacheMiss(
                f"{len(missing_idx)} of {len(texts)} texts are not in the embedding cache "
                f"and GEMINI_API_KEY is unset.\n"
                "  Set a key and run 'make full' to regenerate, or use backend='tfidf'."
            )
        # Deduplicate before spending API calls: threads repeat boilerplate.
        uniq: dict[str, list[int]] = {}
        for i in missing_idx:
            uniq.setdefault(texts[i], []).append(i)
        uniq_texts = list(uniq.keys())
        vectors = _embed_gemini_uncached(uniq_texts, model, dim)
        cache.put_many([_text_key(t, model, dim) for t in uniq_texts], vectors)
        cache.save()

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
    if backend == "gemini":
        return embed_gemini(texts)
    if backend != "auto":
        raise ValueError(f"unknown embedding backend: {backend!r}")
    try:
        return embed_gemini(texts)
    except llm.OfflineCacheMiss:
        return TfidfSvdEmbedder().fit(fit_corpus or texts).transform(texts)
