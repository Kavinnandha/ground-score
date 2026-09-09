"""Embed every customer message in the corpus into the committed cache.

Run once with a key (`make full`); afterwards every downstream step -- the
retrieval index, intent clustering, the labelling CLI -- reads vectors from
cache/emb_cache.npz and needs no network.

Embeds golden-pool messages too. Those never enter the retrieval index (see
scripts/build_dataset.py), but they must be embeddable so the agent can be
evaluated on them offline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundscore import config, embed, llm  # noqa: E402
from groundscore.ingest import read_jsonl  # noqa: E402

BATCH_TEXTS = 500  # retained for reference; checkpointing now happens per request


def main() -> int:
    if not config.THREADS_PATH.exists():
        print(f"Missing {config.THREADS_PATH}. Run: python scripts/build_dataset.py")
        return 1

    threads = read_jsonl(config.THREADS_PATH)
    texts = sorted({t["customer_msg"] for t in threads})
    print(f"{len(threads)} threads -> {len(texts)} unique customer messages")

    cache = embed.EmbeddingCache()
    print(f"cache holds {len(cache)} vectors")

    todo = [t for t in texts if cache.get(embed._text_key(t, llm.MODEL_EMBED, embed.EMBED_DIM)) is None]
    print(f"{len(todo)} to embed")
    if not todo:
        print("nothing to do")
        return 0

    if not llm.have_key():
        print("GEMINI_API_KEY unset -- cannot embed. Set it in .env and retry.")
        return 1

    started = time.monotonic()
    done = 0

    def progress(n: int) -> None:
        nonlocal done
        done += n
        elapsed = time.monotonic() - started
        rate = done / max(elapsed, 1e-6)
        eta = (len(todo) - done) / max(rate, 1e-6)
        print(f"  {done}/{len(todo)}  ({rate * 60:.0f} texts/min, eta {eta / 60:.1f} min)", flush=True)

    # embed_gemini checkpoints the cache after every request, so an interrupted
    # or crashed run resumes from where it stopped rather than re-spending quota.
    embed.embed_gemini(todo, cache=cache, on_progress=progress)

    cache.save()
    size_mb = embed.EMB_CACHE_PATH.stat().st_size / 1e6
    print(f"done: {len(cache)} vectors, {size_mb:.1f} MB at {embed.EMB_CACHE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
