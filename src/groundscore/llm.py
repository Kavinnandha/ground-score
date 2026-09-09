"""Gemini adapter with an on-disk, content-addressed response cache.

Why this file exists in this shape
----------------------------------
The brief requires a reviewer to reproduce the headline results in under 15
minutes. Live LLM calls make that impossible: they are slow, they cost money,
they need a key, and they are not deterministic across runs.

So every model call is content-addressed by SHA256(provider, model, prompt,
schema, temperature) and stored in a SQLite file that is COMMITTED to the repo.
A reviewer with no GEMINI_API_KEY set replays the exact responses that produced
the numbers in results/. A cache miss without a key is a hard error rather than
a silent fallback -- a silent fallback would let the pipeline quietly diverge
from the published numbers, which is the failure mode this design exists to
prevent.

Temperature is pinned to 0.0 everywhere. That does not make Gemini strictly
deterministic (it is not), which is precisely why the cache, not the
temperature, is what carries reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "cache" / "llm_cache.sqlite"

# Model tiers. The judge deliberately runs on a different (stronger) tier than
# the drafter. That does not eliminate same-family self-preference bias -- it
# cannot -- but it removes the degenerate "identical model grades its own
# output" case. The residual bias is measured in eval/judge_agreement.py and
# reported rather than hidden.
MODEL_FAST = "gemini-2.5-flash"      # classification, drafting, bulk work
MODEL_JUDGE = "gemini-2.5-pro"       # LLM-as-judge only
MODEL_EMBED = "gemini-embedding-001"

DEFAULT_TEMPERATURE = 0.0


class OfflineCacheMiss(RuntimeError):
    """Raised when a call is not cached and no API key is available.

    Deliberately fatal. See module docstring.
    """


@dataclass
class LLMStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"cache_hits": self.hits, "cache_misses": self.misses, "errors": self.errors}


STATS = LLMStats()

_local = threading.local()
_init_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    """One SQLite connection per thread (SQLite objects are not thread-safe)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CACHE_PATH), timeout=30.0)
        with _init_lock:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key        TEXT PRIMARY KEY,
                    model      TEXT NOT NULL,
                    prompt     TEXT NOT NULL,
                    response   TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        _local.conn = conn
    return conn


def cache_key(model: str, prompt: str, schema: Any, temperature: float, system: str | None) -> str:
    payload = json.dumps(
        {
            "provider": "gemini",
            "model": model,
            "prompt": prompt,
            "system": system,
            "schema": schema,
            "temperature": temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    row = _conn().execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(key: str, model: str, prompt: str, response: str) -> None:
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO llm_cache (key, model, prompt, response, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (key, model, prompt, response, time.time()),
    )
    conn.commit()


def api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return key.strip() if key and key.strip() else None


def have_key() -> bool:
    return api_key() is not None


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # lazy: offline replay must not need the SDK

        _client = genai.Client(api_key=api_key())
    return _client


def _call_gemini(
    model: str,
    prompt: str,
    schema: Any,
    temperature: float,
    system: str | None,
    max_retries: int = 4,
) -> str:
    from google.genai import errors as genai_errors

    config: dict[str, Any] = {"temperature": temperature}
    if system:
        config["system_instruction"] = system
    if schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _get_client().models.generate_content(
                model=model, contents=prompt, config=config
            )
            text = resp.text
            if text is None:
                # Safety block or empty candidate. Raise so the retry loop sees
                # it; if it persists the caller records a failure for this row.
                reasons = [c.finish_reason for c in (resp.candidates or [])]
                raise ValueError(f"empty response (finish reasons: {reasons})")
            return text
        except (genai_errors.APIError, ValueError) as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            time.sleep(2.0 * (2 ** attempt))  # 2s, 4s, 8s
    STATS.errors += 1
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_exc}") from last_exc


def complete(
    prompt: str,
    *,
    model: str = MODEL_FAST,
    schema: Any = None,
    temperature: float = DEFAULT_TEMPERATURE,
    system: str | None = None,
) -> str:
    """Return raw model text, served from cache when possible.

    Raises OfflineCacheMiss if uncached and no API key is configured.
    """
    key = cache_key(model, prompt, schema, temperature, system)
    cached = _cache_get(key)
    if cached is not None:
        STATS.hits += 1
        return cached

    if not have_key():
        raise OfflineCacheMiss(
            "LLM call not in cache and GEMINI_API_KEY is unset.\n"
            f"  model={model} key={key[:12]}...\n"
            "  'make reproduce' must run entirely from the committed cache. If you changed a\n"
            "  prompt, config, or input, the cache key changed too -- set GEMINI_API_KEY and\n"
            "  run 'make full' to regenerate, or revert the change."
        )

    STATS.misses += 1
    text = _call_gemini(model, prompt, schema, temperature, system)
    _cache_put(key, model, prompt, text)
    return text


def complete_json(
    prompt: str,
    *,
    schema: Any,
    model: str = MODEL_FAST,
    temperature: float = DEFAULT_TEMPERATURE,
    system: str | None = None,
    default: Any = None,
) -> Any:
    """complete() plus JSON parsing.

    Gemini occasionally wraps JSON in a markdown fence even under a response
    schema, so strip that before parsing. On unparseable output return `default`
    when one is provided, so a single bad row cannot abort a 200-row evaluation
    -- the caller records it as a failure instead.
    """
    raw = complete(prompt, model=model, schema=schema, temperature=temperature, system=system)
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        if default is not None:
            return default
        raise


def cache_stats() -> dict[str, Any]:
    row = _conn().execute(
        "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM llm_cache"
    ).fetchone()
    return {
        "entries": row[0],
        "first_written": row[1],
        "last_written": row[2],
        "path": str(CACHE_PATH),
        **STATS.as_dict(),
    }
