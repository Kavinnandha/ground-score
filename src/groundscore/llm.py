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

# Model choice is constrained by what this API key can actually reach. Probed
# 2026-09: every *-pro model returns 429 RESOURCE_EXHAUSTED (no pro quota on
# this tier) and gemini-3.8-flash returns 503. So the original plan of "flash
# drafts, pro judges" is not available and the tiering is done differently:
#
#   drafter      gemini-3.5-flash     cheap, fast, adequate for 280-char replies
#   judge        gemini-3.7-flash     newer generation than the drafter, so not
#                                     literally the same weights grading itself
#   cross-judge  gemma-4-31b-it       DIFFERENT MODEL FAMILY (open-weights
#                                     Gemma, not Gemini). Run on a subset to
#                                     estimate how much of the judge's approval
#                                     is same-family self-preference.
#
# The cross-family judge is the honest part: a Gemini judge scoring Gemini
# drafts cannot rule out self-preference on its own, so the bias is measured
# against an outside model rather than asserted away. See DECISIONS.md.
MODEL_FAST = "gemini-3.5-flash"
MODEL_JUDGE = "gemini-3.7-flash"
MODEL_JUDGE_CROSS = "gemma-4-31b-it"
MODEL_EMBED = "gemini-embedding-001"

DEFAULT_TEMPERATURE = 0.0

# Free-tier keys are rate limited per minute. Exceeding the limit costs more
# wall-clock in backoff than throttling does up front, so calls are paced.
RPM_LIMIT = int(os.environ.get("GROUNDSCORE_RPM", "10"))


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


def load_dotenv() -> None:
    """Populate os.environ from .env without overriding real env vars.

    Called at import so every entry point (scripts, tests, the labelling CLI)
    picks the key up the same way. .env is gitignored; nothing here is logged.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv()


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


_rate_lock = threading.Lock()
_last_call_at = 0.0


def _throttle() -> None:
    """Space out live API calls to stay under the per-minute quota.

    Cache hits never reach here, so a fully-cached replay runs at full speed.
    """
    global _last_call_at
    if RPM_LIMIT <= 0:
        return
    min_gap = 60.0 / RPM_LIMIT
    with _rate_lock:
        wait = min_gap - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def _call_gemini(
    model: str,
    prompt: str,
    schema: Any,
    temperature: float,
    system: str | None,
    max_retries: int = 5,
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
            _throttle()
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
            # 429 (quota) and 503 (model overloaded) both need a long, growing
            # pause; anything else is likely permanent but cheap to retry once.
            message = str(exc)
            slow = "429" in message or "503" in message or "RESOURCE_EXHAUSTED" in message
            time.sleep((15.0 if slow else 2.0) * (2 ** attempt))
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
