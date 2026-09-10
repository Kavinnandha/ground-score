"""Model access with an on-disk, content-addressed response cache.

Why this file exists in this shape
----------------------------------
The brief requires a reviewer to reproduce the headline results in under 15
minutes. Live model calls make that impossible: they are slow, they may need a
key, and they are not deterministic across runs.

So every call is content-addressed by SHA256(provider, model, prompt, schema,
temperature, system) and stored in a SQLite file that is COMMITTED to the repo.
A reviewer replays the exact responses that produced the numbers in results/.
The provider is part of the key, so a cache built locally is never confused
with one built against a hosted API.

A cache miss during a reproduction run is a hard error, never a silent
fallback. That distinction carries the whole reproducibility claim, and it
needs `offline()` rather than just an unset API key: the default backend is
local Ollama, which needs no credentials, so removing keys alone would let a
miss be served by a live local call while still looking like a clean replay.

Temperature is pinned to 0.0 everywhere. That does not make any of these models
strictly deterministic, which is precisely why the cache, not the temperature,
is what carries reproducibility.
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

from . import providers

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "cache" / "llm_cache.sqlite"

# Gemini ids, used only when GROUNDSCORE_PROVIDER=gemini. Probed 2026-09 on the
# free tier: every *-pro model returns 429 (no pro quota), gemini-3.8-flash
# returns 503, and generate_content is capped at 20 requests PER DAY PER MODEL
# -- which is why the default backend is local. See providers.py and
# DECISIONS.md #22.
GEMINI_FAST = "gemini-3.5-flash"
GEMINI_JUDGE = "gemini-3.7-flash"
GEMINI_JUDGE_CROSS = "gemma-4-31b-it"
GEMINI_EMBED = "gemini-embedding-001"


def provider() -> str:
    return providers.provider_name()


def _resolve(role: str) -> str:
    """Model id for a role under the active provider.

    Roles rather than hard-coded ids, because the drafter/judge/cross-judge
    split has to hold on both backends: the judge must never be the same model
    as the drafter, or the reply scores measure a model grading itself.
    """
    if provider() == "gemini":
        return {"fast": GEMINI_FAST, "judge": GEMINI_JUDGE,
                "cross": GEMINI_JUDGE_CROSS, "embed": GEMINI_EMBED}[role]
    return {"fast": providers.OLLAMA_FAST, "judge": providers.OLLAMA_JUDGE,
            "cross": providers.OLLAMA_FAST, "embed": providers.OLLAMA_EMBED}[role]


# Resolved once at import so a single run cannot silently mix backends.
MODEL_FAST = _resolve("fast")
MODEL_JUDGE = _resolve("judge")
MODEL_JUDGE_CROSS = _resolve("cross")
MODEL_EMBED = _resolve("embed")

DEFAULT_TEMPERATURE = 0.0

# Free-tier keys are rate limited per minute. Measured empirically on this key:
# generate_content starts returning 429 at the 5th call inside a minute, well
# below the documented allowance. Exceeding the limit costs more wall clock in
# backoff than pacing does up front, so calls are spaced to stay under it.
RPM_LIMIT = int(os.environ.get("GROUNDSCORE_RPM", "4"))


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


def cache_key(model: str, prompt: str, schema: Any, temperature: float,
              system: str | None, prov: str | None = None) -> str:
    payload = json.dumps(
        {
            "provider": prov or provider(),
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


def offline() -> bool:
    """True during a reproduction run.

    Stripping API keys is enough to force a hard failure on a cache miss when
    the backend is a hosted API. It is NOT enough for a local backend: Ollama
    needs no credentials, so a miss would quietly be served by a live local
    call and the "these numbers came from the committed cache" claim would be
    false without anything appearing to go wrong. This flag closes that hole.
    """
    return os.environ.get("GROUNDSCORE_OFFLINE", "").strip() == "1"


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

    if offline():
        raise OfflineCacheMiss(
            "Call not in cache and GROUNDSCORE_OFFLINE=1.\n"
            f"  model={model} key={key[:12]}...\n"
            "  This is a reproduction run: it must replay the committed cache exactly.\n"
            "  A miss means the committed cache does not cover this code path -- the\n"
            "  prompt, config or inputs changed. Regenerate with 'make full', or revert."
        )

    if provider() == "ollama":
        if not providers.available():
            raise OfflineCacheMiss(
                "Call not in cache and Ollama is not reachable at "
                f"{providers.OLLAMA_HOST}.\n"
                "  'make reproduce' must run entirely from the committed cache. Start the\n"
                "  Ollama app and run 'make full' to regenerate, or revert your change."
            )
        STATS.misses += 1
        text = providers.generate(model, prompt, system=system, schema=schema,
                                  temperature=temperature)
        _cache_put(key, model, prompt, text)
        return text

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
