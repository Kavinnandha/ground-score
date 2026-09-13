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
# Overridable so a smoke run against a stub backend cannot write fabricated
# responses into the committed cache that backs the published numbers.
CACHE_PATH = Path(os.environ.get(
    "GROUNDSCORE_CACHE_PATH", REPO_ROOT / "cache" / "llm_cache.sqlite"))

# Gemini ids, used when a role resolves to the `gemini` provider. All are
# env-overridable, because the binding constraint on the free tier is requests
# per DAY, and that number moves.
#
# Measured against this key (2026-09, free tier). The daily cap, not capability,
# is what picked these:
#
#   gemini-3.x-flash        5 RPM /    20 RPD  unusable: a judged split is ~450 calls
#   gemini-3.x-flash-lite  15 RPM /   500 RPD  the smallest tier that covers a split
#   gemma-4-31b-it         30 RPM / 14400 RPD  huge budget, but see below
#
# gemma-4-31b-it looks like the obvious judge on budget alone and is not the
# default, for two reproducible reasons: it returns 503 "high demand" under
# ordinary load, and it returns 500 on `response_schema` unless a
# `system_instruction` is sent alongside it. Both are handled (see
# `_call_gemini`) so it stays a working escape hatch for when the 500/day
# lite budget runs out -- GROUNDSCORE_GEMINI_JUDGE=gemma-4-31b-it -- but a
# judge that intermittently 503s cannot be what a headline number rests on.
#
# The judge is pinned to a different generation from the drafter so the two
# cannot collapse onto one model if drafting is also moved to Gemini.
GEMINI_FAST = os.environ.get("GROUNDSCORE_GEMINI_FAST", "gemini-3.5-flash-lite")
GEMINI_JUDGE = os.environ.get("GROUNDSCORE_GEMINI_JUDGE", "gemini-3.1-flash-lite")

# The self-preference probe re-scores with the DRAFTER's own model -- that is
# the entire point of the probe -- so this tracks GEMINI_FAST rather than being
# a third independent id that could drift away from it.
GEMINI_JUDGE_CROSS = GEMINI_FAST

# Only reachable via GROUNDSCORE_PROVIDER=gemini for embeddings, which is not
# the shipped path: the committed vectors are nomic-embed-text and the cache is
# keyed on the model name, so switching here starts a second, empty vector
# space rather than reusing anything. See embed.py.
GEMINI_EMBED = os.environ.get("GROUNDSCORE_GEMINI_EMBED", "gemini-embedding-001")

# Free-tier ceilings per model id, as (requests/minute, requests/day). Used to
# pace live calls and to fail early rather than after a hundred 429s. Anything
# not listed falls back to CONSERVATIVE_LIMIT.
GEMINI_LIMITS: dict[str, tuple[int, int]] = {
    "gemini-3.5-flash-lite": (15, 500),
    "gemini-3.1-flash-lite": (15, 500),
    "gemini-2.5-flash-lite": (10, 20),
    "gemini-3.8-flash": (5, 20),
    "gemini-3.7-flash": (5, 20),
    "gemini-3.6-flash": (5, 20),
    "gemini-3.5-flash": (5, 20),
    "gemini-3-flash": (5, 20),
    "gemini-2.5-flash": (5, 20),
    # 30 RPM on paper, but the 16K TPM ceiling binds first at judge-sized
    # prompts, so it is paced as if it were 8 RPM.
    "gemma-4-31b-it": (8, 14400),
    "gemma-4-26b-a4b-it": (8, 14400),
    "gemini-embedding-001": (100, 1000),
    "gemini-embedding-2": (100, 1000),
}
CONSERVATIVE_LIMIT = (4, 20)


def provider() -> str:
    """The default provider. Roles may override it -- see `role_chain`."""
    return providers.provider_name()


# Roles are addressed by token, not by a model id frozen at import. The drafter
# and the judge are separate roles precisely so they can sit on different
# vendors: a model grading its own output cannot rule out self-preference.
MODEL_FAST = "@fast"
MODEL_JUDGE = "@judge"
MODEL_JUDGE_CROSS = "@cross"

# NOT a role token. embed._text_key() hashes this string into every embedding
# cache key, so it must stay the literal model name or the committed vectors
# become unreachable.
MODEL_EMBED = providers.OLLAMA_EMBED

_ROLE_MODELS = {
    "gemini": {"fast": GEMINI_FAST, "judge": GEMINI_JUDGE, "cross": GEMINI_JUDGE_CROSS},
    "ollama": {"fast": providers.OLLAMA_FAST, "judge": providers.OLLAMA_JUDGE,
               "cross": providers.OLLAMA_FAST},
}

# The judge defaults to a DIFFERENT vendor from the drafter, with a local
# fallback. That is the whole point of a per-role chain: it buys back the
# cross-family independence that a single hosted key cannot provide.
# Override with e.g. GROUNDSCORE_ROLE_JUDGE=ollama,gemini
DEFAULT_ROLE_CHAINS = {"judge": ("gemini", "ollama")}


def model_for(prov: str, role: str) -> str:
    try:
        return _ROLE_MODELS[prov][role]
    except KeyError:
        raise KeyError(f"no model for role {role!r} on provider {prov!r}") from None


def _dedupe(names) -> tuple[str, ...]:
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def role_chain(role: str) -> tuple[str, ...]:
    """Ordered providers to try for `role`, most preferred first."""
    env = os.environ.get(f"GROUNDSCORE_ROLE_{role.upper()}")
    if env:
        chain = _dedupe(p.strip().lower() for p in env.split(","))
    elif role in DEFAULT_ROLE_CHAINS:
        chain = DEFAULT_ROLE_CHAINS[role]
    else:
        chain = _dedupe((provider(), "ollama"))
    for name in chain:
        if name not in providers.KNOWN_PROVIDERS:
            raise providers.ProviderError(
                f"role chain for {role!r} names unknown provider {name!r}; "
                f"valid: {', '.join(providers.KNOWN_PROVIDERS)}")
    return chain


def _why_unusable(prov: str) -> str:
    if prov == "ollama":
        return f"not reachable at {providers.OLLAMA_HOST}"
    return "no API key (GEMINI_API_KEY unset)"


def usable(prov: str) -> bool:
    """Whether `prov` could serve a live call right now."""
    if prov == "ollama":
        return providers.available()
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


# Which provider actually served each role this run, and every forced switch.
# Written into the results so a reader can tell whether one split was scored by
# one judge or by two -- a mixed split cannot be pooled into a single number.
_pinned: dict[str, str] = {}
SERVING: dict[str, str] = {}
PROVIDER_EVENTS: list[dict[str, Any]] = []


def _note_serving(role: str, prov: str, concrete: str, problems: list[str]) -> None:
    previous = _pinned.get(role)
    if previous == prov:
        return
    _pinned[role] = prov
    SERVING[role] = f"{prov}:{concrete}"
    event = {"role": role, "provider": prov, "model": concrete,
             "previous": previous, "reason": problems[-1] if problems else "first use"}
    PROVIDER_EVENTS.append(event)
    if previous is not None:
        print(f"\n  !! role {role!r} switched provider {previous} -> {prov}\n"
              f"     reason: {event['reason']}\n"
              f"     This split is now scored by TWO different models. Results are\n"
              f"     tagged per row; do not pool them into one number.\n", flush=True)


def provider_report() -> dict[str, Any]:
    """Serving providers and any mid-run switches, for the results files."""
    return {"serving": dict(SERVING), "switches":
            [e for e in PROVIDER_EVENTS if e["previous"] is not None],
            "chains": {r: role_chain(r) for r in ("fast", "judge", "cross")},
            "live_calls": quota_report()}

DEFAULT_TEMPERATURE = 0.0

# Pacing is per MODEL, not per run, because free-tier ceilings differ by an
# order of magnitude across the ids this project uses (5 RPM for flash, 15 for
# flash-lite). A single global RPM either throttles the fast models pointlessly
# or hammers the slow ones into backoff, and backoff costs more wall clock than
# pacing does up front. GROUNDSCORE_RPM overrides the table for every model.
RPM_OVERRIDE = int(os.environ.get("GROUNDSCORE_RPM", "0"))

# Fraction of the documented RPM actually used. The ceilings are enforced on
# the server's clock, not ours, so sitting exactly on the limit produces 429s
# from clock skew alone.
RPM_HEADROOM = 0.8


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
    """Credential for the active provider, or None.

    Provider-aware on purpose: a single global lookup would report "have key"
    for a run whose active backend cannot use that key, sail past the offline
    guard, and fail deep inside a 1000-call evaluation instead of at startup.
    """
    if provider() == "ollama":
        return None  # local backend needs no credential; see offline()
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
_last_call_at: dict[str, float] = {}
_calls_today: dict[str, int] = {}


class DailyQuotaExhausted(RuntimeError):
    """The in-process count for a model has reached its documented RPD.

    Raised rather than left to the server so the chain can fall through to the
    next provider on a clear signal, instead of after a retry ladder of 429s
    that costs minutes per row across a long run.
    """


def limits_for(model: str) -> tuple[int, int]:
    rpm, rpd = GEMINI_LIMITS.get(model, CONSERVATIVE_LIMIT)
    return (RPM_OVERRIDE or rpm), rpd


def quota_report() -> dict[str, dict[str, int]]:
    """Live calls made per model this process, against the daily ceiling.

    In-process only: it does not know what an earlier run spent today. It is a
    guard against burning a whole day's budget inside one run, not an accountant.
    """
    return {m: {"calls": n, "daily_limit": limits_for(m)[1]}
            for m, n in sorted(_calls_today.items())}


def _throttle(model: str) -> None:
    """Space out live API calls for `model` to stay under its per-minute quota.

    Cache hits never reach here, so a fully-cached replay runs at full speed.
    """
    rpm, rpd = limits_for(model)
    with _rate_lock:
        if _calls_today.get(model, 0) >= rpd:
            raise DailyQuotaExhausted(
                f"{model} has served {rpd} live calls in this process, its documented "
                f"free-tier daily ceiling. Further calls would only collect 429s.\n"
                f"  Use a model with a larger budget (GROUNDSCORE_GEMINI_JUDGE="
                f"gemma-4-31b-it is 14400/day), run the role on Ollama, or wait for "
                f"the quota to roll over."
            )
        if rpm > 0:
            min_gap = 60.0 / (rpm * RPM_HEADROOM)
            wait = min_gap - (time.monotonic() - _last_call_at.get(model, 0.0))
            if wait > 0:
                time.sleep(wait)
        _last_call_at[model] = time.monotonic()
        _calls_today[model] = _calls_today.get(model, 0) + 1


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
        if model.startswith("gemma") and not system:
            # Reproducible on gemma-4-31b-it: `response_schema` without a
            # `system_instruction` returns 500 INTERNAL on every attempt, and
            # returns valid JSON as soon as any system instruction is present.
            # Injected here rather than at the call sites so the cache key stays
            # the system prompt the CALLER wrote -- the same logical call must
            # hash identically whichever model happens to serve it.
            config["system_instruction"] = "Respond with JSON matching the given schema."

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            _throttle(model)
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
            # 429 (quota) and 503 (model overloaded -- gemma-4-31b-it returns
            # this routinely) both need a long, growing pause; anything else is
            # likely permanent but cheap to retry once.
            message = str(exc)
            slow = "429" in message or "503" in message or "RESOURCE_EXHAUSTED" in message
            time.sleep((15.0 if slow else 2.0) * (2 ** attempt))
    STATS.errors += 1
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_exc}") from last_exc


def _dispatch(prov: str, model: str, prompt: str, schema: Any,
              temperature: float, system: str | None) -> str:
    if prov == "ollama":
        return providers.generate(model, prompt, system=system, schema=schema,
                                  temperature=temperature)
    return _call_gemini(model, prompt, schema, temperature, system)


def complete(
    prompt: str,
    *,
    model: str = MODEL_FAST,
    schema: Any = None,
    temperature: float = DEFAULT_TEMPERATURE,
    system: str | None = None,
) -> str:
    """Return raw model text, served from cache when possible.

    `model` is either a concrete model id or a role token (`@judge`). A role is
    resolved through its provider chain (see `role_chain`).

    Replay checks EVERY provider in the chain before deciding it is a miss, so
    a cache built when the judge ran on Gemini still replays after the chain is
    reordered. Only a live call is subject to the chain's preference order.
    """
    role = model[1:] if model.startswith("@") else None
    if role is None:
        candidates = [(provider(), model)]
    else:
        candidates = [(p, model_for(p, role)) for p in role_chain(role)]

    for prov, concrete in candidates:
        cached = _cache_get(cache_key(concrete, prompt, schema, temperature, system, prov=prov))
        if cached is not None:
            STATS.hits += 1
            if role:
                # Provenance matters on replay too: a cached split that was
                # judged by two providers must still report as mixed, or
                # `make reproduce` would launder it into a single clean number.
                _note_serving(role, prov, concrete, ["served from cache"])
            return cached

    shown = ", ".join(f"{p}:{m}" for p, m in candidates)
    if offline():
        raise OfflineCacheMiss(
            "Call not in cache and GROUNDSCORE_OFFLINE=1.\n"
            f"  tried {shown}\n"
            "  This is a reproduction run: it must replay the committed cache exactly.\n"
            "  A miss means the committed cache does not cover this code path -- the\n"
            "  prompt, config or inputs changed. Regenerate with 'make full', or revert."
        )

    # A role stays pinned to whichever provider first served it, so one split is
    # not judged half by one model and half by another after a mid-run quota
    # failure. A forced switch is recorded and surfaced, never silent.
    order = candidates
    if role and _pinned.get(role):
        pin = _pinned[role]
        order = ([c for c in candidates if c[0] == pin]
                 + [c for c in candidates if c[0] != pin])

    problems: list[str] = []
    for prov, concrete in order:
        if not usable(prov):
            problems.append(f"{prov}: {_why_unusable(prov)}")
            continue
        try:
            text = _dispatch(prov, concrete, prompt, schema, temperature, system)
        except Exception as exc:  # noqa: BLE001 - classified by the chain
            problems.append(f"{prov}: {type(exc).__name__}: {str(exc)[:160]}")
            if role and prov != order[-1][0]:
                continue  # fall through to the next provider in the chain
            raise
        if role:
            _note_serving(role, prov, concrete, problems)
        STATS.misses += 1
        _cache_put(cache_key(concrete, prompt, schema, temperature, system, prov=prov),
                   concrete, prompt, text)
        return text

    raise OfflineCacheMiss(
        f"No provider in the chain for {model!r} could serve this call.\n"
        + "".join(f"  {p}\n" for p in problems)
        + "  Set a key for one of them, start Ollama, or run 'make reproduce' against\n"
          "  the committed cache."
    )


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
