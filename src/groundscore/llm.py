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

# Gemini ids, used only when GROUNDSCORE_PROVIDER=gemini. Probed 2026-09 on the
# free tier: every *-pro model returns 429 (no pro quota), gemini-3.8-flash
# returns 503, and generate_content is capped at 20 requests PER DAY PER MODEL
# -- which is why the default backend is local. See providers.py and
# DECISIONS.md #22.
GEMINI_FAST = "gemini-3.5-flash"
GEMINI_JUDGE = "gemini-3.7-flash"
GEMINI_JUDGE_CROSS = "gemma-4-31b-it"
GEMINI_EMBED = "gemini-embedding-001"

# Anthropic ids, used when GROUNDSCORE_PROVIDER=anthropic.
#
# The drafter is deliberately the cheap, fast model: a support triage route is
# high-volume and latency-sensitive, so Haiku is what this system would actually
# run in production, and evaluating a model nobody would deploy proves nothing.
#
# The judge is deliberately STRONGER than the drafter. On the local backend the
# judge was a different family (gemma judging qwen); on one Anthropic key that
# separation is impossible, so it is replaced with a capability gap in the safe
# direction -- a strong model grading a weaker one. The residual same-vendor
# risk is not waved away: MODEL_JUDGE_CROSS re-scores with the drafter's own
# model, so self-preference is measured rather than assumed absent. See
# DECISIONS.md #5 and the report's limitations section.
ANTHROPIC_FAST = os.environ.get("GROUNDSCORE_ANTHROPIC_FAST", "claude-haiku-4-5")
ANTHROPIC_JUDGE = os.environ.get("GROUNDSCORE_ANTHROPIC_JUDGE", "claude-opus-5")

# Anthropic has no embeddings endpoint. The embedding cache is keyed on
# (model, dim, text) and NOT on provider, so the committed nomic-embed-text
# vectors stay valid while generation runs on a hosted API. Every corpus and
# golden-set text is already in that cache; a miss raises rather than silently
# switching backends and changing what "similar" means mid-evaluation.
ANTHROPIC_EMBED = providers.OLLAMA_EMBED

# Sampling parameters were removed on the 4.6+ generation: sending temperature
# to Opus 5 or Sonnet 5 is a 400, not a warning. Only the models that still
# accept it get it, which is why temperature cannot carry reproducibility here
# -- the cache does.
ANTHROPIC_ACCEPTS_TEMPERATURE = ("claude-haiku-4-5",)


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
    "anthropic": {"fast": ANTHROPIC_FAST, "judge": ANTHROPIC_JUDGE, "cross": ANTHROPIC_FAST},
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
    return f"no API key ({'ANTHROPIC_API_KEY' if prov == 'anthropic' else 'GEMINI_API_KEY'} unset)"


def usable(prov: str) -> bool:
    """Whether `prov` could serve a live call right now."""
    if prov == "ollama":
        return providers.available()
    if prov == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY")
                    or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
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
            "chains": {r: role_chain(r) for r in ("fast", "judge", "cross")}}

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
    """Credential for the active provider, or None.

    Provider-aware on purpose: with a single global lookup, running the
    Anthropic backend with only a stale Gemini key exported would report
    "have key", sail past the offline guard, and fail deep inside a 1000-call
    evaluation instead of at startup.
    """
    if provider() == "anthropic":
        # AUTH_TOKEN counts as a credential but is not an api_key -- the SDK
        # sends it on a different header, so only API_KEY is passed explicitly
        # to the client constructor (see _get_anthropic_client).
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    elif provider() == "ollama":
        return None  # local backend needs no credential; see offline()
    else:
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


def _strict_schema(schema: Any) -> Any:
    """Add `additionalProperties: false` to every object node.

    Anthropic's structured outputs reject an object schema without it. Done
    here rather than in the schema literals so the cache key stays the schema
    the caller wrote -- the same prompt must hash identically whichever backend
    happens to serve it.
    """
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(schema, list):
        return [_strict_schema(v) for v in schema]
    return schema


_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # lazy: offline replay must not need the SDK

        # The SDK already retries 408/409/429/5xx with exponential backoff, so
        # the outer loop below only handles what it does not: empty output and
        # policy refusals.
        _anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY") or None, max_retries=5)
    return _anthropic_client


def _call_anthropic(
    model: str,
    prompt: str,
    schema: Any,
    temperature: float,
    system: str | None,
    max_retries: int = 3,
) -> str:
    import anthropic

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": int(os.environ.get("GROUNDSCORE_MAX_TOKENS", "2048")),
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    output_config: dict[str, Any] = {}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": _strict_schema(schema)}
    if model not in ANTHROPIC_ACCEPTS_TEMPERATURE:
        # Sampling params are rejected on this generation; effort is the knob
        # that replaced them. Haiku is the inverse -- it takes temperature and
        # rejects effort -- so exactly one of the two is ever sent.
        output_config["effort"] = os.environ.get("GROUNDSCORE_ANTHROPIC_EFFORT", "low")
    else:
        kwargs["temperature"] = temperature
    if output_config:
        kwargs["output_config"] = output_config

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _get_anthropic_client().messages.create(**kwargs)
            if resp.stop_reason == "refusal":
                category = getattr(resp.stop_details, "category", None)
                raise ValueError(f"refused by safety classifier (category={category})")
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if not text:
                raise ValueError(f"empty response (stop_reason={resp.stop_reason})")
            return text
        except anthropic.BadRequestError:
            # A malformed request will fail identically on every retry, and
            # retrying it 3x across 1000 rows just burns money and wall clock.
            STATS.errors += 1
            raise
        except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError) as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            time.sleep(2.0 * (2 ** attempt))
    STATS.errors += 1
    raise RuntimeError(
        f"Anthropic call failed after {max_retries} attempts: {last_exc}") from last_exc


def _dispatch(prov: str, model: str, prompt: str, schema: Any,
              temperature: float, system: str | None) -> str:
    if prov == "ollama":
        return providers.generate(model, prompt, system=system, schema=schema,
                                  temperature=temperature)
    if prov == "anthropic":
        return _call_anthropic(model, prompt, schema, temperature, system)
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
