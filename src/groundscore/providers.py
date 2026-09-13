"""Model backends: local Ollama and hosted Gemini.

This module holds the Ollama transport; the hosted path lives in `llm.py`
alongside the cache that fronts both.

How a backend is chosen
-----------------------
Not globally. Each ROLE has its own ordered provider chain (see llm.role_chain):
the drafter heads at `ollama`, the judge heads at `gemini`, embeddings are
`ollama` only. The judge must not share the drafter's lineage, and making that a
property of the role rather than of the run means it holds by default instead of
by remembering to set an env var. DECISIONS.md #32.

Replay is keyless: the response cache is committed and `make reproduce` runs
with keys stripped. Only *regenerating* results needs a key.

Why generation is local and only judging is hosted
--------------------------------------------------
Volume decides it. A full evaluation of this project is roughly a thousand
calls; a judged split alone is ~450. The Gemini free tier caps
`generate_content` per DAY per model, and even the most generous id this key can
reach that is reliable enough to trust is 500/day. There is no hosted budget for
the drafting side, so drafting runs locally, where there is no quota at all.

Locally, on a GTX 1660 Ti (6GB), a 4B model at q4 sits entirely in VRAM and
answers in a few seconds. That is what makes the local drafter a real backend
rather than a bottleneck -- an earlier revision of this project ran on Intel
integrated graphics, where the same model took 60-90s per call and a full
evaluation took 12-24 hours. The hardware, not the architecture, was what
changed. See DECISIONS.md #28.

Running the drafter locally buys three things the hosted path could not:

  * **Reproducibility.** A reviewer with the same model tag reproduces the
    outputs. No key, no billing, no rate limit, no model deprecation.
  * **A genuinely independent judge.** The drafter is Qwen (local) and the judge
    is Gemini (hosted) -- different vendor, different family, different weights.
    The original design could only *measure* same-family self-preference; this
    removes most of it by construction.
  * **Volume.** Bias probes and ablations become affordable, so the evaluation
    can be thorough instead of rationed.

The cost is capability: a 4B local model is weaker than a hosted flash model at
instruction-following and JSON discipline. That shows up in the results as lower
reply quality, and the report says so plainly rather than implying the
architecture is what limits the ceiling.

Drafting on Gemini is one variable away (GROUNDSCORE_PROVIDER=gemini), and is
worth doing on a paid key; on the free tier it exhausts in a few hundred rows.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Local model roles. The judge is deliberately a different family from the
# drafter -- see the module docstring.
#
# These tags are hashed into the committed response cache, so changing a default
# here orphans every cached row that used it and `make reproduce` starts
# demanding live calls. A 6GB card has headroom for a 7-8B q4 drafter, but that
# is a cache-invalidating change, not a free upgrade: regenerate deliberately
# with `make full` rather than by editing this line.
OLLAMA_FAST = os.environ.get("GROUNDSCORE_MODEL_FAST", "qwen3:4b")
OLLAMA_JUDGE = os.environ.get("GROUNDSCORE_MODEL_JUDGE", "gemma3:4b")
OLLAMA_EMBED = os.environ.get("GROUNDSCORE_MODEL_EMBED", "nomic-embed-text")

DEFAULT_TIMEOUT = 300


class ProviderError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise ProviderError(f"ollama HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(
            f"cannot reach Ollama at {OLLAMA_HOST} ({exc.reason}). "
            "Is the Ollama app running?"
        ) from exc


def available() -> bool:
    try:
        request = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(request, timeout=5):
            return True
    except Exception:  # noqa: BLE001
        return False


def installed_models() -> list[str]:
    try:
        request = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    schema: Any = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """One completion from a local model.

    `schema` is passed to Ollama's structured-output `format` field, which
    constrains decoding to valid JSON matching the schema. That matters much
    more for a 4B model than for a hosted one: without it, small models emit
    prose around their JSON often enough to corrupt a long evaluation run.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            # Qwen3 and similar models emit <think> blocks by default, which
            # blow up latency and can swallow the JSON. Disabled explicitly.
            "num_ctx": 8192,
        },
        "think": False,
    }
    if system:
        payload["system"] = system
    if schema is not None:
        payload["format"] = schema

    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            data = _post("/api/generate", payload)
            text = (data.get("response") or "").strip()
            if not text:
                raise ProviderError("empty response from ollama")
            return text
        except ProviderError as exc:
            last = exc
            if attempt == max_retries - 1:
                break
            time.sleep(2.0 * (2 ** attempt))
    raise ProviderError(f"ollama generate failed after {max_retries} attempts: {last}")


def embed(model: str, texts: list[str], *, timeout: int = DEFAULT_TIMEOUT) -> list[list[float]]:
    data = _post("/api/embed", {"model": model, "input": texts}, timeout=timeout)
    vectors = data.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise ProviderError(
            f"ollama returned {len(vectors or [])} embeddings for {len(texts)} texts")
    return vectors


KNOWN_PROVIDERS = ("ollama", "gemini")


def provider_name() -> str:
    """Active backend: 'ollama' (default) or 'gemini'.

    An unrecognised value is rejected rather than defaulted. A typo used to
    fall through to the hosted branch, which meant a run could quietly use a
    different model than the one named in the results it wrote.
    """
    name = os.environ.get("GROUNDSCORE_PROVIDER", "ollama").strip().lower()
    if name not in KNOWN_PROVIDERS:
        raise ProviderError(
            f"GROUNDSCORE_PROVIDER={name!r} is not one of {', '.join(KNOWN_PROVIDERS)}")
    return name
