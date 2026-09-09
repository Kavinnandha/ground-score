"""Model backends: local Ollama (default) and Gemini.

Why local is the default
------------------------
The Gemini free tier meters `generate_content` at **20 requests per day, per
model** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, confirmed from the
429 body). A full evaluation of this project needs roughly a thousand calls.
Even spread across every reachable Gemini model that ceiling is ~120/day, so the
API was not a viable backend for the work regardless of how the calls were paced.

Running locally through Ollama removes the quota entirely and buys three things
the hosted path could not offer:

  * **Reproducibility.** A reviewer with the same model tag reproduces the
    outputs. No key, no billing, no rate limit, no model deprecation.
  * **A genuinely independent judge.** The drafter is Qwen and the judge is a
    different family. The original design could only *measure* same-family
    self-preference; this removes most of it by construction.
  * **Volume.** Bias probes and ablations become affordable, so the evaluation
    can be thorough instead of rationed.

The cost is capability: a 4B local model is weaker than Gemini flash at
instruction-following and JSON discipline. That shows up in the results as
lower reply quality, and the report says so plainly rather than implying the
architecture is what limits the ceiling.

Gemini remains fully supported. Set GROUNDSCORE_PROVIDER=gemini to use it.
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


def provider_name() -> str:
    """'ollama' unless explicitly overridden to 'gemini'."""
    return os.environ.get("GROUNDSCORE_PROVIDER", "ollama").strip().lower()
