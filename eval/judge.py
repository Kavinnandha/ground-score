"""LLM-as-judge for reply quality.

The rubric lives in eval/judge_rubric.md and is read at runtime, so the judge
prompt and the instructions given to the human validator cannot drift apart.

Blinding
--------
The judge is never told which system produced a reply. System identity is
stripped before the prompt is built, and the cache key therefore depends only
on (rubric, customer message, precedents, reply) -- meaning two systems that
happen to produce an identical reply get the identical score, for free, which
is a small but real guard against inconsistent grading.

Known limits, measured rather than asserted (see judge_agreement.py):
  * The judge is a Gemini model grading Gemini-written replies. A cross-family
    judge (Gemma) scores a subset so that self-preference can be estimated.
  * LLM judges are known to reward verbosity. A length-perturbation probe
    quantifies it here rather than assuming it away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundscore import llm  # noqa: E402

RUBRIC_PATH = Path(__file__).resolve().parent / "judge_rubric.md"

DIMENSIONS = ("groundedness", "resolution", "tone_fit", "safety")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness": {"type": "integer"},
        "resolution": {"type": "integer"},
        "tone_fit": {"type": "integer"},
        "safety": {"type": "integer"},
        "would_send": {"type": "boolean"},
        "justification": {"type": "string"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["groundedness", "resolution", "tone_fit", "safety",
                 "would_send", "justification", "unsupported_claims"],
}

SYSTEM = (
    "You are a strict quality reviewer for customer support replies. "
    "You apply the provided rubric literally and you output JSON only. "
    "You do not know or care which system wrote the reply."
)

PROMPT = """{rubric}

---

# Case to score

## Customer message
"{message}"

## Precedent replies from this brand to similar messages
These define what the reply is allowed to claim.

{exemplars}

## Reply under review
"{reply}"

---

Score the reply against the four dimensions and the binary gate.

- unsupported_claims: quote any specific claim in the reply that no precedent
  supports. Empty list if there are none. Be concrete -- quote the words.
- justification: two sentences maximum, naming the deciding factor.

Return JSON only."""


@lru_cache(maxsize=1)
def rubric_text() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8")


@dataclass
class JudgeScore:
    groundedness: int
    resolution: int
    tone_fit: int
    safety: int
    would_send: bool
    justification: str = ""
    unsupported_claims: list[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""

    @property
    def mean_score(self) -> float:
        return sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS)

    def as_dict(self) -> dict:
        return {
            **{d: getattr(self, d) for d in DIMENSIONS},
            "mean_score": round(self.mean_score, 3),
            "would_send": self.would_send,
            "justification": self.justification,
            "unsupported_claims": self.unsupported_claims,
            "ok": self.ok,
            "error": self.error,
        }


EMPTY_REPLY_SCORE = JudgeScore(1, 1, 1, 1, False, "empty reply", [])


def render_exemplars(exemplars: list[dict]) -> str:
    if not exemplars:
        return "(no precedents were retrieved for this message)"
    return "\n\n".join(
        f'E{i} (similarity {e.get("similarity", 0):.2f})\n'
        f'  customer: "{e.get("customer_msg", "")}"\n'
        f'  brand: "{e.get("brand_reply", "")}"'
        for i, e in enumerate(exemplars, start=1)
    )


def build_prompt(message: str, reply: str, exemplars: list[dict]) -> str:
    return PROMPT.format(
        rubric=rubric_text(),
        message=message,
        exemplars=render_exemplars(exemplars),
        reply=reply,
    )


def _clamp(value, low: int = 1, high: int = 5) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return low


def judge_reply(
    message: str,
    reply: str,
    exemplars: list[dict],
    *,
    model: str = llm.MODEL_JUDGE,
) -> JudgeScore:
    """Score one reply. Blind to the system that produced it."""
    if not reply or not reply.strip():
        return EMPTY_REPLY_SCORE

    prompt = build_prompt(message, reply, exemplars)
    fallback = {d: 1 for d in DIMENSIONS} | {
        "would_send": False, "justification": "judge output unparseable",
        "unsupported_claims": [],
    }
    try:
        raw = llm.complete_json(prompt, schema=JUDGE_SCHEMA, model=model,
                                system=SYSTEM, default=fallback)
    except llm.OfflineCacheMiss:
        raise
    except Exception as exc:  # noqa: BLE001
        return JudgeScore(1, 1, 1, 1, False, "judge error", [], ok=False, error=str(exc)[:200])

    return JudgeScore(
        groundedness=_clamp(raw.get("groundedness")),
        resolution=_clamp(raw.get("resolution")),
        tone_fit=_clamp(raw.get("tone_fit")),
        safety=_clamp(raw.get("safety")),
        would_send=bool(raw.get("would_send", False)),
        justification=str(raw.get("justification", ""))[:400],
        unsupported_claims=[str(c)[:200] for c in raw.get("unsupported_claims", [])][:5],
    )


def judge_outputs(
    outputs: list[dict],
    *,
    model: str = llm.MODEL_JUDGE,
    on_progress=None,
) -> list[dict]:
    """Judge a list of agent output dicts, preserving thread_id for joining."""
    scored = []
    for i, out in enumerate(outputs, start=1):
        score = judge_reply(
            out["customer_msg"], out.get("reply", ""), out.get("exemplars", []), model=model,
        )
        scored.append({
            "thread_id": out["thread_id"],
            # Which model actually produced this score. The judge role has a
            # provider fallback chain, so a quota failure part-way through a
            # split can mean two different judges scored one table. Recorded
            # per row so that is visible instead of silently averaged.
            # None means no model was consulted (empty reply, scored by rule).
            "judge_model": (None if score is EMPTY_REPLY_SCORE
                            else llm.SERVING.get("judge", model)),
            "system": out.get("diagnostics", {}).get("system", "unknown"),
            "reply": out.get("reply", ""),
            **score.as_dict(),
        })
        if on_progress:
            on_progress(i, len(outputs))
    return scored


def write_scores(scores: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for score in scores:
            fh.write(json.dumps(score, ensure_ascii=False) + "\n")
