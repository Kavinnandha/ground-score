"""Intent classification.

Retrieved neighbours are included in the prompt as dynamic few-shot context.
They are shown as *similar past messages*, not as labelled examples, because
the history is unlabelled -- presenting them as labels would be fabricating an
answer key. What they actually supply is a sense of the brand's traffic
distribution, which measurably reduces the model's tendency to invent classes
that sound plausible but do not exist in this brand's inbound.

A note on `confidence`
----------------------
The model self-reports a confidence, and self-reported LLM confidence is known
to be poorly calibrated -- it clusters near 0.9 and is closer to a fluency
signal than a probability. It is used here anyway, as ONE input to routing,
because the alternative (no confidence signal) is worse. Its calibration is
measured explicitly in the evaluation (reliability curve in results/) rather
than assumed, and the report treats a miscalibrated confidence as a known
weakness rather than a solved problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import llm, taxonomy
from .retrieve import Exemplar

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "secondary_intent": {"type": "string"},
    },
    "required": ["intent", "confidence", "rationale", "secondary_intent"],
}

SYSTEM = (
    "You are an intent classifier for a customer support triage system. "
    "You output JSON only. You never invent intent labels outside the provided taxonomy."
)

PROMPT = """Classify this customer message sent to {brand} on Twitter.

## Intent taxonomy
{taxonomy}

## Similar past messages to {brand}
These are real messages from the same support queue, retrieved by similarity.
They are UNLABELLED -- use them only to judge what this brand's traffic looks
like. Do not assume they share this message's intent.

{neighbours}

## Message to classify
"{message}"

## Instructions
- Choose exactly one intent from the taxonomy above. Use "other" if none fit;
  do not stretch a class to cover a message it was not written for.
- confidence: 0.0-1.0, your probability that a careful human annotator applying
  the definitions above would choose the same label. Be honest: use values
  below 0.5 when the message is genuinely ambiguous or too short to tell.
- secondary_intent: if the message contains a second distinct ask, name it
  (taxonomy label). Otherwise the empty string. Multi-intent messages are a
  known failure source, so this field is recorded even though the headline
  metric is single-label.
- rationale: one short sentence.

Return JSON only."""


@dataclass
class Classification:
    intent: str
    confidence: float
    rationale: str
    secondary_intent: str = ""
    ok: bool = True
    error: str = ""
    neighbours: list[Exemplar] = field(default_factory=list)

    @property
    def max_similarity(self) -> float:
        return max((e.similarity for e in self.neighbours), default=0.0)

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "secondary_intent": self.secondary_intent,
            "max_similarity": round(self.max_similarity, 4),
            "ok": self.ok,
            "error": self.error,
        }


def _render_neighbours(neighbours: list[Exemplar]) -> str:
    if not neighbours:
        return "(no similar messages found)"
    return "\n".join(
        f'{i + 1}. (similarity {e.similarity:.2f}) "{e.customer_msg}"'
        for i, e in enumerate(neighbours)
    )


def build_prompt(message: str, neighbours: list[Exemplar], brand: str) -> str:
    return PROMPT.format(
        brand=brand,
        taxonomy=taxonomy.render_for_prompt(),
        neighbours=_render_neighbours(neighbours),
        message=message,
    )


def classify(
    message: str,
    neighbours: list[Exemplar],
    brand: str,
    *,
    model: str = llm.MODEL_FAST,
) -> Classification:
    prompt = build_prompt(message, neighbours, brand)
    fallback = {"intent": taxonomy.OTHER, "confidence": 0.0,
                "rationale": "classifier failed", "secondary_intent": ""}
    try:
        raw = llm.complete_json(prompt, schema=CLASSIFY_SCHEMA, model=model,
                                system=SYSTEM, default=fallback)
    except llm.OfflineCacheMiss:
        raise
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill a 200-row run
        return Classification(taxonomy.OTHER, 0.0, "classifier error", ok=False,
                              error=str(exc)[:200], neighbours=neighbours)

    intent = str(raw.get("intent", "")).strip()
    valid = set(taxonomy.names())
    if intent not in valid:
        # A hallucinated label is a real failure, not something to silently
        # coerce. It is mapped to `other` with confidence 0 so routing escalates
        # it, and the error is recorded so it shows up in the failure analysis.
        return Classification(
            taxonomy.OTHER, 0.0,
            f"model returned out-of-taxonomy label {intent!r}",
            ok=False, error=f"invalid_label:{intent}", neighbours=neighbours,
        )

    try:
        confidence = min(max(float(raw.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0

    secondary = str(raw.get("secondary_intent", "") or "").strip()
    if secondary not in valid:
        secondary = ""

    return Classification(
        intent=intent,
        confidence=confidence,
        rationale=str(raw.get("rationale", ""))[:300],
        secondary_intent=secondary,
        neighbours=neighbours,
    )
