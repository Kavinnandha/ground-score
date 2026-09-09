"""Classify and draft in a single model call.

Why this exists
---------------
`classify.py` and `draft.py` are separate stages with separate prompts, which is
the right decomposition for reasoning about failures. It is the wrong
decomposition for this API budget: the free tier meters generate_content at
roughly 5 requests/minute, and running the two stages separately doubles the
wall clock of every evaluation for no measured benefit.

The two stages share their entire context -- the customer message and the same
five retrieved exemplars -- and the drafter needs the intent anyway. Merging
them is therefore a cost decision, not a modelling one.

What is preserved:
  * the same output fields, validation and failure handling as the split stages,
  * `route.py` still runs separately, rules first, so the safety gate is not
    entangled with generation,
  * the split path (`classify.classify` + `draft.draft_reply`) is kept and still
    works, so the merge can be undone or A/B'd on a paid key.

What is given up: the model sees the drafting instructions before committing to
an intent, so the intent label is no longer independent of the reply. If the
model talks itself into a fluent reply, the intent may be rationalised to match.
That is a real coupling and it is disclosed in the report rather than hidden.
"""

from __future__ import annotations

from . import llm, taxonomy
from .classify import Classification
from .draft import MAX_REPLY_CHARS, Draft, render_exemplars
from .retrieve import Exemplar

RESPOND_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "secondary_intent": {"type": "string"},
        "reply": {"type": "string"},
        "grounded_in": {"type": "array", "items": {"type": "string"}},
        "uncertain": {"type": "boolean"},
        "uncertainty_reason": {"type": "string"},
    },
    "required": ["intent", "confidence", "rationale", "secondary_intent",
                 "reply", "grounded_in", "uncertain", "uncertainty_reason"],
}

SYSTEM = (
    "You triage and draft replies for a brand's customer support team. "
    "You output JSON only. You never promise refunds, credits, compensation, "
    "delivery dates, or policy exceptions unless the provided precedent replies "
    "show the brand making that exact promise."
)

PROMPT = """A customer has tweeted at {brand}. Classify the message and draft a reply.

## Intent taxonomy
{taxonomy}

## How {brand} has replied to similar messages
Each entry is a REAL past exchange from this brand's support queue. These are
your only evidence about what {brand} can offer and how it speaks.

{exemplars}

## Customer message
"{message}"

## Part 1 - classify
- intent: exactly one label from the taxonomy. Use "other" if none fit; do not
  stretch a class to cover a message it was not written for.
- confidence: 0.0-1.0, your probability that a careful human annotator applying
  the definitions above would choose the same label. Be honest -- use values
  below 0.5 when the message is genuinely ambiguous or too short to tell.
- secondary_intent: a second distinct ask if the message contains one
  (taxonomy label), otherwise "".
- rationale: one short sentence.

## Part 2 - draft
1. GROUNDING. Every factual claim, instruction, policy statement, timeline or
   offer must be supported by at least one precedent above. If the precedents
   do not cover what this customer is asking, do NOT improvise -- write a short
   acknowledgement that moves them to a human and set uncertain=true.
2. NEVER invent: refund amounts, delivery dates, compensation, account-specific
   facts, order status, or policy no precedent shows.
3. At most {max_chars} characters. This is a tweet.
4. Match the voice of the precedent replies.
5. Do not use the customer's @handle; the sending system adds it.
6. grounded_in: the exemplar ids (e.g. "E1", "E3") whose content actually
   supports your reply. Cite only what you used. Empty list if none, and then
   set uncertain=true.
7. uncertain: true if you would not want this sent without a human reading it.
   uncertainty_reason: one short sentence, or "".

Return JSON only."""


def build_prompt(message: str, exemplars: list[Exemplar], brand: str) -> str:
    return PROMPT.format(
        brand=brand,
        taxonomy=taxonomy.render_for_prompt(),
        exemplars=render_exemplars(exemplars),
        message=message,
        max_chars=MAX_REPLY_CHARS,
    )


def respond(
    message: str,
    exemplars: list[Exemplar],
    brand: str,
    *,
    model: str = llm.MODEL_FAST,
) -> tuple[Classification, Draft]:
    prompt = build_prompt(message, exemplars, brand)
    fallback = {
        "intent": taxonomy.OTHER, "confidence": 0.0, "rationale": "unparseable",
        "secondary_intent": "", "reply": "", "grounded_in": [],
        "uncertain": True, "uncertainty_reason": "model output unparseable",
    }
    try:
        raw = llm.complete_json(prompt, schema=RESPOND_SCHEMA, model=model,
                                system=SYSTEM, default=fallback)
    except llm.OfflineCacheMiss:
        raise
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
        return (
            Classification(taxonomy.OTHER, 0.0, "model error", ok=False,
                           error=str(exc)[:200], neighbours=exemplars),
            Draft("", [], uncertain=True, uncertainty_reason="model error",
                  ok=False, error=str(exc)[:200]),
        )

    valid = set(taxonomy.names())
    intent = str(raw.get("intent", "")).strip()
    if intent in valid:
        try:
            confidence = min(max(float(raw.get("confidence", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.0
        secondary = str(raw.get("secondary_intent", "") or "").strip()
        classification = Classification(
            intent=intent,
            confidence=confidence,
            rationale=str(raw.get("rationale", ""))[:300],
            secondary_intent=secondary if secondary in valid else "",
            neighbours=exemplars,
        )
    else:
        # An out-of-taxonomy label is a real failure, recorded rather than
        # silently coerced, and routed to a human via confidence 0.
        classification = Classification(
            taxonomy.OTHER, 0.0,
            f"model returned out-of-taxonomy label {intent!r}",
            ok=False, error=f"invalid_label:{intent}", neighbours=exemplars,
        )

    reply = str(raw.get("reply", "")).strip()
    truncated = False
    if len(reply) > MAX_REPLY_CHARS:
        reply = reply[:MAX_REPLY_CHARS].rstrip()
        truncated = True

    draft = Draft(
        reply=reply,
        grounded_in=[str(g) for g in raw.get("grounded_in", []) if str(g).strip()],
        uncertain=bool(raw.get("uncertain", False)),
        uncertainty_reason=str(raw.get("uncertainty_reason", ""))[:300],
        truncated=truncated,
        ok=bool(reply),
        error="" if reply else "empty_reply",
    )
    return classification, draft
