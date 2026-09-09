"""Reply drafting, grounded in retrieved historical replies.

The grounding contract
----------------------
The drafter is given the brand's real past replies to similar messages and told
it may not assert anything those replies do not support. That constraint is
enforced in two places:

  * in the prompt, as an explicit rule with the failure cases named, and
  * in the output schema, via `grounded_in` -- the drafter must cite which
    exemplars support its reply.

`grounded_in` is what turns "is this reply grounded?" from a subjective judgment
into something checkable. An empty citation list is treated as a routing signal
(escalate) rather than as a formatting nit, because a reply the model cannot
trace to any precedent is exactly the reply most likely to invent a refund
policy the brand does not have.

This does not *prove* groundedness -- a model can cite an exemplar and still
contradict it. That residual is what the LLM judge's groundedness dimension is
for, and its agreement with a human is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import llm
from .cleaning import is_deflection
from .retrieve import Exemplar

MAX_REPLY_CHARS = 280

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "grounded_in": {"type": "array", "items": {"type": "string"}},
        "uncertain": {"type": "boolean"},
        "uncertainty_reason": {"type": "string"},
    },
    "required": ["reply", "grounded_in", "uncertain", "uncertainty_reason"],
}

SYSTEM = (
    "You draft public Twitter replies for a brand's customer support team. "
    "You output JSON only. You never promise refunds, credits, compensation, "
    "delivery dates, or policy exceptions unless the provided precedent replies "
    "show the brand making that exact promise."
)

PROMPT = """Draft a public Twitter reply from {brand} to this customer.

## How {brand} has replied to similar messages
Each entry is a REAL past exchange from this brand's support queue.

{exemplars}

## Customer message
"{message}"

## Detected intent
{intent} (classifier confidence {confidence:.2f})

## Rules
1. GROUNDING. Every factual claim, instruction, policy statement, timeline or
   offer in your reply must be supported by at least one precedent above. If
   the precedents do not cover what this customer is asking, do NOT improvise
   an answer -- write a short acknowledgement that moves them to a human, and
   set uncertain=true.
2. NEVER invent: refund amounts, delivery dates, compensation, account-specific
   facts, order status, or policy that no precedent shows.
3. Length: at most {max_chars} characters. This is a tweet.
4. Match the voice of the precedent replies -- their formality, their greeting
   style, their use of names or initials.
5. Do not use the customer's @handle; it is added by the sending system.
6. grounded_in: list the exemplar ids (e.g. "E1", "E3") whose content actually
   supports your reply. Cite only what you used. If you used none, return an
   empty list and set uncertain=true.
7. uncertain: true if you are not confident this reply is safe to send without
   a human reading it first. uncertainty_reason: one short sentence, or "".

Return JSON only."""


@dataclass
class Draft:
    reply: str
    grounded_in: list[str] = field(default_factory=list)
    uncertain: bool = False
    uncertainty_reason: str = ""
    ok: bool = True
    error: str = ""
    truncated: bool = False

    @property
    def is_deflection(self) -> bool:
        return is_deflection(self.reply)

    def as_dict(self) -> dict:
        return {
            "reply": self.reply,
            "grounded_in": self.grounded_in,
            "uncertain": self.uncertain,
            "uncertainty_reason": self.uncertainty_reason,
            "is_deflection": self.is_deflection,
            "chars": len(self.reply),
            "truncated": self.truncated,
            "ok": self.ok,
            "error": self.error,
        }


def render_exemplars(exemplars: list[Exemplar]) -> str:
    if not exemplars:
        return "(no similar past exchanges found)"
    blocks = []
    for i, e in enumerate(exemplars, start=1):
        blocks.append(
            f'E{i} (similarity {e.similarity:.2f})\n'
            f'  customer: "{e.customer_msg}"\n'
            f'  {"brand"}: "{e.brand_reply}"'
        )
    return "\n\n".join(blocks)


def build_prompt(message: str, exemplars: list[Exemplar], brand: str,
                 intent: str, confidence: float) -> str:
    return PROMPT.format(
        brand=brand,
        exemplars=render_exemplars(exemplars),
        message=message,
        intent=intent,
        confidence=confidence,
        max_chars=MAX_REPLY_CHARS,
    )


def draft_reply(
    message: str,
    exemplars: list[Exemplar],
    brand: str,
    intent: str,
    confidence: float,
    *,
    model: str = llm.MODEL_FAST,
) -> Draft:
    prompt = build_prompt(message, exemplars, brand, intent, confidence)
    fallback = {"reply": "", "grounded_in": [], "uncertain": True,
                "uncertainty_reason": "drafter output unparseable"}
    try:
        raw = llm.complete_json(prompt, schema=DRAFT_SCHEMA, model=model,
                                system=SYSTEM, default=fallback)
    except llm.OfflineCacheMiss:
        raise
    except Exception as exc:  # noqa: BLE001
        return Draft("", [], uncertain=True, uncertainty_reason="drafter error",
                     ok=False, error=str(exc)[:200])

    reply = str(raw.get("reply", "")).strip()
    truncated = False
    if len(reply) > MAX_REPLY_CHARS:
        # Over-length is recorded, not silently trimmed away: it is a rule
        # violation and route.py treats it as a reason to escalate.
        reply = reply[:MAX_REPLY_CHARS].rstrip()
        truncated = True

    grounded = [str(g) for g in raw.get("grounded_in", []) if str(g).strip()]

    return Draft(
        reply=reply,
        grounded_in=grounded,
        uncertain=bool(raw.get("uncertain", False)),
        uncertainty_reason=str(raw.get("uncertainty_reason", ""))[:300],
        truncated=truncated,
        ok=bool(reply),
        error="" if reply else "empty_reply",
    )
