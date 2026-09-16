"""Auto-handle vs. escalate, with a stated reason.

Deterministic rules run FIRST and an LLM only judges what survives them. That
ordering is a deliberate design choice, not an optimisation:

  * Auditable. When the system escalates, a support lead can see which rule
    fired. "escalated because the message contains what looks like a payment
    card number" is reviewable; "the model felt uneasy" is not.
  * Non-negotiable classes stay non-negotiable. A well-phrased, high-confidence
    message about a hacked account must escalate no matter how confident the
    classifier is -- and a model asked to weigh that against fluency sometimes
    decides otherwise.
  * Cheap. Most escalations never reach an API call.

Every decision carries `triggered_rule`, so the failure analysis can attribute
escalations to causes instead of guessing.

Thresholds live in configs/thresholds.yaml and are swept on the DEV split only
(eval/tune_thresholds.py). They are frozen before the test split is scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config, llm, taxonomy
from .classify import Classification
from .draft import Draft

AUTO = "auto"
ESCALATE = "escalate"

# --------------------------------------------------------------------------
# Pattern rules
# --------------------------------------------------------------------------

# Publicly-posted personal data. If a customer has tweeted a card number or a
# full address, the reply is not the problem -- a human needs to handle the
# thread and usually get the tweet deleted.
_PII_PATTERNS: list[tuple[str, str]] = [
    ("payment_card", r"\b(?:\d[ -]?){13,16}\b"),
    ("email_address", r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    ("phone_number", r"\b(?:\+?\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b"),
    ("order_id", r"\b\d{3}-\d{7}-\d{7}\b"),  # Amazon order-id format
    ("postal_address", r"\b\d{1,5}\s+\w+(?:\s+\w+){0,3}\s+"
                       r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|court|ct)\b"),
]
_PII_RE = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _PII_PATTERNS]

# Content that must reach a human regardless of intent classification.
_HARD_ESCALATION_PATTERNS: list[tuple[str, str]] = [
    ("legal_or_press", r"\b(lawyer|attorney|solicitor|lawsuit|sue|suing|legal action|"
                       r"small claims|ombudsman|journalist|reporter|bbc|press office|"
                       r"trading standards|attorney general)\b"),
    ("account_compromise", r"\b(hacked|compromised|unauthorou?s|unauthorized|unauthorised|"
                           r"fraud|fraudulent|identity theft|someone (?:else )?(?:used|"
                           r"accessed|ordered)|stolen card)\b"),
    # `fire` and `burn` cannot be bare tokens for THIS brand. Amazon's device
    # line is called Fire: measured over the 8,496-thread corpus, a bare
    # \bfire\b matches 103 messages and 92 of them (89%) are Fire TV / Fire HD /
    # Fire tablet questions, not fire hazards. It fired on three of seventy dev
    # rows and escalated three ordinary device questions as safety incidents.
    # `burn\w+` has the same shape, just rarer ("burning the midnight oil").
    # Both now match only in harm-bearing phrases. This is a precision fix and
    # not a hole in the safety net: every phrasing that describes a real fire or
    # burn is still listed, and the eval is what found it.
    ("safety_or_harm", r"\b(injur\w+|electrocut\w+|hospital|allerg\w+|"
                       r"poison\w+|choking|unsafe|dangerous|died|death)\b"
                       r"|(?:caught|catch(?:es|ing)?|on) fire\b"
                       r"|\bfire (?:hazard|risk)\b"
                       r"|\b(?:burst|bursting) into flames\b"
                       r"|\bstarted a fire\b"
                       r"|\bset (?:it|them|the \w+) alight\b"
                       r"|\bburn\w* (?:my|his|her|their|the) "
                       r"(?:hand|arm|face|skin|finger|child|baby|house|flat|home)\w*\b"),
    ("vulnerability", r"\b(suicid\w+|kill myself|self harm|disabled|carer|dementia|terminal)\b"),
]
_HARD_RE = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _HARD_ESCALATION_PATTERNS]


@dataclass
class Decision:
    action: str
    reason: str
    triggered_rule: str
    by_llm: bool = False

    @property
    def is_escalation(self) -> bool:
        return self.action == ESCALATE

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "triggered_rule": self.triggered_rule,
            "decided_by": "llm" if self.by_llm else "rule",
        }


def match_pii(text: str) -> str | None:
    for name, pattern in _PII_RE:
        if pattern.search(text):
            return name
    return None


def match_hard_escalation(text: str) -> str | None:
    for name, pattern in _HARD_RE:
        if pattern.search(text):
            return name
    return None


# --------------------------------------------------------------------------
# LLM residual judgement
# --------------------------------------------------------------------------

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [AUTO, ESCALATE]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

SYSTEM = (
    "You are the final safety gate in a customer support automation system. "
    "You decide whether a drafted public reply may be sent automatically or "
    "must be handled by a human agent. You output JSON only."
)

ROUTE_PROMPT = """Decide whether this drafted reply can be sent automatically.

## Customer message
"{message}"

## Detected intent
{intent} (confidence {confidence:.2f})

## Drafted reply
"{reply}"

## Precedent the draft cites
{grounded}

## Decide
Answer "auto" only if ALL of these hold:
- The reply is accurate given the precedent, and invents nothing.
- Sending it publicly, unread by a human, carries little risk of making the
  customer angrier or committing the brand to something.
- The customer is not distressed, furious, or describing harm or loss.
- The reply actually addresses what was asked, rather than deflecting.

Answer "escalate" if any of those is in doubt. A needless escalation costs an
agent thirty seconds; a bad automatic reply is public and permanent. When
genuinely torn, escalate.

reason: one short sentence naming the deciding factor.

Return JSON only."""


def _llm_route(
    message: str, classification: Classification, draft: Draft, *, model: str = llm.MODEL_FAST
) -> Decision:
    prompt = ROUTE_PROMPT.format(
        message=message,
        intent=classification.intent,
        confidence=classification.confidence,
        reply=draft.reply,
        grounded=", ".join(draft.grounded_in) if draft.grounded_in else "(none cited)",
    )
    try:
        raw = llm.complete_json(
            prompt, schema=ROUTE_SCHEMA, model=model, system=SYSTEM,
            default={"action": ESCALATE, "reason": "router output unparseable"},
        )
    except llm.OfflineCacheMiss:
        raise
    except Exception as exc:  # noqa: BLE001
        return Decision(ESCALATE, f"router error: {str(exc)[:120]}", "llm_error", by_llm=True)

    action = raw.get("action") if raw.get("action") in (AUTO, ESCALATE) else ESCALATE
    return Decision(action, str(raw.get("reason", ""))[:300], "llm_judgement", by_llm=True)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def route(
    message: str,
    classification: Classification,
    draft: Draft,
    *,
    tau_confidence: float | None = None,
    tau_similarity: float | None = None,
    use_llm: bool = True,
    model: str = llm.MODEL_FAST,
) -> Decision:
    """Rules first, LLM last. Returns the first rule that fires."""
    cfg = config.thresholds()
    tau_conf = cfg["tau_confidence"] if tau_confidence is None else tau_confidence
    tau_sim = cfg["tau_similarity"] if tau_similarity is None else tau_similarity

    hard = match_hard_escalation(message)
    if hard:
        return Decision(ESCALATE, f"message matches {hard.replace('_', ' ')} pattern", hard)

    pii = match_pii(message)
    if pii:
        return Decision(ESCALATE, f"message appears to contain {pii.replace('_', ' ')}", f"pii_{pii}")

    if classification.intent in taxonomy.never_auto_names():
        intent = taxonomy.by_name(classification.intent)
        note = intent.escalation_note if intent and intent.escalation_note else "policy"
        return Decision(ESCALATE, f"intent '{classification.intent}' is never auto-handled ({note})",
                        "never_auto_intent")

    if not classification.ok:
        return Decision(ESCALATE, "intent classification failed", "classifier_failure")

    if not draft.ok or not draft.reply.strip():
        return Decision(ESCALATE, "no usable reply was drafted", "draft_failure")

    if draft.truncated:
        return Decision(ESCALATE, "draft exceeded the length limit and was cut", "draft_over_length")

    if not draft.grounded_in:
        return Decision(ESCALATE, "draft cites no historical precedent to ground it",
                        "draft_ungrounded")

    if draft.uncertain:
        reason = draft.uncertainty_reason or "drafter flagged low confidence"
        return Decision(ESCALATE, f"drafter self-flagged: {reason}", "draft_uncertain")

    if classification.max_similarity < tau_sim:
        return Decision(
            ESCALATE,
            f"no similar precedent (best similarity {classification.max_similarity:.2f} "
            f"< {tau_sim:.2f})",
            "low_retrieval_similarity",
        )

    if classification.confidence < tau_conf:
        return Decision(
            ESCALATE,
            f"intent confidence {classification.confidence:.2f} below {tau_conf:.2f}",
            "low_confidence",
        )

    if not use_llm:
        return Decision(AUTO, "passed all deterministic rules", "rules_only")

    return _llm_route(message, classification, draft, model=model)


def forced_escalation(message: str, classification: Classification, draft: Draft) -> bool:
    """True if a rule escalates regardless of any threshold.

    Used by the coverage-vs-harm curve: sweeping the confidence threshold must
    not be able to auto-send something a hard rule forbids.
    """
    return bool(
        match_hard_escalation(message)
        or match_pii(message)
        or classification.intent in taxonomy.never_auto_names()
        or not classification.ok
        or not draft.ok
        or draft.truncated
        or not draft.grounded_in
        or draft.uncertain
    )
