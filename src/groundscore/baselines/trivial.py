"""Trivial baselines. No learning, no retrieval, no LLM.

These exist to answer "how much of the headline number is the system, and how
much is the dataset being easy?" Two variants, because the two routing failure
modes have opposite trivial optima:

  always_auto      majority intent + one canned reply, never escalates.
                   Maximum coverage, and its false-auto rate is exactly the
                   base rate of escalation-worthy traffic.
  always_escalate  never sends anything. Zero customer-facing harm and zero
                   business value -- the degenerate point that any "we keep
                   harm below X%" claim must be measured against.

The canned reply is not a strawman. It is a lightly-edited real AmazonHelp
reply pattern, and on a support queue whose most common answer is a polite
acknowledgement plus a pointer, a fixed string scores better than it has any
right to. That result belongs in the report rather than hidden -- it is the
main evidence for how much of the agent's reply-quality score is genuinely
earned.
"""

from __future__ import annotations

from collections import Counter

from .. import taxonomy
from ..agent import AgentOutput

CANNED_REPLY = (
    "Sorry for the trouble! We'd like to help. Please share more details "
    "so we can look into this for you."
)


class TrivialBaseline:
    """Majority intent + fixed reply + fixed routing decision."""

    def __init__(self, majority_intent: str, *, always_escalate: bool = False,
                 name: str | None = None):
        self.majority_intent = majority_intent
        self.always_escalate = always_escalate
        self.name = name or ("trivial_always_escalate" if always_escalate else "trivial_always_auto")

    @classmethod
    def from_labels(cls, labels: list[str], **kwargs) -> "TrivialBaseline":
        """Fit the majority class on the DEV split only.

        Fitting it on the test split would leak the test distribution into the
        baseline and make the trivial system look stronger than it is.
        """
        majority = Counter(labels).most_common(1)[0][0] if labels else taxonomy.OTHER
        return cls(majority, **kwargs)

    def handle(self, thread_id: str, message: str) -> AgentOutput:
        action = "escalate" if self.always_escalate else "auto"
        reason = ("fixed policy: escalate everything" if self.always_escalate
                  else "fixed policy: auto-handle everything")
        return AgentOutput(
            thread_id=thread_id,
            customer_msg=message,
            intent=self.majority_intent,
            confidence=1.0,
            secondary_intent="",
            rationale="majority class",
            reply="" if self.always_escalate else CANNED_REPLY,
            grounded_in=[],
            action=action,
            reason=reason,
            triggered_rule="fixed_policy",
            decided_by="rule",
            max_similarity=0.0,
            diagnostics={"system": self.name},
        )

    def run(self, items, on_progress=None) -> list[AgentOutput]:
        items = list(items)
        out = [self.handle(tid, msg) for tid, msg in items]
        if on_progress:
            on_progress(len(items), len(items))
        return out
