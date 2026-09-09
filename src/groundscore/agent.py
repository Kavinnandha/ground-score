"""The support agent: retrieve -> classify -> draft -> route.

Kept deliberately thin. Each stage is its own module with its own prompt and
its own failure mode, so the evaluation can attribute a bad outcome to a stage
rather than to "the agent". The ablation baseline (`no_retrieval`) is the same
pipeline with the retrieval step disabled, which is what makes the contribution
of grounding measurable instead of asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import classify as classify_mod
from . import config
from . import draft as draft_mod
from . import llm, respond, route
from .retrieve import Exemplar, Retriever


@dataclass
class AgentOutput:
    thread_id: str
    customer_msg: str
    intent: str
    confidence: float
    secondary_intent: str
    rationale: str
    reply: str
    grounded_in: list[str]
    action: str
    reason: str
    triggered_rule: str
    decided_by: str
    max_similarity: float
    exemplars: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "customer_msg": self.customer_msg,
            "intent": self.intent,
            "confidence": self.confidence,
            "secondary_intent": self.secondary_intent,
            "rationale": self.rationale,
            "reply": self.reply,
            "grounded_in": self.grounded_in,
            "action": self.action,
            "reason": self.reason,
            "triggered_rule": self.triggered_rule,
            "decided_by": self.decided_by,
            "max_similarity": self.max_similarity,
            "exemplars": self.exemplars,
            "diagnostics": self.diagnostics,
        }


class SupportAgent:
    def __init__(
        self,
        retriever: Retriever | None,
        *,
        brand: str | None = None,
        top_k: int | None = None,
        use_llm_router: bool = True,
        model: str = llm.MODEL_FAST,
        name: str = "agent",
        combined: bool = True,
    ):
        self.retriever = retriever
        self.brand = brand or config.brand()
        self.top_k = top_k if top_k is not None else config.brand_config()["retrieval"]["top_k"]
        self.use_llm_router = use_llm_router
        self.model = model
        self.name = name
        # One call for classify+draft instead of two. See respond.py -- this is
        # an API-budget decision, and the split path below still works.
        self.combined = combined

    def _retrieve(self, message: str) -> list[Exemplar]:
        if self.retriever is None:
            return []
        return self.retriever.search(message, k=self.top_k)

    def handle(self, thread_id: str, message: str) -> AgentOutput:
        exemplars = self._retrieve(message)

        if self.combined:
            classification, draft = respond.respond(
                message, exemplars, self.brand, model=self.model)
        else:
            classification = classify_mod.classify(
                message, exemplars, self.brand, model=self.model)
            draft = draft_mod.draft_reply(
                message, exemplars, self.brand,
                classification.intent, classification.confidence, model=self.model,
            )
        decision = route.route(
            message, classification, draft,
            use_llm=self.use_llm_router, model=self.model,
        )

        return AgentOutput(
            thread_id=thread_id,
            customer_msg=message,
            intent=classification.intent,
            confidence=classification.confidence,
            secondary_intent=classification.secondary_intent,
            rationale=classification.rationale,
            reply=draft.reply,
            grounded_in=draft.grounded_in,
            action=decision.action,
            reason=decision.reason,
            triggered_rule=decision.triggered_rule,
            decided_by="llm" if decision.by_llm else "rule",
            max_similarity=round(classification.max_similarity, 4),
            exemplars=[e.as_dict() for e in exemplars],
            diagnostics={
                "system": self.name,
                "combined_call": self.combined,
                "classification": classification.as_dict(),
                "draft": draft.as_dict(),
                "forced_escalation": route.forced_escalation(message, classification, draft),
            },
        )

    def run(
        self,
        items: Iterable[tuple[str, str]],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[AgentOutput]:
        items = list(items)
        out = []
        for i, (thread_id, message) in enumerate(items, start=1):
            out.append(self.handle(thread_id, message))
            if on_progress:
                on_progress(i, len(items))
        return out


def write_outputs(outputs: list[AgentOutput], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for out in outputs:
            fh.write(json.dumps(out.as_dict(), ensure_ascii=False) + "\n")


def read_outputs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
