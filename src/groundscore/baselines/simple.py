"""Simple baseline: TF-IDF classifier + verbatim nearest-neighbour reply.

No LLM anywhere in this system. It answers the question the trivial baseline
cannot: how much of the agent's benefit needs a language model at all?

  intent  TF-IDF + logistic regression, trained on the DEV split's human
          labels. Dev-only training is what keeps the comparison honest -- the
          agent never sees test labels either.
  reply   the brand's actual historical reply to the most similar past message,
          copied verbatim. This is a strong baseline for grounding by
          construction: a real reply from this brand cannot hallucinate policy.
          What it cannot do is be *relevant* when the nearest neighbour is only
          superficially similar, which is exactly where the LLM should earn its
          cost.
  route   thresholds only -- escalate on low classifier probability or low
          retrieval similarity. Same deterministic rules as the real router, so
          the comparison isolates the LLM's contribution to routing.

Because the reply is a real human reply, this baseline sets a meaningful
ceiling on tone and a meaningful floor on relevance.
"""

from __future__ import annotations

import numpy as np

from .. import route as route_mod
from .. import taxonomy
from ..agent import AgentOutput
from ..classify import Classification
from ..draft import Draft
from ..retrieve import Retriever


class SimpleBaseline:
    name = "simple_tfidf_nn"

    def __init__(self, retriever: Retriever, *, top_k: int = 5, seed: int = 42):
        self.retriever = retriever
        self.top_k = top_k
        self.seed = seed
        self.pipeline = None
        self.classes_: list[str] = []

    def fit(self, messages: list[str], labels: list[str]) -> "SimpleBaseline":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        # Word + char n-grams: char n-grams carry most of the signal on noisy
        # tweets (typos, elongations, missing spaces) where word features miss.
        self.pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                            sublinear_tf=True, max_features=100_000),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.seed),
        )
        self.pipeline.fit(messages, labels)
        self.classes_ = list(self.pipeline.classes_)
        return self

    def handle(self, thread_id: str, message: str) -> AgentOutput:
        if self.pipeline is None:
            raise RuntimeError("SimpleBaseline.fit() must be called before handle()")

        probabilities = self.pipeline.predict_proba([message])[0]
        best = int(np.argmax(probabilities))
        intent, confidence = self.classes_[best], float(probabilities[best])

        exemplars = self.retriever.search(message, k=self.top_k)
        top = exemplars[0] if exemplars else None
        reply = top.brand_reply if top else ""
        similarity = top.similarity if top else 0.0

        classification = Classification(
            intent=intent, confidence=confidence,
            rationale="tfidf+logreg", neighbours=exemplars,
        )
        # grounded_in cites the retrieved thread because the reply IS that
        # thread's reply -- grounding is exact here, not claimed.
        draft = Draft(reply=reply, grounded_in=["E1"] if top else [], ok=bool(reply))

        decision = route_mod.route(message, classification, draft, use_llm=False)

        return AgentOutput(
            thread_id=thread_id,
            customer_msg=message,
            intent=intent,
            confidence=confidence,
            secondary_intent="",
            rationale="tfidf+logreg",
            reply=reply,
            grounded_in=draft.grounded_in,
            action=decision.action,
            reason=decision.reason,
            triggered_rule=decision.triggered_rule,
            decided_by="rule",
            max_similarity=round(similarity, 4),
            exemplars=[e.as_dict() for e in exemplars],
            diagnostics={
                "system": self.name,
                "copied_from_thread": top.thread_id if top else None,
                "forced_escalation": route_mod.forced_escalation(message, classification, draft),
            },
        )

    def run(self, items, on_progress=None) -> list[AgentOutput]:
        items = list(items)
        out = [self.handle(tid, msg) for tid, msg in items]
        if on_progress:
            on_progress(len(items), len(items))
        return out


def majority_fallback(labels: list[str]) -> str:
    from collections import Counter

    return Counter(labels).most_common(1)[0][0] if labels else taxonomy.OTHER
