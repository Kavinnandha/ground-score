"""Automated metrics for intent classification and routing.

Design notes that matter for reading the results
------------------------------------------------
1. Routing accuracy is deliberately NOT the headline. The two routing errors
   have wildly different costs: auto-sending a reply to a customer who needed a
   human is a customer-facing failure, while escalating something the bot could
   have handled costs an agent thirty seconds. A single accuracy number averages
   those together and hides exactly the thing a support lead would ask about.
   So routing is reported as a (coverage, false-auto-rate) pair and as a curve
   over the confidence threshold.

2. `false_auto_rate` is defined as a share of what we auto-sent, not a share of
   what should have escalated. "3% of the replies we sent were ones a human
   should have handled" is the number that maps to customer harm. The
   recall-style view (`escalation_miss_rate`) is reported alongside it, because
   the two diverge sharply when coverage is low.

3. Every headline number carries a bootstrap CI. With a 120-example test split,
   the CI is wide enough that most "improvements" under ~10 points are not
   distinguishable from noise, and the report says so rather than quietly
   reporting three significant figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

AUTO = "auto"
ESCALATE = "escalate"


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    *,
    resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI over item indices.

    `statistic` receives an array of indices, so callers keep their own paired
    data structures instead of this module guessing at their shape.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for i in range(resamples):
        values[i] = statistic(rng.integers(0, n, size=n))
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _with_ci(values: list[float], stat: Callable[[np.ndarray], float], seed: int, resamples: int) -> dict:
    point = stat(np.arange(len(values)))
    lo, hi = bootstrap_ci(stat, len(values), resamples=resamples, seed=seed)
    return {"value": float(point), "ci95": [lo, hi], "n": len(values)}


# --------------------------------------------------------------------------
# Intent classification
# --------------------------------------------------------------------------

@dataclass
class IntentResult:
    accuracy: dict
    macro_f1: dict
    per_class: dict
    confusion: list[list[int]]
    labels: list[str]

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion,
            "labels": self.labels,
        }


def score_intents(
    gold: Sequence[str],
    pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    seed: int = 42,
    resamples: int = 2000,
) -> IntentResult:
    gold_arr, pred_arr = np.asarray(gold, dtype=object), np.asarray(pred, dtype=object)
    if len(gold_arr) != len(pred_arr):
        raise ValueError(f"length mismatch: {len(gold_arr)} gold vs {len(pred_arr)} pred")
    label_list = sorted(labels) if labels is not None else sorted(set(gold_arr) | set(pred_arr))

    def acc(idx: np.ndarray) -> float:
        return float((gold_arr[idx] == pred_arr[idx]).mean())

    def macro(idx: np.ndarray) -> float:
        # zero_division=0: bootstrap resamples can omit a rare class entirely.
        return float(f1_score(gold_arr[idx], pred_arr[idx], labels=label_list,
                              average="macro", zero_division=0))

    report = classification_report(
        gold_arr, pred_arr, labels=label_list, output_dict=True, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": report[lbl]["precision"],
            "recall": report[lbl]["recall"],
            "f1": report[lbl]["f1-score"],
            "support": int(report[lbl]["support"]),
        }
        for lbl in label_list if lbl in report
    }

    return IntentResult(
        accuracy=_with_ci(list(gold_arr), acc, seed, resamples),
        macro_f1=_with_ci(list(gold_arr), macro, seed, resamples),
        per_class=per_class,
        confusion=confusion_matrix(gold_arr, pred_arr, labels=label_list).tolist(),
        labels=label_list,
    )


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

@dataclass
class RoutingResult:
    coverage: dict
    false_auto_rate: dict
    escalation_miss_rate: dict
    escalation_precision: float
    escalation_recall: float
    counts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "false_auto_rate": self.false_auto_rate,
            "escalation_miss_rate": self.escalation_miss_rate,
            "escalation_precision": self.escalation_precision,
            "escalation_recall": self.escalation_recall,
            "counts": self.counts,
        }


def score_routing(
    gold: Sequence[str],
    pred: Sequence[str],
    *,
    seed: int = 42,
    resamples: int = 2000,
) -> RoutingResult:
    """Score auto/escalate decisions.

    coverage             share of messages handled without a human
    false_auto_rate      of the messages we auto-handled, share that were gold
                         'escalate' -- i.e. customer-facing harm density
    escalation_miss_rate of the messages that should have escalated, share we
                         auto-handled anyway
    """
    gold_arr = np.asarray([g == ESCALATE for g in gold])
    pred_arr = np.asarray([p == ESCALATE for p in pred])

    def coverage(idx: np.ndarray) -> float:
        return float((~pred_arr[idx]).mean())

    def false_auto(idx: np.ndarray) -> float:
        autoed = ~pred_arr[idx]
        if autoed.sum() == 0:
            return 0.0  # escalating everything sends nothing, so it harms nobody
        return float(gold_arr[idx][autoed].mean())

    def miss_rate(idx: np.ndarray) -> float:
        should = gold_arr[idx]
        if should.sum() == 0:
            return 0.0
        return float((~pred_arr[idx][should]).mean())

    tp = int((gold_arr & pred_arr).sum())
    fp = int((~gold_arr & pred_arr).sum())
    fn = int((gold_arr & ~pred_arr).sum())
    tn = int((~gold_arr & ~pred_arr).sum())

    return RoutingResult(
        coverage=_with_ci(list(gold), coverage, seed, resamples),
        false_auto_rate=_with_ci(list(gold), false_auto, seed, resamples),
        escalation_miss_rate=_with_ci(list(gold), miss_rate, seed, resamples),
        escalation_precision=tp / (tp + fp) if (tp + fp) else 0.0,
        escalation_recall=tp / (tp + fn) if (tp + fn) else 0.0,
        counts={
            "escalate_correct": tp,
            "escalate_unnecessary": fp,
            "auto_but_should_escalate": fn,
            "auto_correct": tn,
            "n": len(gold),
        },
    )


def coverage_curve(
    gold: Sequence[str],
    confidences: Sequence[float],
    forced_escalate: Sequence[bool] | None = None,
    *,
    steps: int = 41,
) -> list[dict]:
    """Sweep the confidence threshold and trace (coverage, false-auto-rate).

    This is the operating-point picture: a support lead does not want a single
    number, they want "how much can I automate before the harm rate crosses my
    tolerance". `forced_escalate` carries the deterministic rules (never-auto
    intents, PII hits) which no threshold can override.
    """
    gold_arr = np.asarray([g == ESCALATE for g in gold])
    conf = np.asarray(confidences, dtype=float)
    forced = (np.asarray(forced_escalate, dtype=bool) if forced_escalate is not None
              else np.zeros(len(gold_arr), dtype=bool))

    out = []
    for tau in np.linspace(0.0, 1.0, steps):
        auto = (conf >= tau) & ~forced
        n_auto = int(auto.sum())
        out.append({
            "threshold": round(float(tau), 4),
            "coverage": n_auto / len(gold_arr) if len(gold_arr) else 0.0,
            "false_auto_rate": float(gold_arr[auto].mean()) if n_auto else 0.0,
            "n_auto": n_auto,
        })
    return out


def best_threshold(curve: Sequence[dict], max_false_auto: float) -> dict | None:
    """Highest-coverage operating point whose harm rate stays within budget.

    Tuned on the dev split only; the chosen value is frozen into
    configs/thresholds.yaml before the test split is ever scored.
    """
    feasible = [p for p in curve if p["false_auto_rate"] <= max_false_auto]
    return max(feasible, key=lambda p: p["coverage"]) if feasible else None


# --------------------------------------------------------------------------
# Agreement (used for label quality and judge validation)
# --------------------------------------------------------------------------

def agreement(a: Sequence, b: Sequence, *, weights: str | None = None) -> dict:
    """Cohen's kappa plus raw agreement.

    weights='quadratic' for ordinal scales (the 1-5 judge dimensions), None for
    nominal ones (intent labels, auto/escalate).
    """
    a_arr, b_arr = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    raw = float((a_arr == b_arr).mean()) if len(a_arr) else float("nan")
    try:
        kappa = float(cohen_kappa_score(a_arr, b_arr, weights=weights))
    except ValueError:
        kappa = float("nan")  # degenerate: one rater used a single class
    return {"raw_agreement": raw, "cohen_kappa": kappa, "n": len(a_arr), "weights": weights}


def correlation(a: Sequence[float], b: Sequence[float]) -> dict:
    """Spearman rho, with the constant-input case handled explicitly.

    Judge dimensions saturate: if the judge gives every reply a 4 on tone, rho
    is undefined rather than 0, and reporting it as 0 would understate agreement.
    """
    from scipy.stats import spearmanr

    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a_arr) < 3 or np.all(a_arr == a_arr[0]) or np.all(b_arr == b_arr[0]):
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n": len(a_arr),
                "note": "undefined: one series is constant"}
    rho, p = spearmanr(a_arr, b_arr)
    return {"spearman_rho": float(rho), "p_value": float(p), "n": len(a_arr),
            "mean_bias": float(np.mean(b_arr - a_arr))}
