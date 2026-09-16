"""How good are the golden labels? Measured, not asserted.

README.md and data/golden/LABELING_NOTES.md both promise a label-quality number
and the report leans on it, so it is computed here rather than left as a claim.

Two passes over the same 50-row subset are compared:

  pass 1  data/golden/golden_v1.jsonl          the adjudicated labels
  pass 2  data/golden/golden_v1_relabel.jsonl  a blind second pass

Which statistic this is depends entirely on who produced pass 2, and the output
says so rather than letting the reader assume:

  * `tools/label_cli.py --relabel` -> the SAME annotator, labels hidden. That is
    INTRA-annotator agreement: self-consistency, an upper bound on label
    quality, not evidence that a second person would agree.
  * `tools/second_annotator.py`    -> a DIFFERENT annotator, blind. That is
    human-versus-model sensitivity check. It is supplementary evidence, not
    inter-annotator agreement between two people.

Reported per field:
  * Cohen's kappa on intent (nominal, 12 classes)
  * Cohen's kappa on action (binary auto/escalate) -- the one routing depends on
  * raw agreement for both, since kappa on a skewed binary is easy to misread
  * a per-stratum breakdown, because the adversarial rows are meant to be hard
  * every disagreement, listed, so they can be read instead of summarised

Output: results/label_agreement.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics  # noqa: E402
from groundscore import config  # noqa: E402

RELABEL_PATH = config.GOLDEN_DIR / "golden_v1_relabel.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    first = {r["thread_id"]: r for r in load_jsonl(config.GOLDEN_PATH)}
    second = load_jsonl(RELABEL_PATH)
    if not first:
        print(f"No adjudicated labels at {config.GOLDEN_PATH}")
        return 1
    if not second:
        print(f"No second pass at {RELABEL_PATH}")
        print("Run: python tools/label_cli.py --relabel   (same annotator, intra)")
        print("  or python tools/second_annotator.py      (different annotator, inter)")
        return 1

    paired = [(first[r["thread_id"]], r) for r in second if r["thread_id"] in first]
    if not paired:
        print("no overlap between the two passes")
        return 1

    models = sorted({r.get("relabel_model", "") for _, r in paired if r.get("relabel_model")})
    kind = "human_vs_model" if models else "intra_annotator"

    a_intent = [a["intent"] for a, _ in paired]
    b_intent = [b["relabel_intent"] for _, b in paired]
    a_action = [a["action"] for a, _ in paired]
    b_action = [b["relabel_action"] for _, b in paired]

    report = {
        "n": len(paired),
        "kind": kind,
        "pass_1": "golden_v1.jsonl (manually adjudicated by the project author)",
        "pass_2": (f"golden_v1_relabel.jsonl (second annotator: {', '.join(models)})"
                   if models else "golden_v1_relabel.jsonl (same annotator, blind)"),
        "caveat": (
            "Pass 1 was manually adjudicated by the project author. Pass 2 was "
            "produced by a model, so this is a human-versus-model sensitivity "
            "check, not inter-annotator agreement between two people."
            if kind == "human_vs_model" else
            "One annotator, labels hidden on the second pass. This is "
            "self-consistency: a CEILING on label quality, not inter-annotator "
            "agreement. A second annotator would very likely agree less."
        ),
        "intent": metrics.agreement(a_intent, b_intent),
        "action": metrics.agreement(a_action, b_action),
        # Kappa on a binary label is unreadable without the marginals. If one
        # annotator escalates nearly everything, kappa collapses towards zero
        # because chance agreement is already near the observed agreement --
        # that is a fact about that annotator's prior, not about how ambiguous
        # the labels are. Reporting the base rates next to the coefficient is
        # what stops the number being read as "the labels are noise".
        "action_base_rates": {
            "pass_1_escalate": round(a_action.count("escalate") / len(a_action), 3),
            "pass_2_escalate": round(b_action.count("escalate") / len(b_action), 3),
        },
        "by_stratum": {},
        "action_confusion": {},
        "disagreements": [],
    }
    skew = report["action_base_rates"]
    if abs(skew["pass_1_escalate"] - skew["pass_2_escalate"]) >= 0.20:
        report["action_kappa_warning"] = (
            f"The two passes escalate at very different rates "
            f"({skew['pass_1_escalate']:.0%} vs {skew['pass_2_escalate']:.0%}). Cohen's "
            f"kappa on the action is therefore driven mostly by that prior and not by "
            f"label ambiguity; read raw_agreement and action_confusion instead, and "
            f"treat the intent kappa as the usable label-quality figure."
        )

    for stratum in sorted({a["stratum"] for a, _ in paired}):
        sub = [(a, b) for a, b in paired if a["stratum"] == stratum]
        report["by_stratum"][stratum] = {
            "n": len(sub),
            "intent": metrics.agreement([a["intent"] for a, _ in sub],
                                        [b["relabel_intent"] for _, b in sub]),
            "action": metrics.agreement([a["action"] for a, _ in sub],
                                        [b["relabel_action"] for _, b in sub]),
        }

    report["action_confusion"] = {
        f"{x}->{y}": n for (x, y), n in
        Counter(zip(a_action, b_action)).most_common()
    }

    for a, b in paired:
        if a["intent"] != b["relabel_intent"] or a["action"] != b["relabel_action"]:
            report["disagreements"].append({
                "thread_id": a["thread_id"],
                "stratum": a["stratum"],
                "customer_msg": a["customer_msg"][:160],
                "pass_1": {"intent": a["intent"], "action": a["action"],
                           "confidence": a.get("annotator_confidence")},
                "pass_2": {"intent": b["relabel_intent"], "action": b["relabel_action"],
                           "confidence": b.get("relabel_annotator_confidence")},
                "pass_1_note": a.get("note", "")[:200],
                "pass_2_note": b.get("relabel_note", "")[:200],
            })

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "label_agreement.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{kind.replace('_', '-')} agreement on n={report['n']}")
    print(f"  intent   kappa={report['intent']['cohen_kappa']:+.3f}  "
          f"raw={report['intent']['raw_agreement']:.3f}")
    print(f"  action   kappa={report['action']['cohen_kappa']:+.3f}  "
          f"raw={report['action']['raw_agreement']:.3f}  "
          f"(escalate rate {skew['pass_1_escalate']:.0%} vs {skew['pass_2_escalate']:.0%})")
    if "action_kappa_warning" in report:
        print(f"  ! {report['action_kappa_warning']}")
    for stratum, stats in report["by_stratum"].items():
        print(f"  {stratum:13s} n={stats['n']:2d}  "
              f"intent raw={stats['intent']['raw_agreement']:.3f}  "
              f"action raw={stats['action']['raw_agreement']:.3f}")
    print(f"  {len(report['disagreements'])} disagreements listed")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
