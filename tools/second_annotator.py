"""Independent blind second annotation of the golden-set subset.

`tools/label_cli.py --relabel` re-presents 50 examples to the SAME annotator
with the labels hidden, which measures self-consistency and gives a ceiling on
label quality. This script does the other half: it puts a genuinely separate
annotator on the same 50 examples, so the number reported is agreement between
two annotators rather than one annotator with themselves.

What "independent" means here, precisely
----------------------------------------
The primary labels are human-authored; two models supply supporting signals:

  * the weak labeller that pre-filled the proposals (`qwen3:4b`, local),
  * the project author, who adjudicated `golden_v1.jsonl`,
  * this second annotator, which runs on the `@judge` role chain -- Gemini by
    default, a third vendor and family.

So this output is a blind human-versus-model sensitivity check. It is useful for
finding model-specific disagreement, but it is not a replacement for a second
human annotation. See data/golden/LABELING_NOTES.md.

What is hidden from this annotator
----------------------------------
  * the weak label and its rationale,
  * the first-pass adjudicated label, action and note,
  * the retrieved neighbours,
  * the brand's own historical reply.

It sees the customer message and the taxonomy, which is what the written
guideline gives a human annotator. The subset is chosen by exactly the same
stable hash `label_cli.py --relabel` uses, so the two passes cover the same
rows.

Output: data/golden/golden_v1_relabel.jsonl, in the schema label_cli writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundscore import config, llm, taxonomy  # noqa: E402
from groundscore.cleaning import stable_bucket  # noqa: E402

RELABEL_SUBSET = 50  # must match tools/label_cli.py

ESCALATION_REASONS = [
    "none",
    "needs_account_access",
    "angry_or_distressed",
    "policy_exception_or_refund",
    "safety_or_legal",
    "ambiguous_or_underspecified",
    "novel_no_precedent",
    "contains_personal_data",
    "multi_intent",
]

SYSTEM = (
    "You are a support-operations annotator building an evaluation set. You "
    "apply a written labelling guideline literally, including its boundary "
    "rules, and you do not soften a label because it seems harsh. You output "
    "JSON only."
)

PROMPT = """Label this customer message against the guideline below.

## Guideline

{guideline}

## Routing decision

auto      = a drafted reply could be sent publicly with no human reading it
escalate  = a human must handle this thread

Judge the MESSAGE, not any reply. The expensive mistake is a bad public reply
from a brand account: it is permanent, and the cases where it goes wrong are
the cases where the customer can least absorb it. A needless escalation costs
an agent thirty seconds. When genuinely torn, escalate and record
`ambiguous_or_underspecified`.

Intents marked "never auto-handled" in the guideline must always be escalate.

## Escalation reasons

{reasons}

Use "none" when the action is auto.

## Customer message

"{message}"

## Output

{{"intent": <one intent name>, "action": "auto" | "escalate",
  "escalation_reason": <one reason name>, "confidence": 1 | 2 | 3,
  "note": "<one sentence on the deciding factor>"}}

confidence is YOUR certainty in the label: 3 clear, 2 defensible, 1 a coin toss.

Return JSON only."""


def schema(intents: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": intents},
            "action": {"type": "string", "enum": ["auto", "escalate"]},
            "escalation_reason": {"type": "string", "enum": ESCALATION_REASONS},
            "confidence": {"type": "integer"},
            "note": {"type": "string"},
        },
        "required": ["intent", "action", "escalation_reason", "confidence", "note"],
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def subset(candidates: list[dict], n: int) -> list[dict]:
    """The same rows label_cli.py --relabel picks, by the same stable hash."""
    ordered = sorted(candidates, key=lambda c: stable_bucket(f"relabel:{c['thread_id']}", 10**9))
    return ordered[:n]


def annotate(item: dict, intents: list[str], guideline: str, model: str) -> dict:
    raw = llm.complete_json(
        PROMPT.format(guideline=guideline,
                      reasons="  ".join(ESCALATION_REASONS),
                      message=item["customer_msg"]),
        schema=schema(intents),
        model=model,
        system=SYSTEM,
        default={"intent": taxonomy.OTHER, "action": "escalate",
                 "escalation_reason": "ambiguous_or_underspecified",
                 "confidence": 1, "note": "annotator output unparseable"},
    )
    intent = raw.get("intent") if raw.get("intent") in intents else taxonomy.OTHER
    action = raw.get("action") if raw.get("action") in ("auto", "escalate") else "escalate"
    reason = (raw.get("escalation_reason") if raw.get("escalation_reason") in ESCALATION_REASONS
              else "ambiguous_or_underspecified")

    # The guideline's own rule, enforced here rather than trusted: a never-auto
    # class is never auto, whatever the annotator said.
    if intent in taxonomy.never_auto_names():
        action = "escalate"
    if action == "auto":
        reason = "none"
    elif reason == "none":
        reason = "ambiguous_or_underspecified"

    try:
        conf = int(raw.get("confidence", 2))
    except (TypeError, ValueError):
        conf = 2
    return {"intent": intent, "action": action, "escalation_reason": reason,
            "confidence": min(3, max(1, conf)), "note": str(raw.get("note", ""))[:300]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Blind second annotation of the golden subset")
    parser.add_argument("--model", default=llm.MODEL_JUDGE,
                        help="role token or model id; must not be the adjudicator's model")
    parser.add_argument("--n", type=int, default=RELABEL_SUBSET)
    args = parser.parse_args()

    candidates = load_jsonl(config.GOLDEN_CANDIDATES_PATH)
    if not candidates:
        print(f"No candidates at {config.GOLDEN_CANDIDATES_PATH}")
        return 1

    rows_in = subset(candidates, args.n)
    out_path = config.GOLDEN_DIR / "golden_v1_relabel.jsonl"
    done = {r["thread_id"]: r for r in load_jsonl(out_path)}

    intents = taxonomy.names()
    guideline = taxonomy.render_for_annotator()

    print(f"second annotator: {args.model}   subset {len(rows_in)}   "
          f"already done {len(done)}")

    for i, item in enumerate(rows_in, start=1):
        if item["thread_id"] in done:
            continue
        label = annotate(item, intents, guideline, args.model)
        done[item["thread_id"]] = {
            **{k: item[k] for k in ("thread_id", "customer_msg", "stratum", "split",
                                    "cluster", "adversarial_tags")},
            "relabel_intent": label["intent"],
            "relabel_action": label["action"],
            "relabel_escalation_reason": label["escalation_reason"],
            "relabel_annotator_confidence": label["confidence"],
            "relabel_note": label["note"],
            "relabel_model": args.model,
        }
        with out_path.open("w", encoding="utf-8") as fh:   # save after every item
            for r in done.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if i % 10 == 0:
            print(f"  {i}/{len(rows_in)}", flush=True)

    rows = list(done.values())
    print(f"\nsaved {len(rows)} -> {out_path}")
    print(f"intents  {dict(Counter(r['relabel_intent'] for r in rows))}")
    print(f"actions  {dict(Counter(r['relabel_action'] for r in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
