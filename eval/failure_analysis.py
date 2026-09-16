"""Rank the agent's failure modes by frequency, with real examples attached.

The report has to name its top failure modes. Picking them by hand invites
picking the interesting ones over the common ones, so they are counted here and
the report quotes this output.

Three families are counted separately because they have different costs and
different fixes:

  routing   auto-sent something the golden label says a human should handle
            (false-auto: the expensive direction, public and permanent), and
            escalated something that could have gone out (false-escalate: costs
            an agent thirty seconds)
  intent    predicted the wrong class. Reported as a confusion list rather than
            a single accuracy, because the classes are not equally consequential
            -- a miss INTO a never-auto class is harmless, a miss OUT of one
            removes the only rule protecting that message
  drafting  the reply itself: unfilled placeholders, personal data solicited on
            a public timeline, names carried over from a precedent thread

The last family is found with patterns rather than by the judge, deliberately:
`results/judge_agreement.json` shows the judge scoring several of these replies
as sendable, so a judge-derived failure list would not contain them.

Output: results/failure_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundscore import config, taxonomy  # noqa: E402

# Draft-level defects, matched on the reply text. Each one is a thing that
# should never appear in a public brand tweet, and none of them needs a model
# to detect.
DRAFT_DEFECTS: list[tuple[str, str, str]] = [
    # `[Customer Name]` was in a live dev reply and the first version of this
    # pattern missed it, so brackets are matched generically rather than by
    # enumerating the placeholders already seen. `<URL>` is excluded because it
    # is the corpus's own redaction token and appears in real brand replies.
    ("unfilled_placeholder",
     r"\[(?!URL\])[A-Za-z][A-Za-z _-]{2,30}\]|\{[a-z_]{2,}\}|<insert\b|\bTODO\b|\bXX+\b",
     "shipped a template placeholder instead of a value"),
    ("solicits_pii_publicly",
     r"(share|provide|send|reply (?:to this tweet )?with|confirm)[^.?!]{0,40}"
     r"\b(order (?:id|number|confirmation)|payment method|card details|"
     r"email address|account email)\b",
     "asks the customer to post personal or order data on a public timeline, "
     "which is the thing the brand's own replies warn against"),
    ("unsupported_time_commitment",
     r"\b(within|in) (?:the next )?\d+ (?:hours|hrs|days|minutes)\b|"
     r"\bget back to you\b|\bwill (?:contact|update|call) you\b",
     "promises a follow-up or a deadline that no precedent supports"),
    ("internal_routing_leak",
     r"\bhandled by a human agent\b|\ba human will\b|\bescalat\w+ internally\b",
     "tells the customer about the automation's own internal routing"),
]
_DRAFT_RE = [(name, re.compile(pat, re.I), why) for name, pat, why in DRAFT_DEFECTS]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def carried_over_name(reply: str, exemplars: list[dict]) -> str | None:
    """A first name in the reply that came from a precedent, not this customer.

    The nearest-neighbour baseline copies a precedent verbatim, so it greets
    this customer by the name of a different one. The LLM drafter does it too
    when it leans hard on a single exemplar. Public, and instantly recognisable
    to the person it is addressed to as a machine mistake.
    """
    m = re.match(r"^(?:hi|hey|hello|dear|sorry(?:,| to hear that,)?|"
                 r"thanks(?:,)?)[, ]+([A-Z][a-z]{2,})\b", reply.strip(), re.I)
    if not m:
        m = re.search(r"\b(?:sorry|apologies|thanks)[^.?!]{0,30}, ([A-Z][a-z]{2,})[!.,]", reply)
    if not m:
        return None
    name = m.group(1)
    if name.lower() in {"there", "all", "guys", "team", "again", "about", "for", "to"}:
        return None
    for ex in exemplars or []:
        if re.search(rf"\b{re.escape(name)}\b", ex.get("brand_reply", "")):
            return name
    return None


def analyse(split: str, system: str) -> dict:
    gold = {r["thread_id"]: r for r in load_jsonl(config.GOLDEN_PATH) if r["split"] == split}
    outputs = load_jsonl(config.RESULTS_DIR / f"outputs_{split}_{system}.jsonl")
    if not gold or not outputs:
        raise SystemExit(f"need golden labels and outputs_{split}_{system}.jsonl")

    never_auto = taxonomy.never_auto_names()
    false_auto, false_escalate, intent_errors, draft_defects = [], [], [], []

    for o in outputs:
        g = gold.get(o["thread_id"])
        if not g:
            continue
        row = {"thread_id": o["thread_id"], "stratum": g["stratum"],
               "customer_msg": o["customer_msg"][:180],
               "gold_intent": g["intent"], "pred_intent": o.get("intent"),
               "confidence": o.get("confidence"), "max_similarity": o.get("max_similarity"),
               "reply": (o.get("reply") or "")[:200]}

        if o["action"] == "auto" and g["action"] == "escalate":
            false_auto.append({**row, "gold_escalation_reason": g["escalation_reason"],
                               "annotator_confidence": g["annotator_confidence"],
                               "intent_was_wrong": o.get("intent") != g["intent"],
                               "gold_class_is_never_auto": g["intent"] in never_auto})
        elif o["action"] == "escalate" and g["action"] == "auto":
            false_escalate.append({**row, "triggered_rule": o.get("triggered_rule")})

        if o.get("intent") != g["intent"]:
            intent_errors.append({**row,
                                  "cost": ("lost a never-auto guard"
                                           if g["intent"] in never_auto
                                           and o.get("intent") not in never_auto
                                           else "harmless direction")})

        reply = o.get("reply") or ""
        hits = [(name, why) for name, rx, why in _DRAFT_RE if rx.search(reply)]
        name = carried_over_name(reply, o.get("exemplars", []))
        if name:
            hits.append(("name_carried_from_precedent",
                         f"greets this customer as '{name}', a name that appears only in "
                         f"the precedent thread"))
        for defect, why in hits:
            draft_defects.append({**row, "defect": defect, "why": why})

    n_auto = sum(1 for o in outputs if o["action"] == "auto")
    return {
        "split": split,
        "system": system,
        "n": len(outputs),
        "n_auto": n_auto,
        "false_auto": {
            "n": len(false_auto),
            "share_of_auto_sent": round(len(false_auto) / n_auto, 3) if n_auto else 0.0,
            "by_gold_reason": dict(Counter(r["gold_escalation_reason"] for r in false_auto).most_common()),
            "intent_was_also_wrong": sum(r["intent_was_wrong"] for r in false_auto),
            "gold_class_was_never_auto": sum(r["gold_class_is_never_auto"] for r in false_auto),
            "confidence_range": ([min(r["confidence"] for r in false_auto),
                                  max(r["confidence"] for r in false_auto)] if false_auto else []),
            "rows": false_auto,
        },
        "false_escalate": {
            "n": len(false_escalate),
            "by_rule": dict(Counter(r["triggered_rule"] for r in false_escalate).most_common()),
            "rows": false_escalate,
        },
        "intent_errors": {
            "n": len(intent_errors),
            "confusions": {f"{k[0]} -> {k[1]}": v for k, v in
                           Counter((r["gold_intent"], r["pred_intent"])
                                   for r in intent_errors).most_common()},
            "lost_a_never_auto_guard": sum(1 for r in intent_errors
                                           if r["cost"] == "lost a never-auto guard"),
            "rows": intent_errors,
        },
        "draft_defects": {
            "n": len(draft_defects),
            "by_type": dict(Counter(r["defect"] for r in draft_defects).most_common()),
            "rows": draft_defects,
        },
        "escalation_rules_fired": dict(Counter(o.get("triggered_rule") for o in outputs).most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--system", default="agent")
    parser.add_argument("--all-systems", action="store_true",
                        help="also count draft defects for the baselines, for comparison")
    args = parser.parse_args()

    report = {"primary": analyse(args.split, args.system)}
    if args.all_systems:
        report["draft_defects_by_system"] = {}
        for path in sorted(config.RESULTS_DIR.glob(f"outputs_{args.split}_*.jsonl")):
            name = path.stem.replace(f"outputs_{args.split}_", "")
            if name == "trivial_always_escalate":
                continue
            sub = analyse(args.split, name)["draft_defects"]
            report["draft_defects_by_system"][name] = {"n": sub["n"], "by_type": sub["by_type"]}

    out = config.RESULTS_DIR / f"failure_analysis_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    p = report["primary"]
    print(f"{p['system']} on {p['split']}: {p['n']} rows, {p['n_auto']} auto-sent")
    print(f"  false-auto       {p['false_auto']['n']:3d}  "
          f"({p['false_auto']['share_of_auto_sent']:.1%} of what we sent)")
    for reason, n in p["false_auto"]["by_gold_reason"].items():
        print(f"      {reason:30s}{n}")
    print(f"      intent also wrong on {p['false_auto']['intent_was_also_wrong']}; "
          f"gold class was never-auto on {p['false_auto']['gold_class_was_never_auto']}")
    print(f"      confidence range on these rows: {p['false_auto']['confidence_range']}")
    print(f"  false-escalate   {p['false_escalate']['n']:3d}")
    for rule, n in p["false_escalate"]["by_rule"].items():
        print(f"      {str(rule):30s}{n}")
    print(f"  intent errors    {p['intent_errors']['n']:3d}  "
          f"({p['intent_errors']['lost_a_never_auto_guard']} removed a never-auto guard)")
    for conf, n in list(p["intent_errors"]["confusions"].items())[:8]:
        print(f"      {conf:52s}{n}")
    print(f"  draft defects    {p['draft_defects']['n']:3d}")
    for defect, n in p["draft_defects"]["by_type"].items():
        print(f"      {defect:30s}{n}")
    if "draft_defects_by_system" in report:
        print("  draft defects by system:")
        for name, sub in report["draft_defects_by_system"].items():
            print(f"      {name:24s} {sub['n']:3d}  {sub['by_type']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
