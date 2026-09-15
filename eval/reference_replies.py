"""Score the brand's OWN historical reply with the same rubric as the agent.

Why
---
Every evaluation in this repo treats the brand's historical reply as evidence:
retrieval grounds the drafter in it, and the golden set's weak labels descend
from it. Nothing anywhere asks the obvious next question -- how good is that
reply? "Agent scores 3.4/5" means very little on its own. "Agent scores 3.4
where the human agent who actually answered this ticket scored 3.6" means
something immediately, and it is the only number in this project that a support
lead could act on without reading the rubric first.

It costs one extra judge call per example and needs no labels at all, which is
why it runs before the golden set is adjudicated.

The comparison is deliberately paired: same customer message, same retrieved
precedent, same rubric, same blind judge, two replies. The judge is never told
which is which -- `judge_reply` takes (message, reply, exemplars) and nothing
else -- and identical text would hash to the same cache entry either way.

What this is NOT
----------------
It is not a fair fight, and reading it as one is the mistake to avoid.

  * The human had the account, the order and the tracking page open. The agent
    had five past tweets. On any question that needs account data the human can
    resolve and the agent structurally cannot.
  * Groundedness is therefore not comparable between the two. The rubric scores
    a claim as unsupported when no retrieved precedent backs it, and the human's
    claims are backed by systems the judge cannot see. A low groundedness score
    on a human reply measures the rubric's blind spot, not the reply. It is
    reported separately for exactly that reason and excluded from the headline
    delta.
  * Resolution, tone fit, safety and the would_send gate ARE comparable: they
    ask whether this text, sent to this customer, helps and is safe.

What it buys
------------
Three things that are otherwise assertions in the report:

  1. A human reference point for reply quality, on the same rows.
  2. A measured answer to "the rubric penalises deflection even though the
     brand deflects" -- the historical replies that hand off are scored here
     against the ones that do not, on real data rather than by argument.
  3. The share of the brand's own public replies that this system's would_send
     gate would refuse to send. If that share is high, the gate is stricter than
     the brand itself, and every coverage number in the report has to be read
     against that.

    python eval/reference_replies.py --split dev
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.judge import DIMENSIONS, judge_reply  # noqa: E402
from groundscore import config, llm, retrieve  # noqa: E402
from groundscore.agent import SupportAgent  # noqa: E402
from groundscore.cleaning import is_handoff  # noqa: E402

# Groundedness is excluded: the human's facts come from systems the judge
# cannot see, so scoring them against retrieved precedent measures the rubric,
# not the reply. See the module docstring.
COMPARABLE = ("resolution", "tone_fit", "safety")


def load_candidates(split: str | None) -> list[dict]:
    path = config.GOLDEN_CANDIDATES_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/sample_golden.py")
    with path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if split:
        rows = [r for r in rows if r.get("split") == split]
    return [r for r in rows if r.get("brand_replies")]


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 3) if values else float("nan")


def compare(rows: list[dict], *, limit: int | None = None) -> dict:
    rows = rows[:limit] if limit else rows
    retriever = retrieve.load_retriever()
    agent = SupportAgent(retriever, name="agent")

    paired: list[dict] = []
    for i, row in enumerate(rows, start=1):
        message = row["customer_msg"]
        human_reply = row["brand_replies"][0]

        out = agent.handle(row["thread_id"], message)
        # Both replies are judged against the SAME precedent -- the set the
        # agent was given. Judging the human against some other evidence set
        # would make the two scores incomparable on every dimension, not just
        # groundedness.
        exemplars = out.exemplars

        agent_score = judge_reply(message, out.reply, exemplars)
        human_score = judge_reply(message, human_reply, exemplars)

        paired.append({
            "thread_id": row["thread_id"],
            "stratum": row["stratum"],
            "customer_msg": message,
            "agent_reply": out.reply,
            "human_reply": human_reply,
            "agent_action": out.action,
            "agent_triggered_rule": out.triggered_rule,
            "human_reply_is_handoff": is_handoff(human_reply),
            "agent": agent_score.as_dict(),
            "human": human_score.as_dict(),
            "judge_model": llm.SERVING.get("judge", llm.MODEL_JUDGE),
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)

    return summarise(paired)


def summarise(paired: list[dict]) -> dict:
    if not paired:
        return {"n": 0}

    dims: dict[str, dict] = {}
    for dim in DIMENSIONS:
        agent_vals = [p["agent"][dim] for p in paired]
        human_vals = [p["human"][dim] for p in paired]
        dims[dim] = {
            "agent_mean": mean(agent_vals),
            "human_mean": mean(human_vals),
            "delta_agent_minus_human": round(mean(agent_vals) - mean(human_vals), 3),
            "comparable": dim in COMPARABLE,
        }

    agent_gate = [bool(p["agent"]["would_send"]) for p in paired]
    human_gate = [bool(p["human"]["would_send"]) for p in paired]

    # Head-to-head on the comparable dimensions only.
    def comparable_mean(side: str, row: dict) -> float:
        return statistics.fmean(row[side][d] for d in COMPARABLE)

    wins = sum(1 for p in paired if comparable_mean("agent", p) > comparable_mean("human", p))
    losses = sum(1 for p in paired if comparable_mean("agent", p) < comparable_mean("human", p))

    handoffs = [p for p in paired if p["human_reply_is_handoff"]]
    direct = [p for p in paired if not p["human_reply_is_handoff"]]

    return {
        "n": len(paired),
        "judge_model": paired[0]["judge_model"],
        "dimensions": dims,
        "comparable_dimensions": list(COMPARABLE),
        "comparable_mean": {
            "agent": mean([comparable_mean("agent", p) for p in paired]),
            "human": mean([comparable_mean("human", p) for p in paired]),
        },
        "head_to_head": {
            "agent_better": wins,
            "human_better": losses,
            "tie": len(paired) - wins - losses,
        },
        "would_send_rate": {
            "agent": round(sum(agent_gate) / len(paired), 3),
            "human_historical": round(sum(human_gate) / len(paired), 3),
        },
        # The number that makes the point: replies this brand actually posted
        # in public, which this system's own gate would have held back.
        "human_replies_our_gate_would_block": {
            "n": sum(1 for g in human_gate if not g),
            "share": round(sum(1 for g in human_gate if not g) / len(paired), 3),
        },
        "historical_handoffs": {
            "n": len(handoffs),
            "share": round(len(handoffs) / len(paired), 3),
            "resolution_mean_when_handoff": mean([p["human"]["resolution"] for p in handoffs]),
            "resolution_mean_when_direct": mean([p["human"]["resolution"] for p in direct]),
            "note": ("the rubric scores a handoff as a resolution failure by design; "
                     "this is that rule measured against the brand's real replies "
                     "instead of argued for"),
        },
        "caveats": [
            "The human had account, order and tracking access. The agent had five "
            "past tweets. This is a reference point, not a fair fight.",
            "Groundedness is excluded from the headline comparison: the human's claims "
            "rest on systems the judge cannot see, so the rubric scores them as "
            "unsupported. That measures the rubric's blind spot, not the reply.",
            "Same small judge as everywhere else in this repo. Its agreement with a "
            "human is reported in results/judge_agreement.json and bounds what this "
            "table can claim.",
        ],
        "rows": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge the brand's own historical replies against the agent's")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N rows (smoke runs)")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    rows = load_candidates(split)
    if not rows:
        print(f"No candidates in split {args.split!r}.")
        return 1
    print(f"pairing {len(rows)} rows on split={args.split} "
          f"({2 * len(rows)} judge calls)")

    report = compare(rows, limit=args.limit)
    rows_out = report.pop("rows")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"reference_replies_{args.split}"
    (config.RESULTS_DIR / f"{stem}.json").write_text(
        json.dumps({**report, "providers": llm.provider_report()}, indent=2),
        encoding="utf-8")
    with (config.RESULTS_DIR / f"{stem}.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nn = {report['n']}   judge = {report['judge_model']}")
    print(f"comparable mean ({'+'.join(COMPARABLE)}):")
    print(f"  agent {report['comparable_mean']['agent']:.2f}   "
          f"human {report['comparable_mean']['human']:.2f}")
    h2h = report["head_to_head"]
    print(f"head to head   agent {h2h['agent_better']}  "
          f"human {h2h['human_better']}  tie {h2h['tie']}")
    gate = report["would_send_rate"]
    print(f"would_send     agent {gate['agent']:.2f}   "
          f"human {gate['human_historical']:.2f}")
    blocked = report["human_replies_our_gate_would_block"]
    print(f"our gate would have blocked {blocked['n']} of {report['n']} replies "
          f"this brand actually posted ({blocked['share']:.0%})")
    print(f"\nwrote results/{stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
