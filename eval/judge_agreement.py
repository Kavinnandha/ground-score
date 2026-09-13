"""Does the LLM judge agree with a human, and where does it not?

The assignment asks for evidence, so this produces four things:

  1. Agreement    Spearman rho per dimension against blind human scores, and
                  quadratic-weighted kappa on the would_send gate -- the gate
                  matters most because it is the judgement routing depends on.
  2. Bias         mean(judge - human) per dimension. A judge that correlates
                  well but sits a full point high still misreports absolute
                  quality, and the headline reply score would inherit that.
  3. Verbosity    the same replies re-judged with filler appended. Any score
                  probe        movement is length bias, since content is unchanged.
  4. Self-        the same replies re-judged by a DIFFERENT MODEL FAMILY
     preference   (Gemma vs Gemini). If the Gemini judge rates Gemini-written
                  replies systematically higher than the outside judge does,
                  that gap is self-preference and it inflates the headline.

Ordering is enforced: human scores must already exist on disk before this runs.
That is what makes them blind -- they cannot have been anchored to judge output
that had not been produced yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics  # noqa: E402
from eval.judge import DIMENSIONS, judge_reply  # noqa: E402
from groundscore import config, llm  # noqa: E402

FILLER = (" We really do appreciate you taking the time to reach out to us about "
          "this today, and we thank you for your patience while we look into it.")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def agreement_report(human: list[dict], judged: list[dict]) -> dict:
    by_id = {(j["thread_id"], j["system"]): j for j in judged}
    paired = [(h, by_id[(h["thread_id"], h["system"])]) for h in human
              if (h["thread_id"], h["system"]) in by_id]
    if not paired:
        raise RuntimeError("no overlap between human scores and judge scores")

    report = {"n_paired": len(paired), "dimensions": {}}
    for dim in DIMENSIONS:
        human_scores = [h[dim] for h, _ in paired]
        judge_scores = [j[dim] for _, j in paired]
        report["dimensions"][dim] = {
            **metrics.correlation(human_scores, judge_scores),
            "weighted_kappa": metrics.agreement(
                human_scores, judge_scores, weights="quadratic")["cohen_kappa"],
            "human_mean": round(sum(human_scores) / len(human_scores), 3),
            "judge_mean": round(sum(judge_scores) / len(judge_scores), 3),
        }

    human_gate = [bool(h["would_send"]) for h, _ in paired]
    judge_gate = [bool(j["would_send"]) for _, j in paired]
    report["would_send"] = {
        **metrics.agreement(human_gate, judge_gate),
        "human_send_rate": round(sum(human_gate) / len(human_gate), 3),
        "judge_send_rate": round(sum(judge_gate) / len(judge_gate), 3),
    }

    report["disagreements"] = [
        {
            "thread_id": h["thread_id"],
            "system": h["system"],
            "reply": j.get("reply", "")[:200],
            "human": {d: h[d] for d in DIMENSIONS} | {"would_send": h["would_send"]},
            "judge": {d: j[d] for d in DIMENSIONS} | {"would_send": j["would_send"]},
            "judge_justification": j.get("justification", "")[:200],
        }
        for h, j in paired if bool(h["would_send"]) != bool(j["would_send"])
    ][:15]
    return report


def verbosity_probe(judged: list[dict], outputs_by_id: dict, limit: int = 25) -> dict:
    """Re-judge padded replies. Content is unchanged, so any delta is length bias."""
    deltas = []
    for row in judged[:limit]:
        source = outputs_by_id.get(row["thread_id"])
        if not source or not row.get("reply"):
            continue
        padded = judge_reply(source["customer_msg"], row["reply"] + FILLER,
                             source.get("exemplars", []))
        deltas.append({
            "thread_id": row["thread_id"],
            "original_mean": row["mean_score"],
            "padded_mean": padded.mean_score,
            "delta": round(padded.mean_score - row["mean_score"], 3),
            "would_send_flipped": padded.would_send != row["would_send"],
        })
    if not deltas:
        return {"n": 0}
    mean_delta = sum(d["delta"] for d in deltas) / len(deltas)
    return {
        "n": len(deltas),
        "mean_delta": round(mean_delta, 3),
        "flips": sum(1 for d in deltas if d["would_send_flipped"]),
        "interpretation": (
            "positive mean_delta means the judge rewards length for identical content"),
        "examples": deltas[:10],
    }


def cross_family_probe(judged: list[dict], outputs_by_id: dict, limit: int = 25) -> dict:
    """Re-judge with an outside model family to estimate self-preference."""
    rows = []
    for row in judged[:limit]:
        source = outputs_by_id.get(row["thread_id"])
        if not source or not row.get("reply"):
            continue
        other = judge_reply(source["customer_msg"], row["reply"],
                            source.get("exemplars", []), model=llm.MODEL_JUDGE_CROSS)
        rows.append({
            "thread_id": row["thread_id"],
            "judge_mean": row["mean_score"],
            "cross_mean": other.mean_score,
            "delta": round(row["mean_score"] - other.mean_score, 3),
        })
    if not rows:
        return {"n": 0}
    judge_model = llm.SERVING.get("judge", llm.MODEL_JUDGE)
    cross_model = llm.SERVING.get("cross", llm.MODEL_JUDGE_CROSS)
    same_vendor = judge_model.split(":")[0] == cross_model.split(":")[0]
    return {
        "n": len(rows),
        "judge_model": judge_model,
        "cross_model": cross_model,
        # Whether the probe is a correction or a sanity check depends entirely
        # on this. Same vendor: the headline judge shares the drafter's lineage
        # and the delta is a discount to apply. Different vendors: the delta is
        # a check that no such discount is needed.
        "same_vendor_as_drafter": same_vendor,
        "mean_delta": round(sum(r["delta"] for r in rows) / len(rows), 3),
        "interpretation": (
            "positive mean_delta = the headline judge scores these replies higher than "
            "the drafter's own model does. When same_vendor_as_drafter is true, treat "
            "that gap as an upper bound on self-preference inflation in the headline "
            "reply score; when false, it is a cross-vendor sanity check instead"),
        "examples": rows[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--system", default="agent")
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()

    human = load_jsonl(config.HUMAN_JUDGE_PATH)
    if not human:
        print(f"No human scores at {config.HUMAN_JUDGE_PATH}.")
        print("Run: python tools/score_replies_cli.py   (do this BEFORE reading judge output)")
        return 1

    judged: list[dict] = []
    for path in sorted(config.RESULTS_DIR.glob(f"judge_{args.split}_*.jsonl")):
        judged.extend(load_jsonl(path))
    if not judged:
        print(f"No judge scores for split={args.split}. Run eval/run_eval.py first.")
        return 1

    report = agreement_report(human, judged)

    if not args.skip_probes:
        outputs = load_jsonl(config.RESULTS_DIR / f"outputs_{args.split}_{args.system}.jsonl")
        outputs_by_id = {o["thread_id"]: o for o in outputs}
        agent_judged = [j for j in judged if j["system"] == args.system]
        print("running verbosity probe...")
        report["verbosity_bias"] = verbosity_probe(agent_judged, outputs_by_id)
        print("running cross-family probe...")
        report["self_preference"] = cross_family_probe(agent_judged, outputs_by_id)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "judge_agreement.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\npaired: {report['n_paired']}")
    for dim, stats in report["dimensions"].items():
        print(f"  {dim:14s} rho={stats.get('spearman_rho', float('nan')):+.3f}  "
              f"kappa={stats['weighted_kappa']:+.3f}  "
              f"bias={stats['judge_mean'] - stats['human_mean']:+.2f}")
    gate = report["would_send"]
    print(f"  would_send     kappa={gate['cohen_kappa']:+.3f}  "
          f"raw={gate['raw_agreement']:.3f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
