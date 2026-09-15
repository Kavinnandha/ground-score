"""Sweep routing thresholds on the DEV split and freeze the result.

Writes configs/thresholds.yaml with `tuned: true` and the dev evidence that
produced the values. Running this against the test split is not supported --
the split is hard-coded, not a flag, so it cannot be done by accident.

The operating point is chosen as: the highest coverage whose false-auto rate
stays within a stated budget. The budget is a policy choice, not a discovered
optimum, and it is written into the config so a reader can see the number that
was chosen rather than inferring it from the results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics  # noqa: E402
from eval.run_eval import load_golden  # noqa: E402
from groundscore import config, retrieve  # noqa: E402
from groundscore.agent import SupportAgent  # noqa: E402

DEFAULT_BUDGET = 0.10  # tolerated share of auto-sent replies that should have escalated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-false-auto", type=float, default=DEFAULT_BUDGET)
    args = parser.parse_args()

    rows = load_golden("dev")
    if not rows:
        print("No dev rows labelled yet.")
        return 1
    print(f"tuning on {len(rows)} dev examples, false-auto budget {args.max_false_auto:.0%}")

    # Run the agent once with the LLM router disabled: the sweep is over the
    # deterministic thresholds, and letting the LLM re-decide each point would
    # make the curve non-monotonic for reasons unrelated to the threshold.
    agent = SupportAgent(retrieve.load_retriever(), use_llm_router=False, name="agent_tuning")
    outputs = agent.run([(r["thread_id"], r["customer_msg"]) for r in rows],
                        on_progress=lambda i, t: print(f"  {i}/{t}", flush=True)
                        if i % 20 == 0 else None)

    by_id = {o.thread_id: o for o in outputs}
    gold = [r["action"] for r in rows]
    confidences = [by_id[r["thread_id"]].confidence for r in rows]
    similarities = [by_id[r["thread_id"]].max_similarity for r in rows]
    forced = [bool(by_id[r["thread_id"]].diagnostics.get("forced_escalation")) for r in rows]

    conf_curve = metrics.coverage_curve(gold, confidences, forced)
    sim_curve = metrics.coverage_curve(gold, similarities, forced)

    best_conf = metrics.best_threshold(conf_curve, args.max_false_auto)
    best_sim = metrics.best_threshold(sim_curve, args.max_false_auto)

    if best_conf is None:
        print("No confidence threshold meets the budget -- keeping the conservative default.")
    tau_conf = best_conf["threshold"] if best_conf else 0.9
    tau_sim = best_sim["threshold"] if best_sim else 0.75

    payload = {
        "tuned": True,
        "tuned_on": "dev",
        "n_dev": len(rows),
        "max_false_auto_budget": args.max_false_auto,
        "tau_confidence": float(tau_conf),
        "tau_similarity": float(tau_sim),
        "dev_operating_point": {
            "confidence": best_conf,
            "similarity": best_sim,
        },
        "_note": (
            "Chosen on the dev split only. With ~70 dev examples these values are "
            "themselves noisy estimates; the report treats them as a fitted "
            "parameter, not a discovered constant."
        ),
    }
    config.THRESHOLDS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config.THRESHOLDS_CONFIG.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "threshold_sweep.json").write_text(
        json.dumps({"confidence_curve": conf_curve, "similarity_curve": sim_curve,
                    "chosen": payload}, indent=2), encoding="utf-8")

    print(f"\ntau_confidence = {tau_conf}   tau_similarity = {tau_sim}")
    print(f"wrote {config.THRESHOLDS_CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
