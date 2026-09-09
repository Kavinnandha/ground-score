"""Run every system over a golden split and write the results tables.

Test-split discipline
---------------------
`--split test` refuses to run without `--final`, and every final run appends a
timestamped record to results/test_runs.jsonl. The point is not that the guard
is unbypassable -- it obviously is, it is one flag -- but that bypassing it
leaves a trace. A reader can check how many times the test set was scored, and
"tuned on dev, scored on test once" becomes a claim with evidence behind it
instead of an assurance.

Systems compared
----------------
  trivial_always_auto      majority intent, canned reply, never escalates
  trivial_always_escalate  never sends anything
  simple_tfidf_nn          TF-IDF classifier + verbatim nearest historical reply
  agent_no_retrieval       full LLM pipeline, retrieval disabled (ablation)
  agent                    the system

The ablation is what separates "the LLM is good" from "the grounding is good".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics  # noqa: E402
from eval.judge import judge_outputs, write_scores  # noqa: E402
from groundscore import config, llm, retrieve  # noqa: E402
from groundscore.agent import SupportAgent, write_outputs  # noqa: E402
from groundscore.baselines.simple import SimpleBaseline  # noqa: E402
from groundscore.baselines.trivial import TrivialBaseline  # noqa: E402


def load_golden(split: str | None = None) -> list[dict]:
    if not config.GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{config.GOLDEN_PATH} missing. Run scripts/sample_golden.py then tools/label_cli.py"
        )
    with config.GOLDEN_PATH.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = [r for r in rows if r.get("intent") and r.get("action")]
    if split:
        rows = [r for r in rows if r.get("split") == split]
    return rows


def build_systems(dev_rows: list[dict], *, with_llm: bool) -> dict:
    """Baselines are fitted on DEV only -- never on the split being scored."""
    retriever = retrieve.load_retriever()
    dev_messages = [r["customer_msg"] for r in dev_rows]
    dev_labels = [r["intent"] for r in dev_rows]

    systems: dict[str, object] = {
        "trivial_always_auto": TrivialBaseline.from_labels(dev_labels),
        "trivial_always_escalate": TrivialBaseline.from_labels(dev_labels, always_escalate=True),
        "simple_tfidf_nn": SimpleBaseline(retriever).fit(dev_messages, dev_labels),
    }
    if with_llm:
        systems["agent_no_retrieval"] = SupportAgent(None, name="agent_no_retrieval")
        systems["agent"] = SupportAgent(retriever, name="agent")
    return systems


def score_system(rows: list[dict], outputs: list[dict]) -> dict:
    by_id = {o["thread_id"]: o for o in outputs}
    gold_intent = [r["intent"] for r in rows]
    pred_intent = [by_id[r["thread_id"]]["intent"] for r in rows]
    gold_action = [r["action"] for r in rows]
    pred_action = [by_id[r["thread_id"]]["action"] for r in rows]

    result = {
        "n": len(rows),
        "intent": metrics.score_intents(gold_intent, pred_intent).as_dict(),
        "routing": metrics.score_routing(gold_action, pred_action).as_dict(),
        "escalation_rules": dict(Counter(
            by_id[r["thread_id"]]["triggered_rule"] for r in rows
            if by_id[r["thread_id"]]["action"] == "escalate")),
    }

    # Per-stratum: the pooled number is not the traffic number, because the
    # golden set deliberately over-samples rare and adversarial cases.
    per_stratum = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        subset = [r for r in rows if r["stratum"] == stratum]
        if len(subset) < 3:
            continue
        per_stratum[stratum] = {
            "n": len(subset),
            "intent_accuracy": metrics.score_intents(
                [r["intent"] for r in subset],
                [by_id[r["thread_id"]]["intent"] for r in subset],
            ).accuracy,
            "routing": metrics.score_routing(
                [r["action"] for r in subset],
                [by_id[r["thread_id"]]["action"] for r in subset],
            ).as_dict(),
        }
    result["per_stratum"] = per_stratum
    return result


def markdown_table(all_results: dict) -> str:
    header = ("| system | intent acc | macro F1 | coverage | false-auto | esc. recall |\n"
              "|---|---|---|---|---|---|")
    lines = [header]
    for name, res in all_results.items():
        acc, f1 = res["intent"]["accuracy"], res["intent"]["macro_f1"]
        cov, fa = res["routing"]["coverage"], res["routing"]["false_auto_rate"]
        lines.append(
            f"| {name} "
            f"| {acc['value']:.3f} [{acc['ci95'][0]:.2f}–{acc['ci95'][1]:.2f}] "
            f"| {f1['value']:.3f} [{f1['ci95'][0]:.2f}–{f1['ci95'][1]:.2f}] "
            f"| {cov['value']:.3f} "
            f"| {fa['value']:.3f} [{fa['ci95'][0]:.2f}–{fa['ci95'][1]:.2f}] "
            f"| {res['routing']['escalation_recall']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true",
                        help="required to score the test split; logs a timestamped record")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    parser.add_argument("--no-llm", action="store_true", help="baselines only (no API calls)")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        print(
            "Refusing to score the test split without --final.\n"
            "The test split exists to be scored ONCE, after thresholds and prompts are\n"
            "frozen on dev. If you are iterating, use --split dev."
        )
        return 2

    dev_rows = load_golden("dev")
    rows = load_golden(args.split)
    if not rows:
        print(f"No labelled rows in split '{args.split}'.")
        return 1
    print(f"scoring {len(rows)} rows on split={args.split} "
          f"(baselines fitted on {len(dev_rows)} dev rows)")

    systems = build_systems(dev_rows, with_llm=not args.no_llm)
    items = [(r["thread_id"], r["customer_msg"]) for r in rows]

    all_results, all_outputs = {}, {}
    for name, system in systems.items():
        print(f"\n-- {name}")

        def progress(i: int, total: int, _name=name) -> None:
            if i % 20 == 0 or i == total:
                print(f"   {i}/{total}", flush=True)

        outputs = [o.as_dict() for o in system.run(items, on_progress=progress)]
        write_outputs_path = config.RESULTS_DIR / f"outputs_{args.split}_{name}.jsonl"
        write_outputs_path.parent.mkdir(parents=True, exist_ok=True)
        with write_outputs_path.open("w", encoding="utf-8") as fh:
            for o in outputs:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")
        all_outputs[name] = outputs
        all_results[name] = score_system(rows, outputs)

    if not args.no_judge and not args.no_llm:
        print("\n-- judging replies")
        for name, outputs in all_outputs.items():
            if name == "trivial_always_escalate":
                continue  # produces no replies to judge
            print(f"   {name}")
            scores = judge_outputs(outputs, on_progress=lambda i, t: (
                print(f"     {i}/{t}", flush=True) if i % 20 == 0 or i == t else None))
            write_scores(scores, config.RESULTS_DIR / f"judge_{args.split}_{name}.jsonl")
            sendable = sum(1 for s in scores if s["would_send"])
            all_results[name]["judge"] = {
                "mean_score": round(sum(s["mean_score"] for s in scores) / max(len(scores), 1), 3),
                "would_send_rate": round(sendable / max(len(scores), 1), 3),
                **{d: round(sum(s[d] for s in scores) / max(len(scores), 1), 3)
                   for d in ("groundedness", "resolution", "tone_fit", "safety")},
            }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"eval_{args.split}.json").write_text(
        json.dumps({"split": args.split, "results": all_results,
                    "cache": llm.cache_stats()}, indent=2), encoding="utf-8")
    table = markdown_table(all_results)
    (config.RESULTS_DIR / f"eval_{args.split}.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)

    if args.split == "test":
        record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(rows),
                  "systems": list(systems)}
        with (config.RESULTS_DIR / "test_runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print("\nlogged a test-split run to results/test_runs.jsonl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
