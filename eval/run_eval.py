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


# Baselines that LEARN from labels. They are fitted on dev, so scoring them on
# dev is in-sample and meaningless -- a TF-IDF nearest-neighbour classifier
# scores 1.00 against its own training set. Scored out-of-fold on dev instead;
# see `out_of_fold_outputs`. The LLM systems are not fitted on anything, so they
# are unaffected.
FITTED_BASELINES = ("trivial_always_auto", "trivial_always_escalate", "simple_tfidf_nn")
OOF_FOLDS = 5


def _fit_baseline(name: str, messages: list[str], labels: list[str], retriever) -> object:
    if name == "trivial_always_auto":
        return TrivialBaseline.from_labels(labels)
    if name == "trivial_always_escalate":
        return TrivialBaseline.from_labels(labels, always_escalate=True)
    if name == "simple_tfidf_nn":
        return SimpleBaseline(retriever).fit(messages, labels)
    raise KeyError(name)


def build_systems(dev_rows: list[dict], *, with_llm: bool) -> dict:
    """Fit every learned baseline on the dev split.

    Correct when the split being scored is `test`. When scoring `dev` itself,
    the caller must use `out_of_fold_outputs` for FITTED_BASELINES instead of
    these objects, or the baseline is graded on its own training data.
    """
    retriever = retrieve.load_retriever()
    dev_messages = [r["customer_msg"] for r in dev_rows]
    dev_labels = [r["intent"] for r in dev_rows]

    systems: dict[str, object] = {
        name: _fit_baseline(name, dev_messages, dev_labels, retriever)
        for name in FITTED_BASELINES
    }
    if with_llm:
        systems["agent_no_retrieval"] = SupportAgent(None, name="agent_no_retrieval")
        systems["agent"] = SupportAgent(retriever, name="agent")
    return systems


def out_of_fold_outputs(name: str, rows: list[dict], retriever, *, k: int = 5,
                        on_progress=None) -> list[dict]:
    """Cross-validated predictions for a fitted baseline on its own fit split.

    Each row is predicted by a copy of the baseline that never saw it. Folds are
    assigned by position over rows already ordered by a stable thread-id hash,
    so the partition is deterministic without importing another RNG.
    """
    outputs: list[dict] = []
    done = 0
    for fold in range(k):
        train = [r for i, r in enumerate(rows) if i % k != fold]
        held = [r for i, r in enumerate(rows) if i % k == fold]
        if not held:
            continue
        if not train:
            raise ValueError(f"fold {fold} has no training rows (k={k} too large)")
        system = _fit_baseline(
            name, [r["customer_msg"] for r in train], [r["intent"] for r in train], retriever)
        items = [(r["thread_id"], r["customer_msg"]) for r in held]
        outputs.extend(o.as_dict() for o in system.run(items))
        done += len(held)
        if on_progress:
            on_progress(done, len(rows))
    return outputs


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

    # The no-retrieval ablation has no exemplars, so the grounding and
    # similarity rules fire on every example and it escalates 100% by
    # construction. That is the correct safety behaviour, not a bug, but its
    # routing numbers are therefore degenerate and must not be read as a
    # comparison. Flagged here so the table cannot be misread.
    if all(by_id[r["thread_id"]]["action"] == "escalate" for r in rows):
        result["routing_note"] = (
            "escalates every example; routing metrics are degenerate by construction")
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
    parser.add_argument(
        "--judge-systems", default="",
        help="comma-separated systems to judge; default 'agent' on both splits. "
             "Name them all to compare reply quality across systems -- worth doing "
             "on dev, where the blind reference scores live. Judging every system "
             "on both splits is ~600 hosted calls against a 500/day free tier.")
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
    # Scoring the split the baselines were fitted on. Grading a learned model on
    # its own training data is not a baseline, it is a memorisation check, so
    # those systems are predicted out-of-fold instead.
    in_sample = args.split == "dev"
    print(f"scoring {len(rows)} rows on split={args.split} "
          f"(baselines fitted on {len(dev_rows)} dev rows)")
    if in_sample:
        print(f"  fitted baselines ({', '.join(FITTED_BASELINES)}) scored "
              f"out-of-fold, {OOF_FOLDS}-fold, because this IS their fit split")

    systems = build_systems(dev_rows, with_llm=not args.no_llm)
    items = [(r["thread_id"], r["customer_msg"]) for r in rows]

    all_results, all_outputs = {}, {}
    for name, system in systems.items():
        print(f"\n-- {name}")

        def progress(i: int, total: int, _name=name) -> None:
            if i % 20 == 0 or i == total:
                print(f"   {i}/{total}", flush=True)

        if in_sample and name in FITTED_BASELINES:
            outputs = out_of_fold_outputs(
                name, rows, retrieve.load_retriever(), k=OOF_FOLDS, on_progress=progress)
        else:
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

        # Every system's reply is judged against the SAME precedent set: the one
        # the retrieval-enabled agent saw for that message. Judging each system
        # against its own retrieved context would make groundedness
        # incomparable -- the no-retrieval ablation has no precedent at all, so
        # it would be scored against an empty evidence set and could not lose
        # marks for inventing policy. The canonical set makes "is this claim
        # supported by what the brand has actually said?" the same question for
        # every system.
        canonical = {o["thread_id"]: o.get("exemplars", [])
                     for o in all_outputs.get("agent", [])}

        # Judging defaults to the agent alone, on BOTH splits, and that is a
        # budget decision stated rather than hidden. A judged system costs one
        # hosted call per row; the free tier meters 500 per day per model, and
        # judging all four systems on both splits is ~600. Cross-system reply
        # quality is measured on DEV, where the blind reference scores live and
        # where all four systems are judged explicitly
        # (`--judge-systems a,b,c`). The test split exists to measure routing
        # and intent on held-out rows, neither of which needs a judge.
        #
        # This used to default to every system on test, which meant `make
        # eval-test` either blew the daily quota or fell back mid-run onto a
        # second judge model -- and a split scored half by one model and half by
        # another cannot carry a headline number.
        judge_systems = args.judge_systems.split(",") if args.judge_systems else ["agent"]

        for name, outputs in all_outputs.items():
            if name == "trivial_always_escalate":
                continue  # produces no replies to judge
            if name not in judge_systems:
                continue
            print(f"   {name}")
            if canonical:
                outputs = [{**o, "exemplars": canonical.get(o["thread_id"], o.get("exemplars", []))}
                           for o in outputs]
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
                    "providers": llm.provider_report(),
                    "cache": llm.cache_stats()}, indent=2), encoding="utf-8")
    table = markdown_table(all_results)
    (config.RESULTS_DIR / f"eval_{args.split}.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)

    if args.split == "test":
        record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(rows),
                  "systems": list(systems), "providers": llm.provider_report()}
        with (config.RESULTS_DIR / "test_runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print("\nlogged a test-split run to results/test_runs.jsonl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
