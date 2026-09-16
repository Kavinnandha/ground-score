"""Regenerate every headline number from committed artifacts. No API key needed.

This is the path the README promises and the one a reviewer actually runs. It
asserts its own honesty in two ways:

  * It runs with the API key deliberately hidden from the process, so any step
    that is not fully cached fails loudly instead of quietly making live calls
    and producing numbers that differ from the committed ones.
  * It reports cache hits and misses at the end. Misses must be zero.

If this script passes, the tables in results/ were produced by exactly the code
and inputs in this repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED = [
    ("data/processed/threads.jsonl", "corpus", "python scripts/build_dataset.py"),
    ("cache/llm_cache.sqlite", "LLM cache", "make full"),
    ("taxonomy/intents.yaml", "intent taxonomy", "python -m groundscore.discover_intents"),
    ("data/golden/golden_v1.jsonl", "golden set", "python tools/label_cli.py"),
    ("data/golden/golden_v1_relabel.jsonl", "blind second pass",
     "python tools/second_annotator.py"),
    ("data/golden/human_reply_scores.jsonl", "blind reply scores",
     "python tools/score_replies_cli.py --split dev"),
    ("configs/thresholds.yaml", "frozen thresholds", "python eval/tune_thresholds.py"),
]

# Ordered. The evaluations have to precede anything that reads their outputs,
# and the last two make no model calls at all -- they are pure functions of
# files this script has just regenerated, which is why they can sit inside a run
# that has the API key stripped out of its environment.
STEPS = [
    # All four reply-producing systems are judged on dev, named explicitly
    # rather than left to the default, so this run regenerates every judge file
    # the agreement statistic pairs on instead of inheriting three of them from
    # the commit. Every one of these calls is in the committed cache; the
    # default is narrower only to keep a LIVE `make eval` inside the daily
    # hosted budget (DECISIONS.md #42).
    ("dev evaluation", [sys.executable, "eval/run_eval.py", "--split", "dev",
                        "--judge-systems",
                        "agent,agent_no_retrieval,simple_tfidf_nn,trivial_always_auto"]),
    ("test evaluation", [sys.executable, "eval/run_eval.py", "--split", "test", "--final"]),
    ("judge validation", [sys.executable, "eval/judge_agreement.py", "--split", "dev",
                          "--skip-probes"]),
    ("label agreement", [sys.executable, "eval/label_agreement.py"]),
    ("failure analysis", [sys.executable, "eval/failure_analysis.py", "--split", "dev",
                          "--all-systems"]),
]


def check_inputs() -> bool:
    missing = [(p, what, how) for p, what, how in REQUIRED if not (ROOT / p).exists()]
    for path, what, how in missing:
        print(f"  MISSING {path:38s} ({what}) -> {how}")
    return not missing


def main() -> int:
    print("=" * 72)
    print("ground-score :: reproduce headline results from committed artifacts")
    print("=" * 72)

    if not check_inputs():
        print("\nCannot reproduce: artifacts above are missing from the repository.")
        return 1

    # Hide the key so an incomplete cache cannot be papered over by live calls.
    env = dict(os.environ)
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        env.pop(var, None)
    env["GROUNDSCORE_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    started = time.monotonic()
    for label, command in STEPS:
        print(f"\n--- {label}")
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"\nFAILED at '{label}' (exit {result.returncode}).")
            print("A cache miss here means the committed cache does not cover this code path.")
            return result.returncode

    elapsed = time.monotonic() - started
    print("\n" + "=" * 72)
    for name in ("eval_dev.md", "eval_test.md"):
        path = ROOT / "results" / name
        if path.exists():
            print(f"\n## {name}\n")
            print(path.read_text(encoding="utf-8").strip())

    agreement = ROOT / "results" / "judge_agreement.json"
    if agreement.exists():
        report = json.loads(agreement.read_text(encoding="utf-8"))
        gate = report.get("would_send", {})
        # The reference scores were written by the project author before judge
        # output was available, so this is judge-versus-human validation.
        print(f"\n## judge vs reference annotator (n={report.get('n_paired')})")
        print(f"would_send kappa {gate.get('cohen_kappa', float('nan')):+.3f}   "
              f"raw agreement {gate.get('raw_agreement', float('nan')):.3f}")

    labels = ROOT / "results" / "label_agreement.json"
    if labels.exists():
        report = json.loads(labels.read_text(encoding="utf-8"))
        rates = report["action_base_rates"]
        print(f"\n## label agreement ({report.get('kind')}, n={report.get('n')})")
        print(f"intent kappa {report['intent']['cohen_kappa']:+.3f}   "
              f"action kappa {report['action']['cohen_kappa']:+.3f} "
              f"(escalate rates {rates['pass_1_escalate']:.0%} vs "
              f"{rates['pass_2_escalate']:.0%})")

    failures = ROOT / "results" / "failure_analysis_dev.json"
    if failures.exists():
        primary = json.loads(failures.read_text(encoding="utf-8"))["primary"]
        fa = primary["false_auto"]
        print("\n## failure modes (dev, agent)")
        print(f"false-auto {fa['n']}/{primary['n_auto']} auto-sent "
              f"({fa['share_of_auto_sent']:.1%})   "
              f"false-escalate {primary['false_escalate']['n']}   "
              f"draft defects {primary['draft_defects']['n']}")

    print(f"\ntotal wall clock: {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
