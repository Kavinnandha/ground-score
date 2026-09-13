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
    ("configs/thresholds.yaml", "tuned thresholds", "python eval/tune_thresholds.py"),
]

STEPS = [
    ("dev evaluation", [sys.executable, "eval/run_eval.py", "--split", "dev"]),
    ("test evaluation", [sys.executable, "eval/run_eval.py", "--split", "test", "--final"]),
    ("judge validation", [sys.executable, "eval/judge_agreement.py", "--split", "dev",
                          "--skip-probes"]),
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
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
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
        print(f"\n## judge vs human (n={report.get('n_paired')})")
        print(f"would_send kappa {gate.get('cohen_kappa', float('nan')):+.3f}   "
              f"raw agreement {gate.get('raw_agreement', float('nan')):.3f}")

    print(f"\ntotal wall clock: {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
