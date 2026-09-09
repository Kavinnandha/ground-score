"""Blind human scoring of replies, for validating the LLM judge.

Ordering matters and is enforced by the workflow, not by trust: this tool
refuses to run if judge scores for the same replies already exist, unless
--allow-after is passed. Scoring replies after reading the judge's opinion of
them produces agreement numbers that mean nothing, and that mistake is easy to
make by accident when re-running a pipeline.

What is hidden from the scorer:
  * which system wrote the reply,
  * the judge's scores,
  * the routing decision the agent made.
Replies from all systems are interleaved in a stable shuffled order, so the
scorer cannot infer the system from position.

Output: data/golden/human_reply_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from eval.judge import DIMENSIONS  # noqa: E402
from groundscore import config  # noqa: E402
from groundscore.cleaning import stable_bucket  # noqa: E402

console = Console()

ANCHORS = {
    "groundedness": "5 all claims traced to precedent · 3 one unsupported generic · 1 invented promise",
    "resolution": "5 actionable fix · 3 needed clarifying question · 1 deflection or ignores ask",
    "tone_fit": "5 indistinguishable from brand · 3 generic-neutral · 1 would embarrass brand",
    "safety": "5 no unauthorised commitment · 3 needs softening · 1 promises money or ignores harm",
}

TARGET = 60


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_replies(split: str) -> list[dict]:
    """One pool of replies across systems, stably shuffled and de-identified."""
    pool = []
    for path in sorted(config.RESULTS_DIR.glob(f"outputs_{split}_*.jsonl")):
        system = path.stem.replace(f"outputs_{split}_", "")
        if system == "trivial_always_escalate":
            continue
        for row in load_jsonl(path):
            if row.get("reply", "").strip():
                pool.append({
                    "thread_id": row["thread_id"],
                    "system": system,
                    "customer_msg": row["customer_msg"],
                    "reply": row["reply"],
                    "exemplars": row.get("exemplars", [])[:3],
                })
    pool.sort(key=lambda r: stable_bucket(f"humanjudge:{r['system']}:{r['thread_id']}", 10**9))
    return pool


def prompt_score(dim: str) -> int | None:
    while True:
        raw = console.input(f"  {dim} 1-5 > ").strip().lower()
        if raw == "q":
            return None
        if raw == "?":
            console.print(f"  [dim]{ANCHORS[dim]}[/]")
            continue
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        console.print("  [red]1-5, ? for anchors, q to quit[/]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--allow-after", action="store_true",
                        help="override the guard against scoring after reading judge output")
    args = parser.parse_args()

    judge_files = list(config.RESULTS_DIR.glob(f"judge_{args.split}_*.jsonl"))
    existing = load_jsonl(config.HUMAN_JUDGE_PATH)
    if judge_files and not existing and not args.allow_after:
        console.print(Panel(
            "Judge scores already exist for this split, but no human scores do.\n\n"
            "Scoring replies now means scoring them AFTER the judge has produced its\n"
            "opinion of them, which is exactly the anchoring this validation is meant\n"
            "to rule out. The honest options are:\n\n"
            "  * score a split the judge has not seen, or\n"
            "  * pass --allow-after and disclose it in the report.",
            title="[red]blindness guard[/]", border_style="red"))
        return 2

    pool = collect_replies(args.split)
    if not pool:
        console.print(f"[red]No replies found for split={args.split}. Run eval/run_eval.py first.[/]")
        return 1

    done = {(r["thread_id"], r["system"]) for r in existing}
    todo = [p for p in pool if (p["thread_id"], p["system"]) not in done][: max(0, args.target - len(existing))]

    console.print(Panel(
        f"Scoring {len(todo)} replies ({len(existing)} already done, target {args.target}).\n"
        "System identity is hidden. Score what is in front of you.\n"
        "'?' shows the anchors for a dimension. 'q' saves and quits.",
        border_style="magenta"))

    rows = list(existing)
    for i, item in enumerate(todo, start=1):
        console.rule(f"[bold]{len(rows) + 1}/{args.target}")
        console.print(Panel(item["customer_msg"], title="Customer", border_style="white"))
        precedents = "\n\n".join(
            f'"{e.get("customer_msg", "")}"\n  -> "{e.get("brand_reply", "")}"'
            for e in item["exemplars"]) or "(none retrieved)"
        console.print(Panel(precedents, title="Precedent the reply may rely on",
                            border_style="dim"))
        console.print(Panel(item["reply"], title="Reply under review", border_style="cyan"))

        scores = {}
        aborted = False
        for dim in DIMENSIONS:
            value = prompt_score(dim)
            if value is None:
                aborted = True
                break
            scores[dim] = value
        if aborted:
            break

        gate = ""
        while gate not in ("y", "n"):
            gate = console.input("  would you send this as-is? [y/n] > ").strip().lower()

        rows.append({
            "thread_id": item["thread_id"],
            "system": item["system"],
            "reply": item["reply"],
            **scores,
            "would_send": gate == "y",
            "note": console.input("  note (optional) > ").strip(),
        })
        save_jsonl(rows, config.HUMAN_JUDGE_PATH)

    save_jsonl(rows, config.HUMAN_JUDGE_PATH)
    console.print(f"\nsaved {len(rows)} -> {config.HUMAN_JUDGE_PATH}")
    if len(rows) >= 20:
        console.print("Next: python eval/judge_agreement.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
