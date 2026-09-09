"""Adjudication CLI for the golden set.

The human is shown one message at a time with a pre-filled weak label and must
accept or override it. Overrides require a note. Progress is saved after every
item, so labelling can be done in several sittings.

Why pre-fill at all
-------------------
Typing 200 labels from scratch invites fatigue drift: the last fifty get less
attention than the first fifty. Adjudicating a proposal is faster and more
consistent. The cost is anchoring bias -- the human is pulled toward the
proposal -- and that cost is made visible rather than denied: the override rate
is written into the golden file and reported. A very high acceptance rate means
the golden set partly measures agreement with the weak labeller, and the report
says so.

The re-label pass (`--relabel`) re-presents a 50-example subset with the labels
hidden, so intra-annotator agreement can be computed. That is a ceiling estimate
on label noise, NOT inter-annotator agreement -- there is one annotator here,
and the report states that limitation plainly.

Keys
----
  Enter    accept the proposed label
  1-9,0    pick an intent by number
  a / e    set action to auto / escalate
  ?        show the full labelling guideline
  n        show the brand's actual historical reply
  s        skip (revisit later)
  q        save and quit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from groundscore import config, taxonomy  # noqa: E402
from groundscore.cleaning import stable_bucket  # noqa: E402

console = Console()

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

RELABEL_SUBSET = 50


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


def show_guideline() -> None:
    console.print(Panel(taxonomy.render_for_annotator(), title="Labelling guideline",
                        border_style="cyan"))
    console.print(Panel(
        "auto      = a drafted reply could be sent publicly with no human reading it\n"
        "escalate  = a human must handle this thread\n\n"
        "Judge the MESSAGE, not the reply the model happened to produce.\n"
        "When genuinely torn, escalate -- and record 'ambiguous_or_underspecified'.",
        title="Routing decision", border_style="cyan"))


def render_item(item: dict, index: int, total: int, intents: list[str], hide_weak: bool) -> None:
    console.rule(f"[bold]{index}/{total}[/]  thread {item['thread_id']}  "
                 f"[dim]{item['stratum']} · {item['split']}[/]")

    console.print(Panel(item["customer_msg"], title="Customer message", border_style="white"))

    if item.get("adversarial_tags"):
        console.print(f"[yellow]hard-case tags:[/] {', '.join(item['adversarial_tags'])}")

    if not hide_weak:
        console.print(f"[dim]proposed:[/] [bold]{item['weak_intent']}[/] "
                      f"(conf {item['weak_confidence']:.2f}) — {item['weak_rationale']}")

    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, name in enumerate(intents, start=1):
        key = str(i) if i < 10 else "0"
        table.add_row(f"[cyan]{key}[/]", name)
    console.print(table)


def prompt_intent(item: dict, intents: list[str], hide_weak: bool) -> str | None:
    default = None if hide_weak else item["weak_intent"]
    while True:
        hint = f"[Enter={default}]" if default else ""
        raw = console.input(f"intent {hint} > ").strip().lower()
        if raw == "":
            if default:
                return default
            console.print("[red]no default — pick a number[/]")
            continue
        if raw == "q":
            return None
        if raw == "?":
            show_guideline()
            continue
        if raw == "n":
            console.print(Panel("\n\n".join(item.get("brand_replies", [])) or "(none)",
                                title="What the brand actually replied", border_style="green"))
            continue
        if raw == "s":
            return "__skip__"
        if raw.isdigit():
            idx = 10 if raw == "0" else int(raw)
            if 1 <= idx <= len(intents):
                return intents[idx - 1]
        if raw in intents:
            return raw
        console.print("[red]unrecognised — number, intent name, ? , n, s, or q[/]")


def prompt_action() -> str:
    while True:
        raw = console.input("action [a]uto / [e]scalate > ").strip().lower()
        if raw in ("a", "auto"):
            return "auto"
        if raw in ("e", "escalate"):
            return "escalate"
        console.print("[red]a or e[/]")


def prompt_escalation_reason() -> str:
    console.print("[dim]" + "  ".join(f"{i}={r}" for i, r in enumerate(ESCALATION_REASONS)) + "[/]")
    while True:
        raw = console.input("escalation reason > ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(ESCALATION_REASONS):
            return ESCALATION_REASONS[int(raw)]
        if raw in ESCALATION_REASONS:
            return raw
        console.print("[red]pick a number from the list[/]")


def label_pass(candidates: list[dict], existing: dict[str, dict], *,
               hide_weak: bool, out_path: Path, field_prefix: str = "") -> list[dict]:
    intents = taxonomy.names()
    labelled = dict(existing)
    todo = [c for c in candidates if c["thread_id"] not in labelled]

    console.print(f"[bold]{len(labelled)}[/] already labelled, [bold]{len(todo)}[/] remaining")
    if not todo:
        return list(labelled.values())

    for i, item in enumerate(todo, start=1):
        render_item(item, len(labelled) + 1, len(candidates), intents, hide_weak)

        intent = prompt_intent(item, intents, hide_weak)
        if intent is None:
            break
        if intent == "__skip__":
            continue

        action = prompt_action()
        reason = prompt_escalation_reason() if action == "escalate" else "none"

        overrode = (not hide_weak) and intent != item["weak_intent"]
        note = ""
        if overrode:
            console.print("[yellow]override — one line on why (required)[/]")
            while not note:
                note = console.input("note > ").strip()
        else:
            note = console.input("note (optional) > ").strip()

        confidence = console.input("your confidence 1-3 [3] > ").strip() or "3"

        record = {
            **{k: item[k] for k in ("thread_id", "customer_msg", "stratum", "split",
                                    "cluster", "adversarial_tags")},
            f"{field_prefix}intent": intent,
            f"{field_prefix}action": action,
            f"{field_prefix}escalation_reason": reason,
            f"{field_prefix}annotator_confidence": int(confidence) if confidence.isdigit() else 3,
            f"{field_prefix}note": note,
            f"{field_prefix}overrode_weak_label": overrode,
            "weak_intent": item["weak_intent"],
            "weak_confidence": item["weak_confidence"],
        }
        labelled[item["thread_id"]] = {**labelled.get(item["thread_id"], {}), **record}
        save_jsonl(list(labelled.values()), out_path)  # save every item

    return list(labelled.values())


def summarise(rows: list[dict]) -> None:
    if not rows:
        return
    overrides = sum(1 for r in rows if r.get("overrode_weak_label"))
    console.rule("[bold]summary")
    console.print(f"labelled            {len(rows)}")
    console.print(f"override rate       {overrides}/{len(rows)} = {overrides / len(rows):.1%}")
    console.print(f"intents             {dict(Counter(r['intent'] for r in rows))}")
    console.print(f"actions             {dict(Counter(r['action'] for r in rows))}")
    console.print(f"strata              {dict(Counter(r['stratum'] for r in rows))}")
    console.print(f"splits              {dict(Counter(r['split'] for r in rows))}")
    if overrides / len(rows) < 0.10:
        console.print(
            "[yellow]note: a low override rate means these labels largely agree with the "
            "weak labeller. Report this -- it caps how independent the golden set is.[/]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adjudicate golden-set labels")
    parser.add_argument("--relabel", action="store_true",
                        help="blind re-label of a 50-item subset for intra-annotator agreement")
    args = parser.parse_args()

    candidates = load_jsonl(config.GOLDEN_CANDIDATES_PATH)
    if not candidates:
        console.print(f"[red]No candidates at {config.GOLDEN_CANDIDATES_PATH}[/]")
        console.print("Run: python scripts/sample_golden.py")
        return 1

    if args.relabel:
        subset = sorted(candidates, key=lambda c: stable_bucket(f"relabel:{c['thread_id']}", 10**9))
        subset = subset[:RELABEL_SUBSET]
        out_path = config.GOLDEN_DIR / "golden_v1_relabel.jsonl"
        console.print(Panel(
            "Blind re-label pass. The original labels and the model's proposals are hidden.\n"
            "This measures INTRA-annotator agreement -- how consistent you are with yourself.\n"
            "It is an upper bound on label quality, not a substitute for a second annotator.",
            border_style="magenta"))
        existing = {r["thread_id"]: r for r in load_jsonl(out_path)}
        rows = label_pass(subset, existing, hide_weak=True, out_path=out_path,
                          field_prefix="relabel_")
        save_jsonl(rows, out_path)
        console.print(f"saved {len(rows)} -> {out_path}")
        return 0

    show_guideline()
    existing = {r["thread_id"]: r for r in load_jsonl(config.GOLDEN_PATH)}
    rows = label_pass(candidates, existing, hide_weak=False, out_path=config.GOLDEN_PATH)
    save_jsonl(rows, config.GOLDEN_PATH)
    summarise(rows)
    console.print(f"\nsaved -> {config.GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
