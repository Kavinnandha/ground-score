"""Load the intent taxonomy.

taxonomy/intents.yaml is deliberately the ONLY definition of the intent set. It
is rendered into the classifier prompt and it is rendered into the labelling
guideline shown in the annotation CLI. One file, two consumers.

The alternative -- a prompt in Python and a guideline in Markdown -- guarantees
drift: the prompt gains a class the guideline never mentions, the human labels
against a definition the model was never given, and the resulting disagreement
gets misread as model error rather than specification error.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from . import config

OTHER = "other"


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    never_auto: bool
    boundary_rule: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    escalation_note: str = ""


@lru_cache(maxsize=1)
def load() -> tuple[Intent, ...]:
    raw = yaml.safe_load(config.TAXONOMY_PATH.read_text(encoding="utf-8"))
    intents = tuple(
        Intent(
            name=item["name"],
            definition=item["definition"],
            never_auto=bool(item.get("never_auto", False)),
            boundary_rule=item.get("boundary_rule", ""),
            positive_examples=tuple(item.get("positive_examples", [])),
            negative_examples=tuple(item.get("negative_examples", [])),
            escalation_note=item.get("escalation_note", ""),
        )
        for item in raw["intents"]
    )
    names = [i.name for i in intents]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate intent names in taxonomy: {names}")
    if OTHER not in names:
        raise ValueError(
            f"taxonomy must contain an '{OTHER}' class: without an escape hatch the "
            "classifier is forced to mislabel novel traffic into the nearest class, "
            "which hides distribution shift instead of surfacing it"
        )
    return intents


def names() -> list[str]:
    return [i.name for i in load()]


def by_name(name: str) -> Intent | None:
    return next((i for i in load() if i.name == name), None)


def never_auto_names() -> set[str]:
    return {i.name for i in load() if i.never_auto}


def render_for_prompt(include_examples: bool = True) -> str:
    """The taxonomy as the classifier sees it."""
    blocks = []
    for intent in load():
        lines = [f"### {intent.name}", intent.definition]
        if intent.boundary_rule:
            lines.append(f"Boundary: {intent.boundary_rule}")
        if include_examples and intent.positive_examples:
            lines.append("Examples of this intent:")
            lines.extend(f'  - "{e}"' for e in intent.positive_examples)
        if include_examples and intent.negative_examples:
            lines.append("NOT this intent:")
            lines.extend(f'  - "{e}"' for e in intent.negative_examples)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_for_annotator() -> str:
    """The same taxonomy as the human annotator sees it, plus routing policy."""
    blocks = []
    for intent in load():
        lines = [f"{intent.name}", f"  {intent.definition}"]
        if intent.boundary_rule:
            lines.append(f"  Boundary: {intent.boundary_rule}")
        lines.append(f"  Never auto-handle: {'YES' if intent.never_auto else 'no'}")
        if intent.escalation_note:
            lines.append(f"  Escalation note: {intent.escalation_note}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
