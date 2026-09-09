"""Turn the 2.8M-row twcs.csv into per-brand conversation threads.

Memory strategy
---------------
The raw CSV is ~500MB and the `text` column is most of it. Loading the whole
frame on a laptop is wasteful and unnecessary, so this runs in two passes:

  Pass 1 -- read only the four structural columns (ids, inbound flag, parent
            link). These are small. Build the reply graph, find every thread
            that involves a target brand, and collect the tweet ids needed.
  Pass 2 -- stream the file again, keeping `text`/`created_at` only for the
            ids collected in pass 1.

Peak memory stays in the low hundreds of MB instead of several GB.

Thread model
------------
A usable training/eval unit is: one inbound customer message that opens a
thread, plus the brand's reply chain. The unit of work for the agent is the
*first* customer message -- the agent classifies, drafts and routes at the
moment the ticket arrives, before any human has touched it. Later turns are
kept for context and for the "did the brand actually resolve this" signal, but
are not fed to the agent as input.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

from .cleaning import normalise

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = REPO_ROOT / "data" / "raw" / "twcs.csv"

STRUCT_COLS = ["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"]
TEXT_COLS = ["tweet_id", "created_at", "text"]

CHUNK = 250_000


def _read_structure(csv_path: Path) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=STRUCT_COLS,
        dtype={"tweet_id": "int64", "author_id": "string", "in_response_to_tweet_id": "float64"},
        chunksize=CHUNK,
    ):
        chunk["inbound"] = chunk["inbound"].astype(bool)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def _read_text_for(csv_path: Path, wanted: set[int]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for chunk in pd.read_csv(csv_path, usecols=TEXT_COLS, dtype={"tweet_id": "int64"}, chunksize=CHUNK):
        hit = chunk[chunk["tweet_id"].isin(wanted)]
        for tid, created, text in zip(hit["tweet_id"], hit["created_at"], hit["text"]):
            out[int(tid)] = (str(created), str(text))
    return out


def list_brands(csv_path: Path = RAW_CSV, top: int = 40) -> pd.DataFrame:
    """Outbound tweet volume per brand handle -- used to choose candidates."""
    counts: dict[str, int] = defaultdict(int)
    for chunk in pd.read_csv(
        csv_path, usecols=["author_id", "inbound"], dtype={"author_id": "string"}, chunksize=CHUNK
    ):
        outbound = chunk[~chunk["inbound"].astype(bool)]
        for author, n in outbound["author_id"].value_counts().items():
            counts[str(author)] += int(n)
    return (
        pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1])[:top], columns=["brand", "outbound_tweets"])
    )


def build_threads(brands: list[str], csv_path: Path = RAW_CSV) -> Iterator[dict]:
    """Yield one thread dict per (customer opener -> brand reply chain).

    Emitted shape:
        thread_id       str   -- root tweet id, stable across runs
        brand           str
        customer_msg    str   -- normalised opening customer message
        customer_msg_raw str
        brand_replies   list[{tweet_id, text, text_raw}]
        customer_turns  int   -- how many messages the customer sent in total
        created_at      str
    """
    brandset = {b.lower() for b in brands}
    struct = _read_structure(csv_path)

    author = dict(zip(struct["tweet_id"].tolist(), struct["author_id"].tolist()))
    inbound = dict(zip(struct["tweet_id"].tolist(), struct["inbound"].tolist()))
    parent: dict[int, int] = {
        int(t): int(p)
        for t, p in zip(struct["tweet_id"].tolist(), struct["in_response_to_tweet_id"].tolist())
        if pd.notna(p)
    }

    children: dict[int, list[int]] = defaultdict(list)
    for child, par in parent.items():
        children[par].append(child)

    brand_tweets = [
        int(t) for t, a in author.items()
        if isinstance(a, str) and a.lower() in brandset and not inbound.get(t, True)
    ]

    def root_of(tid: int) -> int:
        seen = set()
        cur = tid
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        return cur

    # Group brand replies by the thread root they belong to.
    roots: dict[int, str] = {}
    for bt in brand_tweets:
        r = root_of(bt)
        if inbound.get(r, False):  # thread must open with a customer, not the brand
            roots[r] = str(author[bt]).lower()

    needed: set[int] = set()
    thread_members: dict[int, list[int]] = {}
    for root, brand in roots.items():
        stack = [root]
        members: list[int] = []
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            members.append(cur)
            stack.extend(children.get(cur, []))
        thread_members[root] = sorted(members)
        needed.update(members)

    texts = _read_text_for(csv_path, needed)

    for root, brand in roots.items():
        if root not in texts:
            continue
        created, root_raw = texts[root]
        replies = []
        customer_turns = 0
        for tid in thread_members[root]:
            if tid not in texts:
                continue
            _, raw = texts[tid]
            a = author.get(tid)
            if isinstance(a, str) and a.lower() == brand and not inbound.get(tid, True):
                replies.append({"tweet_id": tid, "text": normalise(raw), "text_raw": raw})
            elif inbound.get(tid, False) and tid != root:
                customer_turns += 1
        if not replies:
            continue
        yield {
            "thread_id": str(root),
            "brand": brand,
            "customer_msg": normalise(root_raw),
            "customer_msg_raw": root_raw,
            "brand_replies": replies,
            "customer_turns": customer_turns + 1,
            "created_at": created,
        }


def write_jsonl(rows: Iterator[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
