# ground-score

An AI support agent for **@AmazonHelp**, built from real Twitter support
threads, plus the evaluation harness that tells you how much to trust it.

The brief says the proof is worth more than the system, so I took that
literally. The agent itself is small: retrieve → classify → draft → route.
Most of the effort went into the golden set, the eval harness, checking the
judge against an independent annotator, and being upfront about what the
headline number hides.

---

## Reproduce the headline results (no API key, no Kaggle account, no GPU)

```bash
pip install -r requirements.txt
make reproduce
```

If you don't have `make` (stock Windows shell, say), the target is a one-line
wrapper:

```bash
pip install -r requirements.txt && python scripts/reproduce.py
```

`make reproduce` rebuilds every table in `results/` from artifacts that are
committed here: the 8.5k-thread corpus subsample, the embedding cache, and the
model response cache. It runs with API keys stripped out of the environment, so
any step that isn't fully cached blows up instead of quietly making live calls
and handing you numbers that don't match the published ones. Cache misses have
to be zero, and the script prints the count.

### Models: one chain per role, not one provider per run

The drafter and the judge must not share a lineage. A model grading its own
output can't rule out self-preference. So each *role* carries its own ordered
list of providers instead of inheriting one global backend:

| Role | Default chain | Model | Why |
|---|---|---|---|
| classify + draft (`@fast`) | `$GROUNDSCORE_PROVIDER` → `ollama` | `qwen3:4b` locally, `gemini-3.1-flash-lite` when hosted | the high-volume role, ~1000 calls per full run, which no free hosted tier covers |
| judge (`@judge`) | `gemini` → `ollama` | `gemini-3.5-flash-lite` → `gemma3:4b` | different vendor and family from the drafter, so reply scores aren't one lineage grading itself |
| self-preference probe (`@cross`) | same as `@fast` | the drafter's own model | the gap against the headline judge bounds self-preference |
| embeddings | `ollama` only | `nomic-embed-text` | keyless, no quota, and the committed vectors are keyed to this tag |

There are two providers: `ollama` and `gemini`. `GROUNDSCORE_PROVIDER` defaults
to `ollama`, so drafting is local and only judging is hosted. `@judge` is the
only role with a default chain of its own, and that's on purpose: cross-vendor
judging should survive somebody flipping the global provider.

Override any chain with a comma-separated list, most preferred first:

```bash
GROUNDSCORE_ROLE_JUDGE=ollama,gemini   # judge locally, fall back to hosted
GROUNDSCORE_ROLE_FAST=gemini,ollama    # draft on a hosted model (paid key advised)
GROUNDSCORE_PROVIDER=gemini            # default head for roles without a chain
```

Fallback never quietly blends two judges into one number. A role gets pinned to
whichever provider serves it first. If that provider dies mid-run (the free
Gemini tier meters requests per day per model, see the table below and
`DECISIONS.md` #12) the switch prints a warning, lands in
`results/eval_*.json` under `providers.switches`, and every judged row carries
the model that actually scored it in `judge_model`. A split scored by two
models stays visible instead of being averaged away. Replay checks every
provider in the chain, so reordering it doesn't invalidate the committed cache.

I picked the model ids on the free tier's daily budget, not on capability. The
ceiling that actually bites is requests-per-day-per-model, and it varies by
700x across the ids this key can reach. Google stopped publishing that table on
the rate-limits page (it now sends you to the AI Studio dashboard), so these are
probed, not quoted:

| Model | RPM | RPD | Verdict |
|---|---:|---:|---|
| `gemini-3.x-flash` | 5 | 20 | unusable, a judged split is ~450 calls |
| `gemini-3.5-flash-lite` | 15 | 500 | **the judge.** 6/6 reachable when probed; covers one split per day |
| `gemini-3.1-flash-lite` | 15 | 500 | same budget on paper, but **0/8 reachable** — every call returns 503 "high demand", schema or no schema. It is the hosted drafter id, a role that doesn't run by default |
| `gemma-4-31b-it` | 8 | 14,400 | escape hatch only: ~15s a call, returns 503 under ordinary load, and 500 on `response_schema` unless you send a system instruction with it (both handled, but a judge that intermittently 503s can't carry a headline number) |

A model being listed by `models.list()` and documented does not mean you can
call it. That cost a run to find out, and it is why the chain exists.

Live calls are paced per model from that table rather than by one global rate,
and the daily count is enforced in-process: hitting it raises
`DailyQuotaExhausted` so the chain drops through to Ollama immediately, instead
of collecting a retry ladder of 429s on every remaining row. What each model
actually served goes into `results/eval_*.json` under `providers.live_calls`.

If 500/day isn't enough for the day's work, set `GROUNDSCORE_ROLE_JUDGE=ollama`
up front so one model scores the whole split. That's honest; a mid-split
fallback is only visible.

This is not hypothetical. Scoring the test split, the chain failed over mid-run
— on a **socket error**, with 39 of 500 daily calls used, not on quota. The
result is `results/judge_test_agent.jsonl` carrying 38 rows scored by
`gemini-3.5-flash-lite` and 42 by `gemma3:4b`, with the switch and its cause in
`results/eval_test.json`. The mechanism worked and the number is still unusable,
so the test reply scores are excluded from the report and reply quality is a dev
result. Visible-and-mixed beats silent-and-mixed; neither is a headline.

Embeddings never touch a hosted API. The embedding cache is keyed on
`(model, dim, text)` rather than on provider, so the committed
`nomic-embed-text` vectors stay valid whatever generation runs on. Every corpus
and golden-set text is already in there, and a miss is fatal rather than
silently re-embedded into a different vector space.

You only need keys to *regenerate* results:

```bash
pip install -r requirements.txt   # then set keys in .env for the chains you use
```

```bash
make test     # property tests: leakage, split stability, cache, routing rules
```

---

## What the agent does

For each incoming customer message:

1. **Retrieve** the 5 most similar past customer messages from this brand's
   history, along with what the brand actually replied
   (`src/groundscore/retrieve.py`).
2. **Classify** into one of the intents in `taxonomy/intents.yaml`, using those
   neighbours as dynamic few-shot context (`classify.py`).
3. **Draft** a ≤280-character reply that may only assert what the retrieved
   replies support, and has to cite which ones it used in `grounded_in`
   (`draft.py`).
4. **Route** to auto-send or to a human with a stated reason (`route.py`).
   Deterministic rules run first; the LLM only sees what survives them.

### Why rules run before the model

Escalations come with an attributable `triggered_rule`, so a support lead can
see *why* something escalated. Hard rules (account compromise, legal/press,
safety, PII in the message, an ungrounded draft) sit ahead of every threshold,
and `forced_escalation()` stops the coverage curve from sweeping them away. A
confident classifier should not be able to auto-send a fraud report.

### Watching one message go through

```bash
make ui          # or: python tools/ui.py
```

A single page on `http://127.0.0.1:8000` that runs one message through the
pipeline and shows every stage: the routing decision and which rule produced
it, the drafted reply, the classifier's rationale, and the precedent the draft
was allowed to lean on. It's `http.server` and one HTML string, no web
dependency, because `make reproduce` has to stay installable from
`requirements.txt` alone.

It calls a live provider, so drafting needs Ollama running (or
`GROUNDSCORE_PROVIDER=gemini` with a key). The cached replay path is
`make reproduce`, not this. Retrieval uses the configured backend by default;
`--backend tfidf` runs the index keyless. `--no-retrieval` gives you the
ungrounded ablation, so you can sit it in a second tab next to the grounded one.

---

## How performance is reported

Routing is not reported as accuracy. The two errors cost very different
amounts: a needless escalation costs an agent thirty seconds, a bad public
auto-reply is permanent. So the harness reports:

- **coverage**, the share of messages handled with no human
- **false-auto rate**, the share of the replies we auto-sent that should have
  gone to a human (customer-facing harm density)
- the **coverage vs. false-auto curve** over the confidence threshold, so the
  operating point is visible instead of implied

Every headline number carries a bootstrap 95% CI. With 80 test examples those
intervals are wide, and the report says so instead of quoting three decimals.

Reply quality is scored by an LLM judge against `eval/judge_rubric.md`
(groundedness, resolution, tone fit, safety, each 1–5, plus a binary
*would you send this?* gate). How well the judge agrees with a human is
measured rather than assumed. See below.

---

## Baselines

| System | Intent | Reply | Routing |
|---|---|---|---|
| `trivial_always_auto` | majority class | one canned string | never escalates |
| `trivial_always_escalate` | majority class | — | always escalates |
| `simple_tfidf_nn` | TF-IDF + logistic regression | nearest historical reply, verbatim | thresholds only, no LLM |
| `agent_no_retrieval` | LLM | LLM, no precedent | full router |
| `agent` | LLM + retrieval | LLM, grounded | full router |

The third one is an ablation, not something the brief asked for. Trivial and
simple answer "is this dataset easy?". Only the ablation answers "does the
grounding actually do anything?".

Baselines are fitted on the **dev** split only. When the split being scored
*is* dev, the learned baselines get predicted out-of-fold (5-fold) instead of
being handed their own training rows. In-sample, `simple_tfidf_nn` returns
1.000 intent accuracy by memorisation, which makes the agent look hopeless
against a lookup table. Out-of-fold it scores what it can actually generalise
to. The test split isn't affected either way (fitted on dev, scored on test).

---

## Golden set

150 adjudicated examples from threads held out of the retrieval index entirely.
The sampling and labelling procedure, limitations included, is in
[`data/golden/LABELING_NOTES.md`](data/golden/LABELING_NOTES.md).

> **Who labelled these.** I manually adjudicated all 150 examples against
> `taxonomy/intents.yaml`, one at a time. The weak labels were model-generated
> proposals only; the final intent, routing action and notes in
> `golden_v1.jsonl` are my decisions. The optional blind model pass is a
> sensitivity check, not the source of the submitted gold labels. The blind
> reference scores used to validate the LLM judge were also written by me before
> I read any judge output.

Short version: three strata (60% traffic-proportional, 25% rare-intent
oversample, 15% hand-picked hard cases), stratified over unsupervised clusters
rather than predicted intent so the sampling isn't circular, and scored
separately rather than pooled. Dev/test was assigned by stable hash before a
single label was written.

Label quality is measured, not asserted. A 50-row subset also has an optional
blind second pass by a *different* model (`make second-annotator`), and
`eval/label_agreement.py` reports κ on intent and on the auto/escalate action,
per stratum, with every disagreement listed. `make relabel` is the other
variant: the same annotator with the labels hidden, which measures
self-consistency instead. The output records which of the two produced it,
because they are not the same claim.

---

## Judge validation

> I wrote the blind reference scores before running the judge, with the system
> identity hidden. The resulting statistics are judge-versus-human agreement.
> `tools/score_replies_cli.py` preserves that ordering by refusing to run once
> judge scores exist for the split.

`eval/judge_agreement.py` reports:

- **Spearman ρ** per rubric dimension against blind reference scores
- **quadratic-weighted κ** per rubric dimension, and **Cohen's κ** on the
  binary `would_send` gate, which is the headline number because that gate is
  the judgement routing depends on
- **mean bias** (judge − human), since a judge can correlate well and still sit
  a full point high
- a **verbosity probe**: identical replies re-judged with filler appended, and
  any score movement is length bias
- a **self-preference probe**: a subset re-scored with the drafter's own model,
  reporting the gap against the headline judge along with
  `same_vendor_as_drafter`. If the judge chain has fallen back onto the
  drafter's own vendor, that flag flips true and the gap turns into a discount
  to apply to the reply-quality headline rather than a sanity check.

Reference scores get collected *before* anyone reads judge output. The CLI
refuses to run otherwise, and the scorer is shown the reply with the system
identity stripped and the replies from all systems interleaved in a stable
shuffle.

One blindness leak is worth naming rather than claiming perfect blinding: a
reply shown with "(no precedent retrieved)" can only have come from the
no-retrieval ablation or from a trivial baseline, and a reply that is character
for character identical to its top precedent can only have come from the
nearest-neighbour baseline. The system label is hidden; the *architecture* is
partly inferable from the artefact itself. That is a property of the systems
being compared, not something the interface can hide.

---

## The reference point nobody asks for: how good was the brand's own reply?

```bash
make reference        # or: python eval/reference_replies.py --split dev
```

Every other number in this repo is self-referential. "The agent scores 3.4 on a
rubric I wrote, judged by a model I chose" doesn't tell you whether that's good.
So the same blind judge scores the reply @AmazonHelp actually sent for each
golden-pool thread, against the same retrieved precedent, on the same rubric.
Two replies, one message, one judge that isn't told which is which.

It needs no labels, so it runs before the golden set is adjudicated, and it
answers three things the report would otherwise have to argue:

- **A human reference for reply quality.** Same rows, same rubric, a real
  support agent's answer.
- **Whether the deflection rule is fair.** The rubric scores "please reach out
  here" as a resolution failure. This measures that rule against the brand's
  real replies instead of defending it in prose.
- **How strict our own gate is.** The share of replies this brand really posted
  in public that our `would_send` gate would have blocked. If that share is
  high, the gate is stricter than the brand, and every coverage number has to be
  read against that.

It is deliberately not a fair fight, and the output says so in a `caveats`
field. The human had the account, the order and the tracking page open; the
agent had five old tweets. Groundedness in particular is not comparable — the
human's facts are backed by systems the judge can't see, so the rubric scores
them as unsupported. That dimension is reported but excluded from the headline
delta, which uses resolution, tone fit and safety only.

---

## Full rebuild from raw data

Needs Ollama running (drafting, embeddings) and `GEMINI_API_KEY` set (the
judge). `make embeddings` is the one step you shouldn't need to rerun, since
its cache is committed.

Local models run on the GPU: a 4B model at q4 fits a 6GB card with room for the
8K context, which is what makes local drafting a backend rather than a
bottleneck. On integrated graphics the same models take 60–90s per call and a
full run becomes an overnight job.

```bash
make full            # download -> corpus -> embeddings -> intent clusters
                     # then merge taxonomy/intents.draft.yaml -> intents.yaml by hand
make golden          # sample 150 candidates with weak labels
make label           # adjudicate by hand (interactive)
make relabel         # blind re-label of 50 by YOU -> intra-annotator kappa
make second-annotator  # blind re-label of the same 50 by a DIFFERENT model
make label-agreement   # kappa between the two passes, per stratum
make tune            # fit routing thresholds on dev

make replies         # generate dev replies, WITHOUT judging them
make judge-human     # blind reply scoring (interactive; must precede `eval`)
make eval            # score all systems on dev, judge included
make judge-agreement
make eval-test       # score the test split ONCE
```

The order of those middle three is load-bearing, not stylistic. Human reply
scores have to be recorded before the judge has an opinion about the same
replies, otherwise the "human" is anchored to the judge and the agreement
statistic measures nothing. But the replies have to exist before anyone can
score them. Hence `replies` (generate, don't judge) → `judge-human` (blind) →
`eval` (judge). `tools/score_replies_cli.py` refuses to run once judge scores
exist for the split, so getting this wrong fails loudly instead of quietly
producing a flattering κ.

The Kaggle dataset downloads without credentials (checked 2026-09), and
`scripts/download_data.py` falls back to token auth and then to manual
instructions.

---

## Repository layout

```
configs/brand.yaml          brand choice + corpus/split/sampling parameters
configs/thresholds.yaml     routing thresholds, fitted on dev, frozen
taxonomy/intents.yaml       intent definitions — classifier prompt AND annotator guide
data/processed/threads.jsonl  committed 8.5k-thread corpus subsample
data/golden/                golden set + labelling notes
cache/                      committed model + embedding caches (keyless reproduction)
src/groundscore/            ingest, cleaning, retrieval, agent stages, baselines
eval/                       metrics, judge, judge validation, threshold tuning
tools/                      labelling and human-scoring CLIs, plus the browser UI
results/                    generated tables and raw outputs
tests/                      property tests for leakage, splits, caching, routing
```

---

## Results

`make reproduce` writes these into `results/`:

- `brand_profile.md`, the evidence behind the brand choice
- `eval_dev.md`, `eval_test.md`, system comparison tables with CIs
- `threshold_sweep.json`, the coverage vs. false-auto curve
- `judge_agreement.json`, judge-vs-human validation
- `label_agreement.json`, κ between the two labelling passes, per stratum, with
  every disagreement listed
- `failure_analysis_dev.json`, failure modes ranked by frequency with the rows
  attached — routing errors, intent errors in the direction that removes a
  never-auto guard, and draft defects found by pattern rather than by the judge
- `outputs_*.jsonl`, `judge_*.jsonl`, per-example outputs and scores

### The short version of what it says

The classification result is real: on the traffic-proportional dev stratum
(n=39) intent accuracy is 0.667. Pooled over all 70 dev rows it is 0.729 against
0.314 for TF-IDF + logistic regression and 0.171 for the majority class, but the
pooled figure runs six points high because the rare stratum filled with easy
classes (report §5.4); quote 0.667 for expected traffic. On the held-out test
split (n=80, pooled) it is 0.650 against 0.475 and 0.075. **The routing
result is not**: the agent auto-handles 44% of dev and 61% of what it auto-sent
should have gone to a human, and no threshold on either available signal reaches
the 10% false-auto budget at any point that automates anything. The system as
built should not auto-send. That is reported rather than tuned around, and
`configs/thresholds.yaml` carries `budget_feasible: false` to say so.

The ablation is the part worth reading twice. Switching retrieval off moves
intent accuracy by less than its confidence interval (0.729 → 0.700) and moves
draft defects from 7 to **45** — including 32 replies asking a customer to post
an order number, account email or payment method on a public timeline. Grounding
does not make the model understand the customer. It stops it inventing a
procedure.

The written analysis (problem framing, results, top-5 failure modes, the
mandatory *"what is misleading about my headline number"* section, and next
steps) is in [`docs/ground-score-report.pdf`](docs/ground-score-report.pdf). Non-obvious choices and citations are in
[`DECISIONS.md`](DECISIONS.md).
