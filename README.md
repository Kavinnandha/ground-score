# ground-score

An AI support agent for **@AmazonHelp** built from real Twitter support
threads, plus the evaluation harness that says how much to trust it.

The brief's framing — *"the proof is worth more than the system"* — is taken
literally. The agent is deliberately small: retrieve → classify → draft →
route. Most of the work is in the golden set, the evaluation harness, the
judge validation, and the honest accounting of what the headline number hides.

---

## Reproduce the headline results (no API key, no Kaggle account, no GPU)

```bash
pip install -r requirements.txt
make reproduce
```

No `make` (e.g. a stock Windows shell)? The target is a one-line wrapper:

```bash
pip install -r requirements.txt && python scripts/reproduce.py
```

`make reproduce` regenerates every table in `results/` from artifacts committed
to this repository: the 10k-thread corpus subsample, the embedding cache, and
the model response cache. It runs with API keys **stripped from the
environment**, so any step that is not fully cached fails loudly rather than
silently making live calls and producing numbers that differ from the published
ones. Cache misses must be zero; the script reports them.

### Models: one chain per role, not one provider per run

**The drafter and the judge must not share a lineage.** A model grading its own
output cannot rule out self-preference, so each *role* carries its own ordered
list of providers rather than inheriting a single global backend:

| Role | Default chain | Model | Why |
|---|---|---|---|
| classify + draft (`@fast`) | `$GROUNDSCORE_PROVIDER` → `ollama` | `claude-haiku-4-5` when that is `anthropic` | the cheap, fast model a high-volume triage route would actually run |
| judge (`@judge`) | `gemini` → `ollama` | `gemini-3.7-flash` → `gemma3:4b` | **a different vendor from the drafter**, so reply scores are not one family grading itself |
| self-preference probe (`@cross`) | same as `@fast` | the drafter's own model | the gap against the headline judge bounds self-preference |
| embeddings | `ollama` only | `nomic-embed-text` | the only backend here with an embeddings endpoint; served entirely from the committed cache |

`GROUNDSCORE_PROVIDER` still defaults to `ollama`, so set it (or
`GROUNDSCORE_ROLE_FAST`) to put drafting on a hosted model. Only `@judge` has a
default chain of its own — deliberately, so cross-vendor judging survives
someone changing the global provider.

Override any chain with a comma-separated list, preference first:

```bash
GROUNDSCORE_ROLE_JUDGE=ollama,gemini   # judge locally, fall back to hosted
GROUNDSCORE_ROLE_FAST=ollama           # draft locally too
GROUNDSCORE_PROVIDER=anthropic         # default head for roles without a chain
```

**Fallback never silently mixes two judges into one number.** A role is pinned
to whichever provider first serves it. If that provider dies mid-run (Gemini's
free tier is metered at *20 calls per day per model* — see `DECISIONS.md` #22),
the switch prints a warning, is recorded in `results/eval_*.json` under
`providers.switches`, and **every judged row is tagged with the model that
actually scored it** (`judge_model`). A split scored by two models is visible
rather than averaged away. Replay checks every provider in the chain, so
reordering it does not invalidate the committed cache.

**Practical note on the free Gemini tier:** 20 calls/day/model against ~450
judge calls means the Gemini judge will exhaust and fall through to Ollama
almost immediately. Either run the judge on a machine with Ollama and a GPU, or
set `GROUNDSCORE_ROLE_JUDGE=ollama` up front so one model scores the whole
split.

Embeddings never touch a hosted API. The embedding cache is keyed on
`(model, dim, text)` rather than on provider, so the committed
`nomic-embed-text` vectors stay valid whatever generation runs on; every corpus
and golden-set text is already in it, and a miss is fatal rather than silently
re-embedded into a different vector space.

Only needed to *regenerate* results:

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
   history, with the brand's actual replies (`src/groundscore/retrieve.py`).
2. **Classify** into one of the intents in `taxonomy/intents.yaml`, with the
   retrieved neighbours as dynamic few-shot context (`classify.py`).
3. **Draft** a ≤280-character reply that may assert only what the retrieved
   replies support, citing which ones it used in `grounded_in` (`draft.py`).
4. **Route** to auto-send or a human, with a stated reason
   (`route.py`) — deterministic rules first, LLM only for what survives them.

### Why rules run before the model

Escalations carry an attributable `triggered_rule`, so a support lead can see
*why* something escalated. Hard rules (account compromise, legal/press, safety,
PII in the message, an ungrounded draft) sit ahead of every threshold, and
`forced_escalation()` prevents the coverage curve from sweeping them away. A
confident classifier must not be able to auto-send a fraud report.

---

## How performance is reported

**Routing is not reported as accuracy.** The two errors have very different
costs: a needless escalation costs an agent thirty seconds, while a bad public
auto-reply is permanent. So the harness reports:

- **coverage** — share of messages handled with no human
- **false-auto rate** — of the replies we auto-sent, the share that should have
  gone to a human (customer-facing harm density)
- and the **coverage vs. false-auto curve** over the confidence threshold,
  so the operating point is visible rather than implied.

Every headline number carries a bootstrap 95% CI. With 80 test examples those
intervals are wide, and the report says so instead of quoting three decimals.

**Reply quality** is scored by an LLM judge against `eval/judge_rubric.md`
(groundedness, resolution, tone fit, safety, each 1–5, plus a binary
*would you send this?* gate). The judge's agreement with a human is measured,
not assumed — see below.

---

## Baselines

| System | Intent | Reply | Routing |
|---|---|---|---|
| `trivial_always_auto` | majority class | one canned string | never escalates |
| `trivial_always_escalate` | majority class | — | always escalates |
| `simple_tfidf_nn` | TF-IDF + logistic regression | nearest historical reply, verbatim | thresholds only, no LLM |
| `agent_no_retrieval` | LLM | LLM, no precedent | full router |
| `agent` | LLM + retrieval | LLM, grounded | full router |

The third baseline is an **ablation**, not a requirement of the brief. Trivial
and simple answer "is this dataset easy?"; only the ablation answers "does the
grounding actually do anything?".

Baselines are fitted on the **dev** split only. When the split being scored *is*
dev, the learned baselines are predicted **out-of-fold** (5-fold) instead of
being handed their own training rows — in-sample, `simple_tfidf_nn` returns
1.000 intent accuracy by memorisation, which would make the agent look hopeless
against a lookup table. Out-of-fold it scores what it can actually generalise
to. The test split is unaffected (fitted on dev, scored on test).

---

## Golden set

150 hand-adjudicated examples from threads held out of the retrieval index
entirely. Sampling and labelling procedure, including its limitations, is in
[`data/golden/LABELING_NOTES.md`](data/golden/LABELING_NOTES.md).

Short version: three strata (60% traffic-proportional, 25% rare-intent
oversample, 15% hand-picked hard cases), stratified over **unsupervised
clusters** rather than predicted intent to avoid circularity, scored
**separately** rather than pooled. Dev/test assigned by stable hash before any
label was written.

---

## Judge validation

`eval/judge_agreement.py` reports:

- **Spearman ρ** per rubric dimension against blind human scores
- **quadratic-weighted κ** on the `would_send` gate — the headline number,
  because that gate is the judgement routing depends on
- **mean bias** (judge − human): a judge can correlate well and still sit a
  full point high
- **verbosity probe**: identical replies re-judged with filler appended; any
  score movement is length bias
- **self-preference probe**: a subset is re-scored with the drafter's own
  model and the gap against the headline judge is reported, along with
  `same_vendor_as_drafter`. When the judge chain has fallen back onto the
  drafter's own vendor that flag flips true and the gap becomes a **discount to
  apply** to the reply-quality headline rather than a sanity check

Human scores are collected *before* judge output is read; the CLI refuses to
run otherwise.

---

## Full rebuild from raw data

Needs a key for whichever chains you use (`ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`) and/or Ollama running. Only `make embeddings` requires Ollama
specifically — no hosted backend here serves embeddings — and it is the one step
you should not need to rerun, since its cache is committed.

```bash
make full            # download -> corpus -> embeddings -> intent clusters
                     # then merge taxonomy/intents.draft.yaml -> intents.yaml by hand
make golden          # sample 150 candidates with weak labels
make label           # adjudicate by hand (interactive)
make relabel         # blind re-label of 50, for intra-annotator kappa
make tune            # fit routing thresholds on dev

make replies         # generate dev replies, WITHOUT judging them
make judge-human     # blind human reply scoring (interactive)
make eval            # score all systems on dev, judge included
make judge-agreement
make eval-test       # score the test split ONCE
```

**The order of those middle three is load-bearing, not stylistic.** Human reply
scores have to be recorded before the judge has produced an opinion of the same
replies — otherwise the "human" is anchored to the judge and the agreement
statistic measures nothing. But replies have to exist before anyone can score
them. So: `replies` (generate, don't judge) → `judge-human` (blind) → `eval`
(judge). `tools/score_replies_cli.py` refuses to run once judge scores exist for
the split, so getting this wrong fails loudly rather than quietly producing a
flattering κ.

The Kaggle dataset downloads without credentials (verified 2026-09);
`scripts/download_data.py` falls back to token auth and then to manual
instructions.

---

## Repository layout

```
configs/brand.yaml          brand choice + corpus/split/sampling parameters
configs/thresholds.yaml     routing thresholds, fitted on dev, frozen
taxonomy/intents.yaml       intent definitions — classifier prompt AND annotator guide
data/processed/threads.jsonl  committed 10k-thread corpus subsample
data/golden/                golden set + labelling notes
cache/                      committed model + embedding caches (keyless reproduction)
src/groundscore/            ingest, cleaning, retrieval, agent stages, baselines
eval/                       metrics, judge, judge validation, threshold tuning
tools/                      labelling and human-scoring CLIs
results/                    generated tables and raw outputs
tests/                      property tests for leakage, splits, caching, routing
```

---

## Results

Generated into `results/` by `make reproduce`:

- `brand_profile.md` — the evidence behind the brand choice
- `eval_dev.md`, `eval_test.md` — system comparison tables with CIs
- `threshold_sweep.json` — coverage vs. false-auto curve
- `judge_agreement.json` — judge-vs-human validation
- `outputs_*.jsonl`, `judge_*.jsonl` — per-example outputs and scores

The written analysis — problem framing, results, top-5 failure modes, the
mandatory *"what is misleading about my headline number"* section, and next
steps — is in [`REPORT.md`](REPORT.md). Non-obvious choices and citations are
in [`DECISIONS.md`](DECISIONS.md).
