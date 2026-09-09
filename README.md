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

`make reproduce` regenerates every table in `results/` from artifacts committed
to this repository: the 10k-thread corpus subsample, the embedding cache, and
the model response cache. It runs with API keys **stripped from the
environment**, so any step that is not fully cached fails loudly rather than
silently making live calls and producing numbers that differ from the published
ones. Cache misses must be zero; the script reports them.

### Models

Everything runs on **local models via [Ollama](https://ollama.com)** — no API
key, no quota, no network:

| Role | Model | Why |
|---|---|---|
| classify + draft | `qwen3:4b` | fits a 6GB GPU, follows a JSON schema |
| judge | `gemma3:4b` | **different family from the drafter**, so reply scores are not a model grading itself |
| embeddings | `nomic-embed-text` | 768-dim, ~330 texts/min locally |

This was not the original plan. The Gemini free tier turned out to be capped at
**20 `generate_content` calls per day, per model**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), against a workload of
roughly a thousand — so the hosted API could not run this evaluation at all.
Going local removed the ceiling and, as a side effect, made the judge a
genuinely independent model family and the whole pipeline reproducible without
credentials. The cost is capability: a 4B model is weaker than a hosted frontier
model, and the report attributes reply-quality limits to that rather than to the
architecture. Gemini remains supported via `GROUNDSCORE_PROVIDER=gemini`.

Only needed to *regenerate* results:

```bash
ollama pull qwen3:4b && ollama pull gemma3:4b && ollama pull nomic-embed-text
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

Every headline number carries a bootstrap 95% CI. With ~90 test examples those
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

Baselines are fitted on the **dev** split only — never on the split being
scored.

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
- **self-preference probe**: drafter and judge are already different families
  (`qwen3:4b` vs `gemma3:4b`), so this is a check rather than a correction —
  it re-scores a subset with the drafter's own model to confirm the gap that
  a same-family judge would have introduced

Human scores are collected *before* judge output is read; the CLI refuses to
run otherwise.

---

## Full rebuild from raw data

Needs Ollama running with the three models above. The embedding step takes
~30 minutes for 10k messages; the evaluation is bounded by local generation
speed rather than by any quota.

```bash
make full            # download -> corpus -> embeddings -> intent clusters
                     # then merge taxonomy/intents.draft.yaml -> intents.yaml by hand
make golden          # sample 150 candidates with weak labels
make label           # adjudicate by hand (interactive)
make relabel         # blind re-label of 50, for intra-annotator kappa
make tune            # fit routing thresholds on dev
make eval            # score all systems on dev
make judge-human     # blind human reply scoring (interactive)
make judge-agreement
make eval-test       # score the test split ONCE
```

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
