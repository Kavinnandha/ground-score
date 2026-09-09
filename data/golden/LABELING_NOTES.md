# Golden set: how it was sampled and labelled

200 examples from AmazonHelp threads that were **held out of the retrieval
index entirely** (`split: golden_pool`, ~6% of the corpus by stable thread-id
hash). The agent cannot retrieve any thread it is evaluated on;
`tests/test_pipeline.py::test_no_leakage_between_retrieval_index_and_golden_pool`
asserts this.

## Sampling

Three strata, because one scheme cannot serve the three things this set has to
measure. Stratification is over **unsupervised clusters**, never over predicted
intent — using the classifier to choose the evaluation sample would be circular,
since any class the model never predicts would never be sampled and the
evaluation would be blind to exactly the classes it is worst at.

| Stratum | Share | Draw | What it is for |
|---|---|---|---|
| `proportional` | 60% (120) | proportional to cluster size | the only stratum that estimates **real traffic** performance |
| `rare` | 25% (50) | inverse-frequency over clusters | keeps macro-F1 on tail classes from being estimated off 2–3 examples |
| `adversarial` | 15% (30) | heuristic hard-case selection | the cases where auto-reply actually hurts |

Adversarial selection is by **surface heuristics, not model difficulty**: very
short (<40 chars), >70% caps, ≥3 emoji, ≥2 question marks, rage markers
(`!!!`, "worst", "disgrace"), multi-intent connectives, trailing sarcasm
("thanks a lot"). Selecting "cases the model finds hard" would have flattered
the model; these are cases that are objectively underspecified.

**The three strata are scored separately and never pooled into the headline.**
Pooling would mislead in both directions — the rare stratum drags accuracy
below real-traffic performance, and the proportional stratum hides the tail.
The report leads with the traffic-weighted (`proportional`) number.

## Dev/test split

40% dev (80) / 60% test (120), assigned by stable hash of the thread id
**at sampling time, before any label was written**. It therefore cannot have
been chosen after seeing which split flattered the results.

All prompt iteration and all threshold tuning happened on dev
(`eval/tune_thresholds.py` hard-codes the dev split — it is not a flag).
`eval/run_eval.py --split test` refuses to run without `--final`, and each
final run appends a timestamped record to `results/test_runs.jsonl`, so the
number of times the test set was scored is auditable rather than asserted.

## Labelling procedure

Each example was pre-filled with a weak label from the classifier and then
**adjudicated one at a time by hand** in `tools/label_cli.py`. Every override
required a written note.

Fields recorded per example: `intent`, `action` (auto/escalate),
`escalation_reason`, `annotator_confidence` (1–3), `note`,
`overrode_weak_label`.

**Why pre-fill, and what it costs.** Typing 200 labels cold invites fatigue
drift — the last fifty get less care than the first fifty. Adjudicating a
proposal is faster and more consistent. The cost is anchoring: the annotator is
pulled toward the proposal. That cost is measured rather than denied — the
**override rate is recorded in the golden file and reported**. A low override
rate means the golden set partly measures agreement with the weak labeller, and
the report says so in the "misleading number" section.

### Routing label definition

- `auto` — a drafted reply could go out publicly with no human reading it.
- `escalate` — a human must handle the thread.

Judged on the **message**, not on the reply the model happened to produce.
When genuinely torn: escalate, and record `ambiguous_or_underspecified`.

Escalation reason tags: `needs_account_access`, `angry_or_distressed`,
`policy_exception_or_refund`, `safety_or_legal`, `ambiguous_or_underspecified`,
`novel_no_precedent`, `contains_personal_data`, `multi_intent`.

## Label-quality evidence, and its limit

There is **one annotator**. So the agreement statistic reported is
**intra-annotator**: a 50-example subset was re-labelled blind (proposals and
original labels hidden) at least 24 hours later, via
`python tools/label_cli.py --relabel`, and Cohen's κ computed between the two
passes.

This is a **ceiling estimate on label noise, not inter-annotator agreement.**
It measures self-consistency. A second annotator would very likely agree less,
which means the true label noise is higher than reported and every metric
derived from these labels has more slack in it than the confidence intervals
alone suggest. This is stated again in the report's mandatory limitations
section; it is the single biggest caveat on the headline number.
