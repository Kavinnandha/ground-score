# Golden set: how it was sampled and labelled

150 examples from AmazonHelp threads that were held out of the retrieval index
entirely (`split: golden_pool`, ~6% of the corpus by stable thread-id hash). The
agent can't retrieve any thread it's evaluated on, and
`tests/test_pipeline.py::test_no_leakage_between_retrieval_index_and_golden_pool`
asserts that.

## Sampling

Three strata, because one scheme can't serve the three things this set has to
measure. Stratification is over unsupervised clusters, never over predicted
intent. Using the classifier to choose the evaluation sample would be circular:
any class the model never predicts would never get sampled, and the evaluation
would be blind to exactly the classes it's worst at.

| Stratum | Share | Draw | What it's for |
|---|---|---|---|
| `proportional` | 60% (90) | proportional to cluster size | the only stratum that estimates **real traffic** performance |
| `rare` | 25% (38) | inverse-frequency over clusters | keeps macro-F1 on tail classes from being estimated off 2–3 examples |
| `adversarial` | 15% (22) | heuristic hard-case selection | the cases where an auto-reply actually hurts |

**On the size: 150, not 200.** The brief allows 150–250. The binding constraint
is annotation, not compute. Every example gets adjudicated by hand by a single
annotator, and 150 is what can be labelled carefully in one sitting without the
fatigue drift that makes the last fifty worse than the first fifty. (The size
was fixed earlier, under a rate-limited backend I've since abandoned; the
reasoning that keeps it at 150 now is annotator attention.) I picked 150 to keep
the pipeline runnable end to end, not to make any number look better. The cost
is entirely in statistical power, and it's reported: the test split is 80
examples, so confidence intervals are wide and small differences between systems
aren't resolvable.

Adversarial selection is by surface heuristics, not by model difficulty: very
short (<40 chars), >70% caps, ≥3 emoji, ≥2 question marks, rage markers (`!!!`,
"worst", "disgrace"), multi-intent connectives, trailing sarcasm ("thanks a
lot"). Selecting "cases the model finds hard" would have flattered the model.
These are cases that are objectively underspecified.

The three strata get scored separately and never pooled into the headline.
Pooling misleads in both directions: the rare stratum drags accuracy below
real-traffic performance, and the proportional stratum hides the tail. The
report leads with the traffic-weighted (`proportional`) number.

### Known leak in the language filter

The English filter keys on function words and on Latin-1 diacritics, so it
misses short mixed-script messages where Latin brand names dominate, e.g.
`amazon echo 楽しいー`. Measured residue: **10 of 8,496 threads (0.12%)**, of
which exactly **1** landed in the 150 golden candidates, in the `adversarial`
stratum.

I left it in rather than fixing it. Fixing means rebuilding the corpus, which
changes the clustering, which changes the taxonomy, which invalidates the weak
labels: about an hour of compute plus a re-merge, to move 0.12% of the corpus.
And a foreign-language message is a legitimately hard routing case, which is
what the adversarial stratum is for. The residue is reported here so the number
is known rather than assumed to be zero.

## Dev/test split

dev 70 / test 80, assigned by stable hash of the thread id at sampling time,
before any label was written. So it can't have been chosen after seeing which
split flattered the results.

All prompt iteration and all threshold tuning happened on dev
(`eval/tune_thresholds.py` hard-codes the dev split, it isn't a flag).
`eval/run_eval.py --split test` refuses to run without `--final`, and each final
run appends a timestamped record to `results/test_runs.jsonl`, so the number of
times the test set was scored is auditable rather than asserted.

## Labelling procedure

Each example is pre-filled with a weak label from the classifier and then
adjudicated one at a time by hand in `tools/label_cli.py`. Every override needs
a written note.

Fields recorded per example: `intent`, `action` (auto/escalate),
`escalation_reason`, `annotator_confidence` (1–3), `note`,
`overrode_weak_label`.

**Why pre-fill, and what it costs.** Typing 150 labels cold invites fatigue
drift, where the last fifty get less care than the first fifty. Adjudicating a
proposal is faster and more consistent. The cost is anchoring: the annotator
gets pulled toward the proposal. That cost is measured rather than denied, since
the override rate is recorded in the golden file and reported. A low override
rate means the golden set partly measures agreement with the weak labeller, and
the report says so in the "misleading number" section.

### Routing label definition

- `auto` — a drafted reply could go out publicly with no human reading it.
- `escalate` — a human must handle the thread.

Judged on the message, not on the reply the model happened to produce. When
genuinely torn: escalate, and record `ambiguous_or_underspecified`.

Escalation reason tags: `needs_account_access`, `angry_or_distressed`,
`policy_exception_or_refund`, `safety_or_legal`, `ambiguous_or_underspecified`,
`novel_no_precedent`, `contains_personal_data`, `multi_intent`.

## Label-quality evidence, and its limit

There's one annotator, so the agreement statistic reported here is
**intra-annotator**. A 50-example subset gets re-labelled blind (proposals and
original labels hidden) at least 24 hours later, via
`python tools/label_cli.py --relabel`, and Cohen's κ is computed between the two
passes.

That's a ceiling estimate on label noise, not inter-annotator agreement. It
measures self-consistency. A second annotator would very likely agree less,
which means the true label noise is higher than reported and every metric built
on these labels has more slack in it than the confidence intervals alone
suggest. The report repeats this in the mandatory limitations section, because
it's the single biggest caveat on the headline number.
