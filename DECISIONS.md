# Decision log

Non-obvious choices and the reasoning behind them. Ordered roughly by how much
they affect the numbers.

---

**1. The brand was chosen by measurement, and the measurement overruled my guess.**
Before profiling I expected to pick `SpotifyCares`. The profiler
(`results/brand_profile.md`) rejected it: 37.4% of its first replies are pure
deflection ("please DM us") versus 0.8% for `AmazonHelp` — 47×. On a
deflecting brand, "draft a reply grounded in how the brand historically
resolved this" degenerates into learning to emit a non-answer, and every
reply-quality metric silently becomes a measure of deflection mimicry. The
deciding statistic was deflection rate, not volume.

**2. AmazonHelp over hulu_support, despite hulu scoring higher on reply substance.**
hulu_support has a better substantive rate (0.978 vs 0.803) but lexical
diversity of 0.124 vs AmazonHelp's 0.183 and a multi-turn rate of 0.383 vs
0.597. Its intent space collapses to about four classes — the classification
task would have been uninformatively easy, and with few customer follow-ups
there is little signal about whether a reply resolved anything.

**3. Caches are committed, and a cache miss is fatal rather than a fallback.**
`make reproduce` runs with the API key stripped from the environment. Any
uncached call raises `OfflineCacheMiss` instead of silently going live. A
silent fallback would let a reviewer's run diverge from the published numbers
while appearing to succeed — the exact failure this design exists to prevent.

**4. Gemini embeddings instead of sentence-transformers.**
The dev machine runs Python 3.14, which has no PyTorch wheels. Rather than pin
an older interpreter and complicate setup, embeddings come from the API and are
cached to a committed `.npz`. A pure-sklearn TF-IDF+SVD backend remains as a
keyless fallback so every path is runnable with no key and no cache at all.

**5. Judge tiering was forced by quota, and the workaround is better than the plan.**
The plan was flash drafts / pro judge. This API key returns 429 on every `*-pro`
model. So: drafter `gemini-3.5-flash`, judge `gemini-3.7-flash` (different
generation), plus a **cross-family judge on `gemma-4-31b-it`** — different
weights, different family — run on a subset. A Gemini judge grading Gemini
drafts cannot rule out self-preference on its own; an outside-family judge
gives a measurable upper bound on it. That probe is a genuine improvement over
the original plan.

**6. Evaluation threads are excluded from the retrieval index by construction.**
Split assignment is a stable SHA-256 bucket of the thread id, computed at
corpus build time — not `hash()`, which is salted per process and would move
the split between runs. Without this the agent could retrieve the very thread
it is scored on and grade against its own answer key. Asserted in
`tests/test_pipeline.py`.

**7. The taxonomy YAML is simultaneously the classifier prompt and the annotator guideline.**
One file, two consumers. The alternative — a prompt in Python and a guideline in
Markdown — guarantees drift, and the resulting human/model disagreement gets
misread as model error when it is actually specification error.

**8. Taxonomy built by cluster-then-name-then-merge, not by asking an LLM for a taxonomy.**
An LLM asked to invent a taxonomy produces plausible classes that do not match
the real distribution: a "billing" class the brand never receives, and no class
for the traffic that actually dominates. Clustering first anchors the classes to
frequency mass; the LLM only *names* clusters; a human does the merging, which
is where the judgment lives.

**9. The golden set is stratified by unsupervised cluster, never by predicted intent.**
Stratifying on classifier output is circular: any class the model never predicts
would never be sampled, and the evaluation would be structurally blind to the
classes the model is worst at.

**10. The adversarial slice is chosen by surface heuristics, not by model difficulty.**
Caps ratio, emoji count, message length, rage markers, multi-intent
connectives. Selecting "examples the model gets wrong" would have produced a
slice that flatters whatever model built it. These are cases that are
objectively underspecified, independent of any system.

**11. The three strata are scored separately and never pooled into the headline.**
The golden set deliberately over-samples rare and adversarial cases, so its
pooled accuracy is *not* an estimate of production performance. The headline is
the traffic-weighted (`proportional`) stratum; the others are reported beside
it.

**12. Routing is reported as (coverage, false-auto rate), not accuracy.**
The two routing errors have wildly different costs — a needless escalation costs
an agent thirty seconds, a bad public auto-reply is permanent — and a single
accuracy number averages them together. `false_auto_rate` is defined as a share
of what was auto-sent ("3% of replies we sent should have had a human"), because
that maps to customer harm; the recall-style view is reported alongside it.

**13. Deterministic rules run before the LLM router, and hard rules cannot be swept away.**
Escalations get an attributable `triggered_rule`. A confident classifier must
not be able to auto-send an account-compromise report, so hard rules sit ahead
of every threshold, and `forced_escalation()` keeps the coverage curve from
sweeping them away. Tested directly.

**14. `grounded_in` makes groundedness checkable rather than a vibe.**
The drafter must cite which exemplars support its reply. An empty citation list
is treated as a routing signal (escalate), not a formatting nit — a reply the
model cannot trace to any precedent is precisely the one most likely to have
invented a refund policy. It does not *prove* groundedness (a model can cite and
still contradict), which is what the judge's groundedness dimension is for.

**15. The rubric scores deflection as a failure even though the brand really deflects.**
"Please DM us" scores 1 on resolution. This deliberately diverges from the
historical data: the rubric measures whether the customer was helped, not
whether the brand was imitated. Without it, the copy-the-human baseline would
win by construction and the evaluation would reward the wrong thing.

**16. Dev/test assignment happens before any label is written.**
Assigned by stable hash at sampling time, so it cannot have been chosen after
seeing which split flattered the results. `--split test` requires `--final` and
appends to `results/test_runs.jsonl` — the guard is trivially bypassable, but
bypassing it leaves a trace, which turns "scored once" into a checkable claim.

**17. Weak labels are pre-filled for adjudication, and the override rate is published.**
Adjudicating beats typing 200 labels cold (fatigue drift). The cost is anchoring
bias, so it is measured rather than denied: a low override rate means the golden
set partly measures agreement with the weak labeller, and the report says so.

**18. Label-quality evidence is intra-annotator κ, and is labelled as such.**
There is one annotator, so the re-label pass measures self-consistency — a
*ceiling* on label quality, not inter-annotator agreement. Reporting it as if it
were the latter would be the most misleading thing in the whole submission.

**19. Three baselines, including an ablation, not the two required.**
Trivial (canned reply) and simple (TF-IDF + verbatim nearest historical reply)
answer "is the dataset easy?". The third — the full LLM pipeline with retrieval
*disabled* — answers "does the grounding do anything?", which neither of the
required baselines can isolate.

**20. Embedding batch size is 50, not 100, for a quota-shape reason.**
The free tier meters ~100 texts/minute; a 100-text request needs a perfectly
empty window and starves on retry. 50 leaves headroom for two requests per
window. Discovered empirically after a job stalled — noted because it is the
kind of thing that looks arbitrary in code review.

**21. The human reply-scoring CLI refuses to run after judge scores exist.**
Scoring replies after reading the judge's opinion of them produces agreement
numbers that mean nothing. The guard is overridable with `--allow-after`, which
must then be disclosed.

---

## Borrowed / cited

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  (Kaggle, `thoughtvector`), used under its Kaggle licence.
- `scikit-learn` for TF-IDF, KMeans, silhouette, logistic regression, and the
  classification metrics; `scipy` for Spearman.
- `google-genai` SDK for Gemini access.
- Percentile bootstrap and quadratic-weighted κ are standard methods; the
  implementations here are written against `sklearn`/`numpy` primitives rather
  than copied.
- The judge rubric structure (anchored 1–5 dimensions plus a separate binary
  ship/no-ship gate) follows the common LLM-as-judge pattern; the dimensions,
  anchors and the deflection rule are written for this brand and this task.
