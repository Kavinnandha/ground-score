# Decision log

Non-obvious choices and the reasoning behind them. Ordered roughly by how much
they affect the numbers.

---

**1. The brand was chosen by measurement, and the measurement overruled my guess.**
Before profiling I expected to pick `SpotifyCares`. The profiler
(`results/brand_profile.md`) rejected it: 37.7% of its first replies hand the
customer off rather than answer, versus 10.8% for `AmazonHelp`. On a deflecting
brand, "draft a reply grounded in how the brand historically resolved this"
degenerates into learning to emit a non-answer, and every reply-quality metric
silently becomes a measure of deflection mimicry. The deciding statistic was
handoff rate, not volume.

**1b. That metric was wrong the first time, and fixing it is part of the record.**
The original profiler counted only DM-style deflection ("DM us", "send us a
message") and scored AmazonHelp at 0.008. Reading actual retrieved replies
showed the real pattern: AmazonHelp almost never says "DM us", it says *"please
reach out to us here: <URL>"*. Measuring both forms moved AmazonHelp from 0.8%
to 10.8% — a 13× undercount. The brand ranking survived the correction
(Spotify 37.7%, Apple 55.1%, Uber 75.0%), so the decision stands, but it stands
on a number that was wrong until it was checked against the raw text. A bare
URL is deliberately *not* counted: 46% of AmazonHelp replies contain one and
most are genuinely useful (tracking pages, help articles), so the rule requires
a contact verb as well.

**2. AmazonHelp over hulu_support, despite hulu scoring better on reply substance.**
hulu_support has a lower handoff rate (0.057 vs 0.108) and a higher substantive
rate (0.927 vs 0.716), so on the deciding statistic alone it would win. It was
rejected on the other two columns: lexical diversity 0.124 vs 0.183, multi-turn
rate 0.383 vs 0.597. Its intent space collapses to roughly four classes — the
classification task would have been uninformatively easy — and with few customer
follow-ups there is little signal about whether a reply resolved anything. This
is the one place the brand choice is a judgement call rather than a
measurement, and it is worth flagging as such.

**3. Caches are committed, and a cache miss is fatal rather than a fallback.**
`make reproduce` runs with the API key stripped from the environment. Any
uncached call raises `OfflineCacheMiss` instead of silently going live. A
silent fallback would let a reviewer's run diverge from the published numbers
while appearing to succeed — the exact failure this design exists to prevent.

**4. No sentence-transformers, because Python 3.14 has no PyTorch wheels.**
The obvious default for short-text similarity is `all-MiniLM`. It is
unavailable on this interpreter, and pinning an older Python to get it would
complicate setup for every reviewer. That left two options: the Gemini
embeddings API, or pure-sklearn TF-IDF+SVD. Both are implemented and
selectable. Which one actually ships is decided in **#22** — by quota, not
preference.

**5. The judge must never be the same model as the drafter — and on the final backend that guarantee weakened.**
The requirement is constant: a model grading its own output cannot rule out
self-preference. How well it is met changed twice. Plan one was flash drafts /
pro judge (no pro quota). Plan two went local and met it best: drafter
`qwen3:4b`, judge `gemma3:4b` — different families, different weights, so
self-preference was largely removed by construction rather than measured.
Plan three (#28) put the drafter on a hosted Anthropic key. Briefly the judge
went there too, which *lost* the property: one vendor grading its own family,
with only a capability gap (`claude-opus-5` judging `claude-haiku-4-5`) standing
in for independence. That was a regression, so it was not kept.

The shipped design restores independence structurally by making the provider a
property of the **role** rather than of the run (#32): the drafter chain heads
at `anthropic`, the judge chain heads at `gemini`. Different vendors, different
training data — self-preference is again largely removed by construction rather
than merely bounded. The probe in `eval/judge_agreement.py` now reports
`same_vendor_as_drafter`, so a reader can tell which regime produced any given
number: false means the delta is a cross-vendor sanity check, true means the
judge has fallen back onto the drafter's vendor and the delta is a **discount to
apply** to the reply-quality headline.

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

**22. Everything runs on local models via Ollama, because the Gemini free tier is capped at 20 calls/day.**
This is the largest decision in the project and it was forced by a number that
took a while to surface. The free tier's real limit is not a rate but a daily
budget: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, **limit 20** — twenty
`generate_content` calls per day, per model. A full evaluation here needs on the
order of a thousand. Even spread across all six reachable Gemini models the
ceiling is ~120/day, so the hosted API was never going to complete this work,
however carefully the calls were paced.

So generation runs on `qwen3:4b` and embeddings on `nomic-embed-text`, both
local. Three things improve as a result:

  * **Reproducibility gets stronger, not weaker.** A reviewer with the same
    model tags reproduces the outputs. No key, no billing, no rate limit, and
    no model deprecation invalidating the cache later.
  * **The judge becomes genuinely independent.** Drafter is Qwen, judge is
    Gemma — different families, different weights. The original design could
    only *measure* same-family self-preference; this largely removes it by
    construction, and the cross-family probe becomes a check rather than a
    caveat.
  * **Volume stops being rationed.** Bias probes and ablations become free, so
    the evaluation can be thorough.

The cost is capability: a 4B local model is clearly weaker than Gemini flash at
instruction-following and JSON discipline. That shows up directly in reply
quality, and the report attributes it to the model rather than implying the
architecture is the ceiling. The Gemini path stays implemented and is one
environment variable away (`GROUNDSCORE_PROVIDER=gemini`).

**22b. TF-IDF was the shipped retrieval backend for part of this build, and is now the fallback.**
The free-tier embedding endpoint advertises 100 texts/minute. In practice three
separate runs stalled after 40, 600 and 650 of the 10,026 messages, with the
server returning 429s carrying retry hints it then did not honour. Embedding the
corpus was not achievable that way. TF-IDF + TruncatedSVD carried the pipeline
until local `nomic-embed-text` turned out to be already installed and to run at
~324 texts/minute with no quota at all, which is strictly better: real semantic
embeddings, so paraphrases actually match. TF-IDF remains as the keyless
fallback for anyone with neither a key nor Ollama, and it is still what the
`tfidf` backend selects. Its weakness is real and worth stating: char n-grams
match surface form, so "can't log in" and "password reset loop" sit further
apart than they should.

**23. Classification and drafting share one model call.**
Originally a response to Gemini's rate limit; kept after the move to local
models because a 4B model on a 6GB GPU is the new bottleneck and the argument
is unchanged. Running classify and draft as separate calls doubles the wall
clock of every evaluation. They share their entire context and
the drafter needs the intent anyway, so `respond.py` merges them. The split path
is kept and still works. What is given up is disclosed rather than glossed: the
model sees the drafting instructions before committing to an intent, so the
intent label is no longer independent of the reply, and a fluent draft can
rationalise its own label.

**24. The golden set is 150, not 200.**
The brief allows 150–250. At ~5 RPM the full design ran to roughly 8 hours of
API time. 150 keeps the whole pipeline runnable end to end. The cost is
statistical power only — the test split is 80 examples, so the confidence
intervals are wide and small between-system differences are not resolvable. That
is stated in the report rather than papered over, and it was chosen before any
result was seen, not after.

**25. The k-sweep for clustering was cut from 17 values to 8.**
The original sweep (k = 8..24, `n_init=10`) is 170 KMeans fits over 9.4k x 256
vectors and ran for over 45 minutes without finishing. Stepping k by 2 with
`n_init=4` gives the same argmax far faster. Silhouette on short-text embeddings
rises monotonically with k in this range anyway, so the sweep is a starting
point for the human merge, not a precise model-selection result — treating it as
the latter would be over-reading a weak signal.

**26. Clustering caught that a sixth of the corpus was not English, and the script filter had missed it.**
The first clustering run produced twelve clusters, of which four were
*languages* rather than intents: French, Spanish, German and Portuguese, about
16% of the corpus. The existing filter checked Latin **script**, which all four
pass trivially. So the taxonomy was partly a language taxonomy, and any intent
class would have been diluted by non-English traffic that the agent has no way
to serve.

The fix is a dependency-free English check, and its first version was wrong in
an instructive way: testing only for *absence of English function words*
rejected 20.7% of the corpus against a true non-English share near 16%, because
terse but genuinely English tweets ("ur app shows expected delivery on 25th")
contain almost no function words. Requiring **positive evidence of another
language** — function words from the four languages actually present — brought
it to 15.3%, catching 30 of 32 known non-English exemplars while keeping 64 of
64 English ones. It errs toward keeping, which is the right direction: a stray
foreign message in the corpus is a much smaller problem than silently
discarding English traffic.

Multilingual support was already declared out of scope, so this traffic is
removed rather than served badly. Corpus: 10,026 -> 8,496 threads.

**27. The cluster namer had to be shown the labels it had already used.**
Left alone, it named 8 of 12 clusters `delivery_status`. The clusters were
genuinely different — late delivery, missing package, delivered-to-wrong-address,
and disputes about the Prime next-day promise are distinct problems with
distinct resolutions — but the model defaulted to the most obvious topic every
time. Passing the already-assigned labels and requiring a distinct one forces
it to articulate what separates each cluster. Without this the taxonomy would
have discarded most of the structure the clustering actually found, and the
classification task would have looked far easier than it is.

**28. The shipped backend is a hosted Anthropic key, reversing #22 — because the hardware the local plan assumed did not exist.**
Decision #22 moved everything local to escape the Gemini daily cap, and that
was right at the time. It assumed a discrete GPU ("fits a 6GB GPU"). The machine
this was finished on has Intel integrated graphics and 8 CPU cores, where a 4B
model runs at roughly 60–90s per call against a ~1000-call evaluation: 12–24
hours per full run, and every prompt change costs another overnight. That is not
a backend, it is a bottleneck, and it would have forced the evaluation to shrink
to fit — exactly the pressure this project is supposed to resist.

So generation moved to `GROUNDSCORE_PROVIDER=anthropic`: `claude-haiku-4-5`
classifies and drafts, `claude-opus-5` judges. What this costs, stated plainly:

  * **Keyless reproduction is gone for *regeneration*.** It survives for
    *replay*: the response cache is still committed and `make reproduce` still
    runs with keys stripped and `GROUNDSCORE_OFFLINE=1`. A reviewer reproduces
    the published numbers with no key; only changing a prompt needs one.
  * **The independent-family judge is gone** (see #5), replaced by a measured
    correction rather than a structural guarantee.
  * **Model deprecation can invalidate regeneration later** in a way local model
    tags would not have.

What it buys is the ability to run the evaluation at all on this hardware,
several times, including the bias probes and the ablation — and a drafter that
is a realistic production choice for a high-volume triage route rather than a
4B model chosen because it fit in RAM.

Embeddings did **not** move: Anthropic serves none. The embedding cache is keyed
on `(model, dim, text)` and not on provider, so the committed `nomic-embed-text`
vectors remain valid and every corpus and golden text is already in them. A miss
raises rather than silently re-embedding into a second vector space, which would
have quietly changed what "similar" means half way through an index.

**29. `temperature` is not sent to the models that reject it.**
Sampling parameters were removed on the current Anthropic generation:
`temperature=0.0` to `claude-opus-5` is a 400, not a warning. The adapter sends
temperature only to models that still accept it and sends `effort` to the rest.
The pipeline pins temperature 0.0 everywhere it can, but this is a reminder that
temperature never carried reproducibility here — the committed cache does, which
is why that was built first.

**30. The learned baselines are scored out-of-fold on dev, because the obvious way to score them was in-sample.**
`build_systems()` fits the baselines on dev. Scoring `--split dev` then handed
`simple_tfidf_nn` its own training rows, and it returned **1.000 intent
accuracy** — not a strong baseline, a memorisation check. Out-of-fold (5-fold)
it scores **0.386** on the same rows. The gap is the whole point: the in-sample
version would have made the agent look hopeless against a baseline that had
simply looked up the answer, and the natural response to that table would have
been to "fix" an agent that was not broken. The test split was never affected
(fitted on dev, scored on test, which is correct), but dev is the table you
iterate against, so a wrong number there steers every decision after it.
`tests/test_pipeline.py::test_fitted_baselines_are_scored_out_of_fold_on_their_fit_split`
asserts no fold predicts a row it trained on.

**31. Human reply scoring is sequenced before the judge runs, and the tooling enforces it.**
The documented order was `tune -> eval -> judge-human`, which cannot work:
`make eval` runs the judge, and `tools/score_replies_cli.py` then refuses to
start because scoring replies the judge has already graded is precisely the
anchoring the validation exists to rule out. But replies must exist before a
human can score them, so "just score first" is not available either. Resolved
with a `make replies` target that generates dev replies with `--no-judge`, so
the real order is **replies -> judge-human -> eval -> judge-agreement**. The
guard was already correct; the documentation was wrong, which is the more
dangerous of the two failure modes, because a guard that fires is noticed and a
wrong runbook is followed.

**32. The provider is a property of the role, not of the run — with an ordered fallback chain per role.**
`GROUNDSCORE_PROVIDER` was global: one backend served drafting, judging and the
self-preference probe. That makes the single most important property of the
judge — that it does not share the drafter's lineage — an accident of which key
happened to be set. Roles now carry their own chains (`GROUNDSCORE_ROLE_JUDGE`,
default `gemini,ollama`, against a drafter defaulting to `anthropic`), so
cross-vendor judging is the default rather than something to remember.

The chain also solves the quota problem #22 ran into, without pretending it does
not exist. Three properties make the fallback safe to trust:

  * **Replay tries every provider in the chain before declaring a miss.** A
    cache built when the judge ran on Gemini still replays after the chain is
    reordered, so `make reproduce` does not start demanding live calls for
    numbers that are already published.
  * **A role is pinned to whichever provider first serves it.** Without pinning,
    a chain would flap back to the preferred provider on every call and a split
    would end up scored by two models in an interleaved pattern nobody could
    reconstruct.
  * **A forced switch is loud and recorded.** It warns on stderr, lands in
    `results/eval_*.json` under `providers.switches`, and every judged row is
    tagged with the model that actually scored it. A split scored by two judges
    is then *visible* rather than averaged into one number — which is the only
    honest option, because those rows cannot be pooled.

The remaining exposure is stated rather than engineered away: on Gemini's free
tier (20 calls/day/model) the judge will exhaust almost immediately against ~450
judge calls, so in practice the judge must either run locally on a GPU box or on
a paid tier. The chain makes that a visible operational fact instead of a silent
change of judge half way down the table.

---

## Borrowed / cited

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  (Kaggle, `thoughtvector`), used under its Kaggle licence.
- `scikit-learn` for TF-IDF, KMeans, silhouette, logistic regression, and the
  classification metrics; `scipy` for Spearman.
- `anthropic` Python SDK for the shipped hosted backend; `google-genai` SDK for
  the Gemini path; Ollama's HTTP API (no SDK) for the local path.
- Percentile bootstrap and quadratic-weighted κ are standard methods; the
  implementations here are written against `sklearn`/`numpy` primitives rather
  than copied.
- The judge rubric structure (anchored 1–5 dimensions plus a separate binary
  ship/no-ship gate) follows the common LLM-as-judge pattern; the dimensions,
  anchors and the deflection rule are written for this brand and this task.
