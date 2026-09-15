# Decision log

The non-obvious choices and why I made them. Roughly ordered by how much each
one moves the numbers.

---

**1. The brand was chosen by measurement, and the measurement overruled my guess.**
Before profiling I expected to pick `SpotifyCares`. The profiler
(`results/brand_profile.md`) rejected it: 37.7% of its first replies hand the
customer off instead of answering, against 10.8% for `AmazonHelp`. On a
deflecting brand, "draft a reply grounded in how the brand historically resolved
this" degenerates into learning to emit a non-answer, and every reply-quality
metric quietly turns into a measure of deflection mimicry. The deciding
statistic was handoff rate, not volume.

**1b. That metric was wrong the first time, and the fix is part of the record.**
The original profiler counted only DM-style deflection ("DM us", "send us a
message") and scored AmazonHelp at 0.008. Reading actual retrieved replies
showed the real pattern: AmazonHelp almost never says "DM us", it says *"please
reach out to us here: <URL>"*. Counting both forms moved AmazonHelp from 0.8% to
10.8%, a 13× undercount. The ranking survived the correction (Spotify 37.7%,
Apple 55.1%, Uber 75.0%) so the decision stands, but it stood on a number that
was wrong until it got checked against raw text. A bare URL deliberately does
*not* count: 46% of AmazonHelp replies contain one and most are genuinely useful
(tracking pages, help articles), so the rule requires a contact verb too.

**2. AmazonHelp over hulu_support, even though hulu scores better on reply substance.**
hulu_support has a lower handoff rate (0.057 vs 0.108) and a higher substantive
rate (0.927 vs 0.716), so on the deciding statistic alone it wins. I rejected it
on the other two columns: lexical diversity 0.124 vs 0.183, multi-turn rate
0.383 vs 0.597. Its intent space collapses to about four classes, which would
have made the classification task uninformatively easy, and with few customer
follow-ups there's little signal about whether a reply resolved anything. This
is the one place the brand choice is judgement rather than measurement, and it's
worth flagging as such.

**3. Caches are committed, and a cache miss is fatal rather than a fallback.**
`make reproduce` runs with the API key stripped out of the environment. Any
uncached call raises `OfflineCacheMiss` instead of silently going live. A silent
fallback would let a reviewer's run drift away from the published numbers while
appearing to succeed, which is the exact failure this design exists to prevent.

**4. No sentence-transformers, because Python 3.14 has no PyTorch wheels.**
The obvious default for short-text similarity is `all-MiniLM`. It isn't
available on this interpreter, and pinning an older Python to get it would
complicate setup for every reviewer. That left the Gemini embeddings API or
pure-sklearn TF-IDF+SVD. Both are implemented and selectable. Which one actually
ships is settled in **#22**, by quota rather than by preference.

**5. The judge must never be the same model as the drafter, and on one backend that guarantee weakened.**
The requirement never changed: a model grading its own output can't rule out
self-preference. How well it was met changed twice. Plan one was flash drafts
with a pro judge (no pro quota). Plan two went local and met it best: drafter
`qwen3:4b`, judge `gemma3:4b`, different families and different weights, so
self-preference was mostly removed by construction rather than measured. Plan
three (#28) put the drafter on a hosted key, and for a while the judge sat with
the same vendor, which *lost* the property: one vendor grading its own family,
with only a capability gap standing in for independence. That was a regression,
so I didn't keep it.

The shipped design restores independence structurally by making the provider a
property of the **role** rather than of the run (#32). The drafter chain heads
at `ollama` (`qwen3:4b`), the judge chain heads at `gemini`
(`gemini-3.1-flash-lite`). Different vendors, different families, different
weights, so self-preference is again mostly removed by construction rather than
merely bounded. The probe in `eval/judge_agreement.py` reports
`same_vendor_as_drafter`, so a reader can tell which regime produced any given
number: false means the delta is a cross-vendor sanity check, true means the
judge has fallen back onto the drafter's vendor and the delta becomes a
**discount to apply** to the reply-quality headline.

**6. Evaluation threads are excluded from the retrieval index by construction.**
Split assignment is a stable SHA-256 bucket of the thread id, computed at corpus
build time. Not `hash()`, which is salted per process and would move the split
between runs. Without this the agent could retrieve the very thread it's being
scored on and grade itself against its own answer key. Asserted in
`tests/test_pipeline.py`.

**7. The taxonomy YAML is both the classifier prompt and the annotator guideline.**
One file, two consumers. The alternative (a prompt in Python, a guideline in
Markdown) guarantees drift, and then human/model disagreement gets misread as
model error when it's really a spec mismatch.

**8. Taxonomy built by cluster-then-name-then-merge, not by asking an LLM for a taxonomy.**
An LLM asked to invent a taxonomy produces plausible classes that don't match
the real distribution: a "billing" class the brand never receives, and no class
for the traffic that actually dominates. Clustering first anchors the classes to
frequency mass. The LLM only *names* clusters, and a human does the merging,
which is where the judgement lives.

**9. The golden set is stratified by unsupervised cluster, never by predicted intent.**
Stratifying on classifier output is circular: any class the model never predicts
would never be sampled, and the evaluation would be structurally blind to the
classes the model is worst at.

**10. The adversarial slice is chosen by surface heuristics, not by model difficulty.**
Caps ratio, emoji count, message length, rage markers, multi-intent
connectives. Picking "examples the model gets wrong" produces a slice that
flatters whatever model built it. These are cases that are objectively
underspecified, independent of any system.

**11. The three strata are scored separately and never pooled into the headline.**
The golden set deliberately over-samples rare and adversarial cases, so its
pooled accuracy is *not* an estimate of production performance. The headline is
the traffic-weighted (`proportional`) stratum, with the others reported beside
it.

**12. Routing is reported as (coverage, false-auto rate), not accuracy.**
The two routing errors have wildly different costs. A needless escalation costs
an agent thirty seconds; a bad public auto-reply is permanent. A single accuracy
number averages them together. `false_auto_rate` is defined as a share of what
was auto-sent ("3% of the replies we sent should have had a human"), because
that's what maps to customer harm. The recall-style view is reported alongside
it.

**13. Deterministic rules run before the LLM router, and hard rules can't be swept away.**
Escalations get an attributable `triggered_rule`. A confident classifier must
not be able to auto-send an account-compromise report, so hard rules sit ahead
of every threshold and `forced_escalation()` keeps the coverage curve from
sweeping them away. Tested directly.

**14. `grounded_in` makes groundedness checkable instead of a vibe.**
The drafter has to cite which exemplars support its reply. An empty citation
list is treated as a routing signal (escalate), not a formatting nit: a reply the
model can't trace to any precedent is precisely the one most likely to have
invented a refund policy. It doesn't *prove* groundedness, since a model can
cite and still contradict, which is what the judge's groundedness dimension is
for.

**15. The rubric scores deflection as a failure even though the brand really deflects.**
"Please DM us" scores 1 on resolution. That diverges from the historical data on
purpose: the rubric measures whether the customer was helped, not whether the
brand was imitated. Without it the copy-the-human baseline wins by construction
and the evaluation rewards the wrong thing.

**16. Dev/test assignment happens before any label is written.**
Assigned by stable hash at sampling time, so it can't have been chosen after
seeing which split flattered the results. `--split test` requires `--final` and
appends to `results/test_runs.jsonl`. The guard is trivially bypassable, but
bypassing it leaves a trace, which turns "scored once" into a checkable claim.

**17. Weak labels are pre-filled for adjudication, and the override rate is published.**
Adjudicating beats typing 150 labels cold (fatigue drift). The cost is anchoring
bias, so it's measured rather than denied: a low override rate means the golden
set partly measures agreement with the weak labeller, and the report says so.

**18. Label-quality evidence is intra-annotator κ, and it's labelled as such.**
There's one annotator, so the re-label pass measures self-consistency. That's a
*ceiling* on label quality, not inter-annotator agreement. Reporting it as if it
were the latter would be the most misleading thing in the whole submission.

**19. Three baselines including an ablation, not the two the brief requires.**
Trivial (canned reply) and simple (TF-IDF + verbatim nearest historical reply)
answer "is the dataset easy?". The third, the full LLM pipeline with retrieval
*disabled*, answers "does the grounding do anything?", which neither of the
required baselines can isolate.

**20. Embedding batch size is 50, not 100, for a quota-shape reason.**
The free tier meters ~100 texts/minute, and a 100-text request needs a perfectly
empty window and then starves on retry. 50 leaves headroom for two requests per
window. I found this empirically after a job stalled. Noting it because it's the
kind of constant that looks arbitrary in code review.

**21. The human reply-scoring CLI refuses to run after judge scores exist.**
Scoring replies after reading the judge's opinion of them produces agreement
numbers that mean nothing. The guard is overridable with `--allow-after`, and if
you use it you have to disclose it.

**22. Everything runs on local models via Ollama, because the Gemini free tier is capped at 20 calls/day.**
This is the biggest decision in the project and it was forced by a number that
took a while to surface. The free tier's real limit isn't a rate, it's a daily
budget: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, **limit 20**.
Twenty `generate_content` calls per day, per model. A full evaluation here needs
on the order of a thousand. Even spread across all six reachable Gemini models
the ceiling is ~120/day, so the hosted API was never going to finish this work
however carefully I paced the calls.

So generation runs on `qwen3:4b` and embeddings on `nomic-embed-text`, both
local. Three things get better as a result:

  * **Reproducibility gets stronger, not weaker.** A reviewer with the same
    model tags reproduces the outputs. No key, no billing, no rate limit, and no
    model deprecation invalidating the cache later.
  * **The judge becomes genuinely independent.** Drafter is Qwen, judge is
    Gemma: different families, different weights. The original design could only
    *measure* same-family self-preference; this mostly removes it by
    construction, and the cross-family probe becomes a check rather than a
    caveat.
  * **Volume stops being rationed.** Bias probes and ablations become free, so
    the evaluation can be thorough.

The cost is capability. A 4B local model is clearly weaker than Gemini flash at
instruction-following and JSON discipline. That shows up directly in reply
quality, and the report attributes it to the model rather than implying the
architecture is the ceiling. The Gemini path stays implemented and is one
environment variable away (`GROUNDSCORE_PROVIDER=gemini`).

**22b. TF-IDF was the shipped retrieval backend for part of this build, and is now the fallback.**
The free-tier embedding endpoint advertises 100 texts/minute. In practice three
separate runs stalled after 40, 600 and 650 of the 10,026 messages, with the
server returning 429s carrying retry hints it then didn't honour. Embedding the
corpus wasn't achievable that way. TF-IDF + TruncatedSVD carried the pipeline
until local `nomic-embed-text` turned out to be already installed and to run at
~324 texts/minute with no quota at all, which is strictly better: real semantic
embeddings, so paraphrases actually match. TF-IDF stays as the keyless fallback
for anyone with neither a key nor Ollama, and it's still what the `tfidf`
backend selects. Its weakness is real and worth stating: char n-grams match
surface form, so "can't log in" and "password reset loop" sit further apart than
they should.

**23. Classification and drafting share one model call.**
Originally a response to Gemini's rate limit. I kept it after the move to local
models because a 4B model on a 6GB GPU is the new bottleneck and the argument
hasn't changed: running classify and draft as separate calls doubles the wall
clock of every evaluation. They share their entire context and the drafter needs
the intent anyway, so `respond.py` merges them. The split path is kept and still
works. What that gives up is disclosed rather than glossed over: the model sees
the drafting instructions before committing to an intent, so the intent label is
no longer independent of the reply, and a fluent draft can rationalise its own
label.

**24. The golden set is 150, not 200.**
The brief allows 150–250. At ~5 RPM the full design came to roughly 8 hours of
API time. 150 keeps the whole pipeline runnable end to end. The cost is purely
statistical power: the test split is 80 examples, so confidence intervals are
wide and small between-system differences aren't resolvable. That's stated in
the report rather than papered over, and it was chosen before any result was
seen, not after.

**25. The k-sweep for clustering was cut from 17 values to 8.**
The original sweep (k = 8..24, `n_init=10`) is 170 KMeans fits over 9.4k x 256
vectors and ran for over 45 minutes without finishing. Stepping k by 2 with
`n_init=4` finds the same argmax far faster. Silhouette on short-text embeddings
rises monotonically with k in this range anyway, so the sweep is a starting
point for the human merge rather than a precise model-selection result. Treating
it as the latter would be over-reading a weak signal.

**26. Clustering caught that a sixth of the corpus wasn't English, and the script filter had missed it.**
The first clustering run produced twelve clusters, four of which were
*languages* rather than intents: French, Spanish, German and Portuguese, about
16% of the corpus. The existing filter checked Latin **script**, which all four
pass trivially. So the taxonomy was partly a language taxonomy, and any intent
class would have been diluted by non-English traffic the agent has no way to
serve.

The fix is a dependency-free English check, and its first version was wrong in
an instructive way. Testing only for *absence of English function words*
rejected 20.7% of the corpus against a true non-English share near 16%, because
terse but genuinely English tweets ("ur app shows expected delivery on 25th")
contain almost no function words. Requiring **positive evidence of another
language** (function words from the four languages actually present) brought it
to 15.3%, catching 30 of 32 known non-English exemplars while keeping 64 of 64
English ones. It errs toward keeping, which is the right direction: a stray
foreign message in the corpus is a much smaller problem than silently throwing
away English traffic.

Multilingual support was already out of scope, so this traffic gets removed
rather than served badly. Corpus: 10,026 -> 8,496 threads.

**27. The cluster namer had to be shown the labels it had already used.**
Left alone it named 8 of 12 clusters `delivery_status`. The clusters were
genuinely different (late delivery, missing package, delivered-to-wrong-address,
and disputes about the Prime next-day promise are distinct problems with
distinct resolutions) but the model defaulted to the most obvious topic every
time. Passing the already-assigned labels and requiring a distinct one forces it
to articulate what separates each cluster. Without that the taxonomy would have
thrown away most of the structure the clustering actually found, and the
classification task would have looked far easier than it is.

**28. Generation moved to a hosted key when the hardware #22 assumed turned out not to exist, then moved back when it did.**
Decision #22 moved everything local to escape the Gemini daily cap, and it
assumed a discrete GPU ("fits a 6GB GPU"). The machine the first pass got
finished on had Intel integrated graphics and 8 CPU cores, where a 4B model runs
at roughly 60–90s per call against a ~1000-call evaluation: 12–24 hours per full
run, and every prompt change costs another overnight. That isn't a backend, it's
a bottleneck, so generation moved to a hosted key.

The project then moved to a machine with a **GTX 1660 Ti (6GB)**, which is the
hardware #22 originally assumed. A 4B model at q4 fits entirely in that VRAM
alongside the 8K context and answers in seconds, so the reason for going hosted
is simply gone and #22's argument applies again as written. Drafting is local
again (`GROUNDSCORE_PROVIDER=ollama`, `qwen3:4b`), and only the judge is hosted,
because that's the one role where a *different vendor* is worth spending quota
on.

What that recovers, relative to the hosted-drafter version:

  * **Keyless regeneration**, not just keyless replay. Only the judge needs a
    key now, and it has a local fallback.
  * **The independent-family judge as a structural guarantee** (see #5) rather
    than a measured correction.
  * **Immunity to model deprecation** for the ~1000 drafting calls, which are
    pinned to a local model tag rather than a hosted id.

What it costs is capability: a 4B local drafter is weaker at instruction
following and JSON discipline than a hosted flash model, so reply quality is
lower than this architecture could reach. The report says that rather than
implying the ceiling is architectural. `GROUNDSCORE_ROLE_FAST=gemini` switches
drafting back for anyone with a paid key.

Embeddings never moved through any of this. The embedding cache is keyed on
`(model, dim, text)` and not on provider, so the committed `nomic-embed-text`
vectors stayed valid across every backend change. A miss raises instead of
silently re-embedding into a second vector space, which would have quietly
changed what "similar" means half way through an index.

**29. Hosted-model quirks are handled in the adapter, not worked around at the call sites.**
Two of them are load-bearing, and I found both by probing the API rather than by
reading docs. `gemma-4-31b-it` returns **500 INTERNAL** on every attempt when
`response_schema` is sent without a `system_instruction`, and returns valid JSON
as soon as any system instruction is present, so `_call_gemini` injects a
minimal one when the caller didn't supply one. It's injected in the adapter
specifically so the **cache key stays the system prompt the caller wrote**: the
same logical call has to hash identically whichever model happens to serve it,
or reordering a chain silently orphans the committed cache.

The same model also returns 503 "high demand" under ordinary load, which is why
it isn't the default judge despite having 700x the daily budget of the
alternatives. A judge that intermittently fails mid-split is a judge that
produces a split scored by two models.

Temperature is pinned to 0.0 everywhere it's accepted, but it never carried
reproducibility here. The committed cache does, which is why I built that first.

**30. The learned baselines are scored out-of-fold on dev, because the obvious way to score them was in-sample.**
`build_systems()` fits the baselines on dev. Scoring `--split dev` then handed
`simple_tfidf_nn` its own training rows and it returned **1.000 intent
accuracy**, which isn't a strong baseline, it's a memorisation check.
Out-of-fold (5-fold) it scores **0.386** on the same rows. The gap is the whole
point: the in-sample version would have made the agent look hopeless against a
baseline that had simply looked up the answer, and the natural response to that
table would have been to "fix" an agent that wasn't broken. The test split was
never affected (fitted on dev, scored on test, which is correct), but dev is the
table you iterate against, so a wrong number there steers every decision after
it.
`tests/test_pipeline.py::test_fitted_baselines_are_scored_out_of_fold_on_their_fit_split`
asserts that no fold predicts a row it trained on.

**31. Human reply scoring is sequenced before the judge runs, and the tooling enforces it.**
The documented order was `tune -> eval -> judge-human`, which can't work.
`make eval` runs the judge, and `tools/score_replies_cli.py` then refuses to
start, because scoring replies the judge has already graded is exactly the
anchoring the validation exists to rule out. But replies have to exist before a
human can score them, so "just score first" isn't available either. Resolved
with a `make replies` target that generates dev replies with `--no-judge`, so
the real order is **replies -> judge-human -> eval -> judge-agreement**. The
guard was already correct; the documentation was wrong, which is the more
dangerous of the two, because a guard that fires gets noticed and a wrong runbook
just gets followed.

**32. The provider is a property of the role, not of the run, with an ordered fallback chain per role.**
`GROUNDSCORE_PROVIDER` used to be global: one backend served drafting, judging
and the self-preference probe. That makes the single most important property of
the judge (that it doesn't share the drafter's lineage) an accident of which key
happened to be set. Roles now carry their own chains (`GROUNDSCORE_ROLE_JUDGE`,
default `gemini,ollama`, against a drafter defaulting to `ollama`), so
cross-vendor judging is the default rather than something to remember.

The chain also solves the quota problem #22 ran into without pretending it
doesn't exist. Three properties make the fallback safe to trust:

  * **Replay tries every provider in the chain before declaring a miss.** A
    cache built when the judge ran on Gemini still replays after the chain is
    reordered, so `make reproduce` doesn't start demanding live calls for
    numbers that are already published.
  * **A role is pinned to whichever provider serves it first.** Without pinning
    a chain flaps back to the preferred provider on every call, and a split ends
    up scored by two models in an interleaved pattern nobody can reconstruct.
  * **A forced switch is loud and recorded.** It warns on stderr, lands in
    `results/eval_*.json` under `providers.switches`, and every judged row is
    tagged with the model that actually scored it. A split scored by two judges
    is then *visible* instead of averaged into one number, which is the only
    honest option, because those rows can't be pooled.

The remaining exposure is stated rather than engineered away, and #33 narrows
it: the judge now points at the one free-tier class whose daily budget actually
covers a split. It's still one split per day. The chain makes an overrun a
visible operational fact instead of a silent change of judge half way down the
table.

**33. Hosted model ids are chosen on daily budget, and pacing is per model rather than per run.**
The free tier's binding constraint is requests-per-day-per-model, and across the
ids this key can reach it varies by 700x: `gemini-3.x-flash` allows 20/day,
`gemini-3.x-flash-lite` 500/day, `gemma-4-31b-it` 14,400/day. A judged split is
~450 calls. That single fact picked the models: both hosted roles sit on
`*-flash-lite`, the smallest class that covers a split, and I declined the
higher budget of `gemma-4-31b-it` for the reliability reasons in #29.

Two consequences are implemented rather than left to the server:

  * **Pacing is per model.** RPM ranges from 5 to 30 across these ids. One
    global rate either throttles the fast models to a fraction of their
    allowance or drives the slow ones straight into 429s, and backoff costs more
    wall clock than pacing does up front. The table lives in `llm.GEMINI_LIMITS`
    and is used at 80% of the documented rate, because the ceiling is enforced
    on the server's clock and sitting exactly on it produces 429s from skew.
  * **The daily cap is refused locally.** Once a model has served its documented
    RPD in this process, `_throttle` raises `DailyQuotaExhausted` so the chain
    falls through to Ollama on the next call. Letting the server enforce it
    instead costs a full retry ladder per row for the remainder of the run. The
    counter is in-process and says so: it stops one run burning a day's budget,
    it doesn't know what an earlier run spent.

Live call counts per model land in `results/eval_*.json` under
`providers.live_calls`, so a reader can see exactly how much hosted quota
produced a given table.

**34. Anthropic support was removed rather than left in as a third backend.**
It was the shipped backend for one revision (#28) and became dead weight when
drafting moved back to local: no key for it, no rows in the committed cache
keyed to it, and no role pointing at it. Keeping it would have meant three
provider branches, three credential lookups and a `_strict_schema` adapter that
nothing exercised. Code that can't be wrong because it never runs, until someone
sets the env var and finds out. `KNOWN_PROVIDERS` is now `("ollama", "gemini")`
and a chain naming anything else raises at startup, which
`tests/test_pipeline.py` asserts. Nothing in the committed cache was
invalidated: it holds `qwen3:4b` and Gemini rows only.

**35. The brand's own historical reply is scored by the same judge, as a reference row.**
Everything else here is self-referential: a rubric I wrote, a judge I picked, an
agent I built. "3.4 out of 5" carries no information on its own. So
`eval/reference_replies.py` puts the reply @AmazonHelp actually sent through the
same blind judge, on the same message, against the same retrieved precedent. It
needs no golden labels, which is why it can run before adjudication.

Three things stop being arguments and become measurements. Reply quality gets a
human anchor on the same rows. The rubric's deflection rule (#15) gets tested
against the brand's real replies rather than defended in prose. And the
`would_send` gate gets compared against the brand's own publishing bar: the
share of replies this brand really posted that our gate would have blocked says
whether the gate is calibrated to this queue or merely strict.

It is not a fair fight and the output says so in a `caveats` field rather than
leaving a reader to work it out. The human had the account, the order and the
tracking page; the agent had five old tweets. Groundedness especially is not
comparable, because the human's claims rest on systems the judge cannot see and
the rubric scores those as unsupported — that dimension is reported and then
excluded from the headline delta, which uses resolution, tone fit and safety.

**36. A provider fallback on the FIRST call is reported, not just a mid-run switch.**
#32 records forced switches, and that turned out to cover only half the failure.
A switch needs a previous provider. When the very first judge call fails over —
a dead key, Ollama not running — there is no previous, so nothing was recorded
and the results file read as though the preferred provider had served all along:
`judge: ollama:gemma3:4b`, no mention that it asked for Gemini and got a 401.
That is exactly the substitution this project cannot afford to make quietly, so
`provider_report()` now carries `degraded_starts` alongside `switches`, and the
run prints the reason when it happens. Found by running #35 with an expired
key and noticing the results file looked clean.

A cache hit is not a degraded start, which is the second bug in the same ten
lines: the replay path calls the same bookkeeping with "served from cache" as
its note, and the first version flagged every cached run as degraded. Both are
asserted in `tests/test_pipeline.py`.

**37. The judge sits on the model id that answers, not the one the plan named.**
The judge was `gemini-3.1-flash-lite`: documented, listed by `models.list()`,
right daily budget. It also returns **503 UNAVAILABLE on every call** from this
project — 0 of 8 attempts, with and without `response_schema`, over several
minutes. `gemini-3.5-flash-lite` answered 6 of 6 in the same window. So the two
lite ids swapped roles: 3.5 judges, 3.1 becomes the hosted drafter id, which is
the role that doesn't run by default anyway.

Availability is not existence, and nothing in the docs or the model list says
which ids will actually serve you. That is the argument for the chain in #32
stated as a fact rather than a precaution: without it this run would simply have
produced no judged rows. With it, the run produced judged rows scored by a
*different model than the config named*, which is worse, and is why #36 now
records a degraded start. The three pieces only work together.

Also worth knowing for anyone re-running this: Google no longer publishes the
per-model RPM/RPD table on the rate-limits page. It points at the AI Studio
dashboard instead, so the numbers in `llm.GEMINI_LIMITS` are probed against one
key on one day and should be treated as such.

---

## Borrowed / cited

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  (Kaggle, `thoughtvector`), used under its Kaggle licence.
- `scikit-learn` for TF-IDF, KMeans, silhouette, logistic regression and the
  classification metrics; `scipy` for Spearman.
- `google-genai` SDK for the hosted Gemini judge; Ollama's HTTP API (no SDK) for
  the local drafting and embedding paths.
- Percentile bootstrap and quadratic-weighted κ are standard methods. The
  implementations here are written against `sklearn`/`numpy` primitives rather
  than copied.
- The judge rubric structure (anchored 1–5 dimensions plus a separate binary
  ship/no-ship gate) follows the common LLM-as-judge pattern. The dimensions,
  the anchors and the deflection rule are written for this brand and this task.
