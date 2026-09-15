# ground-score — an AI support agent for @AmazonHelp, and how much to trust it

> **On the numbers in here.** Everything in the Results, Failure analysis and
> Judge-validation sections comes out of `make reproduce` into `results/`.
> Sections marked _(pending run)_ get filled from that output. The framing,
> method and limitations sections don't depend on it and are finished.

---

## 1. Problem framing

### What "good" means for this brand

@AmazonHelp is a public, high-volume, logistics-heavy support queue. Three
things about it shape what a useful agent looks like here.

First, most inbound is a status question, not a problem to solve. "Where is my
parcel", "it says delivered and it isn't here", "I was charged twice". The
brand's own reply is usually a short apology plus one instruction or link. An
agent that produces exactly that isn't being lazy, it's matching the job.

Second, the brand rarely hides behind DMs, but it does hand off. 10.8% of first
replies send the customer somewhere else (0.8% to DMs, 9.9% to a contact page).
Low enough that "draft a grounded reply" is a real task here, unlike
Uber_Support at 75%, but high enough that a system which learns to imitate this
brand will learn to hand off about one time in ten.

Third, the expensive mistake is public and permanent. A needless escalation
costs an agent thirty seconds. A wrong auto-reply is a public tweet from a brand
account, and the cases where it goes wrong (fraud, a damaged item, someone
who has already been let down twice) are exactly the cases where the customer
is least able to absorb it.

So "good" here means:

1. **Correct routing matters far more than correct replies.** The value of the
   system is in knowing what it shouldn't answer.
2. **Grounded beats fluent.** A vague-but-true reply is better than a confident
   invented one, and the rubric enforces that: fabricated specifics score below
   honest hedging.
3. **Auditable.** A support lead has to be able to ask "why did this escalate?"
   and get a rule name back, not a vibe.
4. **Coverage is the business case, harm rate is the constraint.** So the
   headline is an operating point, *X% auto-handled at ≤Y% false-auto*, not an
   accuracy figure.

### What I deliberately didn't build

- **Multi-turn dialogue.** The agent acts on the *first* customer message only.
  Later turns are kept for context, and for measuring whether the brand's answer
  ended the conversation, but conversational state is out of scope.
- **Live tool use.** No order lookup, no account access. This is the single
  biggest cap on the ceiling here: a large share of these tickets simply can't
  be answered without account data, and no amount of prompt work fixes that.
  Recognising those and escalating them is the correct behaviour, and it's what
  the system does.
- **PII redaction.** PII is *detected* and used as an escalation trigger. It
  isn't scrubbed or stored.
- **Fine-tuning.** Retrieval plus a taxonomy gets most of the value for a
  fraction of the complexity, and it stays inspectable.
- **Multilingual support.** Non-Latin-script messages are filtered out of the
  corpus. They exist in the queue; this system doesn't serve them.
- **Serving.** No API, no queue consumer, no latency budget. The deliverable is
  an evaluated system, not a deployed one.

---

## 2. Method

| Stage | Approach |
|---|---|
| Brand choice | Profiled 7 candidate brands on handoff rate, reply substance, intent diversity, multi-turn rate (`results/brand_profile.md`) |
| Corpus | 8,496 threads kept from an 11,000-thread sample (957 unusable, 1,530 non-English, 17 near-duplicates dropped), stable-hash split into 7,998 retrieval history / 498 evaluation pool |
| Taxonomy | Cluster → LLM names each cluster → human merges into `taxonomy/intents.yaml` |
| Golden set | 150 adjudicated examples in 3 strata, scored separately |
| Agent | retrieve(k=5) → classify → draft(≤280 chars, cites precedent) → route |
| Routing | Deterministic rules first, LLM only judges what survives them |
| Evaluation | Bootstrap CIs, coverage-vs-harm curve, LLM judge validated against blind human scores |
| Models | `qwen3:4b` (local) classifies + drafts, `gemini-3.5-flash-lite` judges, `nomic-embed-text` embeds from the committed cache |

The reasoning behind each of these is in [`DECISIONS.md`](DECISIONS.md).

### The brand decision, and a correction

| brand | threads | dm_deflection | link_handoff | **handoff** | substantive | multi_turn | lexical_div |
|---|---|---|---|---|---|---|---|
| hulu_support | 3976 | 0.006 | 0.052 | **0.057** | 0.927 | 0.383 | 0.124 |
| **AmazonHelp** | 3653 | 0.008 | 0.099 | **0.108** | 0.716 | 0.597 | **0.183** |
| SpotifyCares | 3937 | 0.374 | 0.003 | **0.377** | 0.555 | 0.355 | 0.135 |
| XboxSupport | 3921 | 0.287 | 0.002 | **0.289** | 0.547 | 0.520 | 0.119 |
| Delta | 3950 | 0.200 | 0.001 | **0.201** | 0.409 | 0.354 | 0.157 |
| AppleSupport | 3949 | 0.536 | 0.035 | **0.551** | 0.394 | 0.350 | 0.130 |
| Uber_Support | 3953 | 0.648 | 0.119 | **0.750** | 0.170 | 0.357 | 0.126 |

My guess going in was SpotifyCares. The data rejected it at 37.7% handoff.

The first version of this table was wrong. It only measured DM-style deflection
and put AmazonHelp at 0.008. Reading actual retrieved replies showed the real
pattern: AmazonHelp says *"please reach out to us here: `<URL>`"*, not "DM us".
Fixing the metric moved it to 0.108, a 13× undercount. The ranking survived, so
the decision stands, but it stood on a number that was wrong until somebody
checked it against raw text. A bare URL deliberately doesn't count as a handoff:
46% of replies contain one and most of those are genuinely useful.

AmazonHelp over hulu_support is the one call here that's judgement rather than
measurement. hulu wins on handoff rate and on substance, but its lexical
diversity (0.124) and multi-turn rate (0.383) point at an intent space that
collapses to about four classes, which would have made the classification
result uninformative.

---

## 3. Results

Two parts. The baseline comparison needs the golden labels and is still pending.
The reference comparison needs no labels at all, and it is the part that changed
what I believe.

### 3.1 The brand's own reply, scored by the same judge

For all 70 dev rows I put @AmazonHelp's actual reply through the same blind
judge, on the same message, against the same retrieved precedent as the agent's
draft (`eval/reference_replies.py`). One model scored all 140 calls
(`gemini-3.5-flash-lite`); `providers.switches` and `degraded_starts` are both
empty, so this is one judge, not two averaged.

| dimension | agent | brand's own reply | delta | comparable? |
|---|---:|---:|---:|---|
| groundedness | 4.39 | 3.64 | +0.74 | no — see below |
| resolution | 4.23 | 3.53 | +0.70 | yes |
| tone fit | 4.76 | 3.94 | +0.81 | yes |
| safety | 4.53 | 3.90 | +0.63 | yes |
| **comparable mean** | **4.50** | **3.79** | **+0.72** | |
| would_send | 0.87 | 0.54 | +0.33 | |

Head to head on the comparable dimensions: agent better on 44 rows, the human
better on 13, tied on 13. Our `would_send` gate would have blocked **32 of the
70 replies this brand actually posted in public (46%)**.

Read literally, a 4B model running on my desktop writes better support replies
than the people who answered these tickets. I don't believe that for a second,
and working out why is worth more than the table.

Groundedness is excluded from the comparable mean for a structural reason: the
human's claims are backed by the order, the account and the tracking page, none
of which the judge can see, so the rubric scores them as unsupported. But the
other three dimensions turned out to be contaminated too, in three ways I could
measure:

**The judge marks down replies it was handed incomplete.** Seven of the 70
historical replies are explicitly part one of two — they end "(1/2)" — because
the brand splits long answers across tweets. Those score **3.00** on the
comparable mean against **3.86** for the rest. The judge is reading half a reply
and scoring it as a whole one.

**It punishes correct support behaviour it cannot verify.** Four replies tell a
customer to stop posting tracking details on a public timeline (threads 288010,
2316996, 420919, 1219160). They average **2.58**. On one, the judge wrote that
the reply "invents an ungrounded policy not found in any precedents" and "would
embarrass the brand". My own `route.py` hard-escalates any message containing
public PII for exactly the reason that human agent gave. The system's safety
rule and the system's judge disagree about the same behaviour, and only one of
them is right.

**It rewards problem-shaped boilerplate.** Thread 2750418: the customer tweets a
photo of a Prime Now delivery that arrived at their hotel, pleased. The human
replied in kind. The agent replied *"I'm sorry you haven't received your Prime
delivery to Orlando"* — inventing a complaint that isn't there — and the judge
scored the agent **3.33 higher**. Two more of the same shape (2687680, 2619718).
That is an agent failure the judge recorded as a win.

So the honest headline of this table is not that the agent is better. It is that
an LLM judge with no account access, no thread context and no world knowledge
systematically prefers grounded-sounding boilerplate to real support work. Every
reply-quality number in this report inherits that bias, which is the strongest
argument I have for why the judge-vs-human agreement statistic — not the rubric's
apparent rigour — is what licenses any claim about reply quality.

One more thing this measured that the report would otherwise have had to argue.
The rubric says a handoff scores 1 on resolution (DECISIONS #15). Six of the 70
historical replies are handoffs, and the judge gave them **4.00** on resolution
against 3.48 for direct replies. Small sample, opposite sign. The judge is not
applying the rule the rubric states, which is a second reason not to read the
agent-versus-human gap at face value.

A note on two numbers that look inconsistent: the judge would send 87% of the
agent's replies, but the router only auto-handles 31 of 70. They are different
questions. The judge scores the text; the router also escalates on intent class
(18 rows hit `never_auto_intent`), on thin precedent (13 hit
`low_retrieval_similarity`) and on hard rules (4 safety, 3 payment-card PII, 1
account compromise). The gate is deliberately not a function of reply quality.

### 3.2 Against the baselines

_(pending run — `results/eval_test.md`)_

Systems compared: `trivial_always_auto`, `trivial_always_escalate`,
`simple_tfidf_nn`, `agent_no_retrieval` (ablation), `agent`.

The ablation matters most for reading the headline. It is the same LLM pipeline
with retrieval switched off, so the gap between it and `agent` is what grounding
actually buys, separated from what the language model would have done anyway.

---

## 4. Failure analysis

The full top five needs the golden labels, since ranking failure modes by
frequency means knowing which rows are wrong. Three are already visible in the
reference run and are here with real thread ids.

**1. The agent assumes a complaint when there isn't one.** Threads 2750418,
2687680, 2619718. Retrieval returns the five nearest past messages, and on this
queue almost every past message is a complaint, so a neutral or positive tweet
gets grounded in complaint precedent and comes back as an apology for something
that never happened. Hypothesis: it is a retrieval problem, not a drafting one —
the drafter is doing what it was told with the evidence it was given. A
similarity floor tuned for "is there precedent" does not catch "is this
precedent the same *kind* of message".

**2. Thin precedent drives a third of all escalations.** 13 of 70 rows escalate
on `low_retrieval_similarity` and 18 on `never_auto_intent`. The first is the
system working as designed. It is also the ceiling on coverage, and no prompt
change moves it — only a better index or more history would.

**3. The judge's blind spots become the agent's scoreboard.** Covered in 3.1:
fragments penalised 0.86, PII enforcement scored 2.58, invented complaints
rewarded. Listed here because it is a failure of this evaluation, not of the
brand, and it is the one I would fix first.

_(remaining modes pending the labelled set)_

---

## 5. What is misleading about my headline number?

The brief makes this section mandatory, and it's the one I'd read first.

**1. There's one annotator, so the agreement figure is intra-annotator.**
The κ reported for label quality comes from re-labelling 50 examples blind, at
least a day later. That measures self-consistency, so it's a *ceiling* on label
quality, not inter-annotator agreement. A second annotator would almost
certainly agree less. Every metric built on these labels has more slack in it
than its confidence interval suggests, because the CI covers sampling noise and
not label noise.

**2. The golden set was pre-labelled by the system it evaluates.**
Weak labels came from the classifier and were adjudicated by hand. The override
rate is published for exactly this reason: the lower it is, the more the golden
set is measuring agreement-with-the-model rather than ground truth. It's an
anchoring bias and no amount of care removes it completely.

**2b. The language filter leaks, and I know by how much.**
10 of 8,496 corpus threads (0.12%) are non-English messages that got through the
filter because Latin brand names dominate the token count
(`amazon echo 楽しいー`). One of them is in the golden set. I left it: fixing it
means rebuilding the corpus, the clustering, the taxonomy and the weak labels to
move 0.12% of the data.

**3. n = 80 on the test split, so the intervals are wide.**
The golden set is 150 rather than 250 because the compute budget was a local 4B
model, not because 150 was enough. That splits 70 dev / 80 test. Differences
under roughly 10 points between systems aren't distinguishable from noise at
this sample size, so any ranking that depends on a small gap should be read as
"not established" rather than "smaller".

**4. The headline is one stratum, and the pooled number isn't production performance.**
The golden set deliberately over-samples rare intents (25%) and hard cases
(15%). Pooling all of them understates real-traffic accuracy; reporting only the
proportional stratum hides the tail. Both are given, but any single number
quoted out of this report will mislead in one direction or the other.

**5. Historical replies are what the brand *did*, not what was *good*.**
Retrieval grounds the agent in precedent, including precedent where the brand
handed off, apologised without acting, or answered a different question. An
agent that imitated this queue perfectly would hand off about 10% of the time.
The rubric deliberately scores deflection as a resolution failure so we don't
reward that, which does mean the reply scores are measured against a standard
the brand's own historical replies don't always meet.

**6. The escalation labels are my judgement, not Amazon's policy.**
"Should a human handle this?" was decided by me against a written guideline. A
real support organisation has staffing, SLA and liability constraints that would
move that line. The routing metrics measure agreement with my policy, not
correctness against a real one.

**7. The judge is a different family from the drafter, but it's still a small model.**
Drafter is Qwen, judge is Gemini (falling back to Gemma): different vendors,
different weights, so same-model self-preference is mostly removed by
construction rather than merely measured. What's left is that a small judge is a
weak judge. Its agreement with a human is measured
(`results/judge_agreement.json`), and that number is what licenses any
reply-quality claim here, not the apparent rigour of the rubric.

**8. The hard-case slice was designed by the same person who built the agent.**
It uses model-independent surface heuristics precisely to limit that, but which
heuristics count as "hard" is still my choice.

**9. Thresholds are fitted on ~70 dev examples.**
`tau_confidence` and `tau_similarity` are point estimates from a small sample.
They won't transfer cleanly to a new period or a new brand, and the operating
point they define is more fragile than one reported coverage number implies.

**10. The drafter is a small cheap model, so absolute quality isn't the architecture's ceiling.**
Reply quality here reflects `qwen3:4b` running locally, which I chose because
~1000 calls per full run exceeds every free hosted budget available to me, not
because it writes the best reply this pipeline could produce. Comparisons
*between* systems are still fair, since every system uses the same generator.
The absolute reply scores aren't a statement about the design. Swapping in a
stronger drafter would move reply quality, and should, but it wouldn't validate
the routing, which is where the value is.

**10b. The confidence score isn't calibrated.**
Routing partly depends on a self-reported LLM confidence, which is known to
cluster high and behave more like a fluency signal than a probability. It's
used because having no confidence signal is worse, not because it's
trustworthy.

**11. The retrieval backend almost entirely determines what the agent sees.**
I ran two embedding backends over the same corpus and the same 120 queries.
They agree on the single most similar precedent **7.5%** of the time, and their
top-5 sets overlap **11.3%** (`results/retrieval_backend_comparison.json`). So
"grounded in how the brand historically resolved this" really means grounded in
whatever neighbours the encoder happens to surface, and a different encoder
would ground the same reply in almost entirely different evidence. Both backends
return plausible precedent when you look at them, which is what makes this easy
to miss: spot-checking retrieval and finding it "fine" doesn't establish that
it's stable. Every reply-quality number here is conditional on this one choice.

**12. One brand, one snapshot.**
All of this is AmazonHelp in late 2017. Nothing here establishes that the
taxonomy, the thresholds or the judge's behaviour transfer to another brand or
another year.

---

## 6. What I'd do with one more week

Ranked by how much they'd change what I currently believe:

1. **A second annotator on 100 examples.** Real inter-annotator κ would replace
   the weakest claim in this report with a measured one. Highest value by a
   distance.
2. **Human preference over pairwise comparisons.** Absolute 1–5 judge scores are
   the least reliable part of the reply evaluation. Pairwise A/B against the
   real historical reply would be more robust, and it answers "is this better
   than what the brand actually sent?" directly.
3. **Calibrate the confidence signal.** Fit a reliability curve on dev, then
   route on calibrated probability instead of raw self-report. The routing
   thresholds are currently sitting on an uncalibrated number.
4. **Transfer test on a second brand.** Re-run the whole pipeline on
   hulu_support without re-tuning anything. Whatever degrades is the part that
   was overfitted to AmazonHelp.
5. **Cost and latency accounting.** Three LLM calls per message is a real
   operating cost and nothing here measures it. A cheaper router (rules only for
   the easy majority) is probably most of the value.
6. **Retrieval quality ablation.** Vary k, compare embedding backends, and
   measure how much reply quality actually depends on retrieval depth.
