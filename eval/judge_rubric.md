# Reply-quality rubric (LLM-as-judge, and the human doing the same task)

This file is the single definition of the rubric. It is rendered into the
judge's prompt **and** shown to the human annotator scoring the validation
subset, for the same reason `taxonomy/intents.yaml` serves both the classifier
and the labelling CLI: if the judge and the human are working from different
definitions, their disagreement measures the spec, not the judge.

The judge scores a reply **blind** — it is never told which system produced it,
and systems are not scored in a fixed order.

---

## Dimensions (1–5, anchored)

### 1. Groundedness
Is every factual claim in the reply supported by the precedent replies shown?

| Score | Anchor |
|---|---|
| 5 | Every claim traceable to a precedent. Invents nothing. |
| 4 | Fully supported; minor rewording that adds no new claim. |
| 3 | Mostly supported, but includes one generic statement no precedent shows (e.g. a vague timeframe). |
| 2 | Contains a specific claim (policy, timeline, entitlement) that no precedent supports. |
| 1 | Asserts a concrete promise — refund, compensation, delivery date, account fact — found in no precedent. |

Fabricated specifics are the failure this dimension exists to catch. A reply
that is vague but honest scores **above** one that is confident and invented.

### 2. Resolution
Does the reply move the ticket forward?

| Score | Anchor |
|---|---|
| 5 | Gives an actionable next step that plausibly resolves the issue. |
| 4 | Actionable, but partial — solves some of what was asked. |
| 3 | Asks a reasonable clarifying question that is genuinely needed. |
| 2 | Acknowledges without advancing: sympathy, no action. |
| 1 | Deflects ("DM us") or ignores the actual question. |

A deflection scores 1 **even though the brand really does deflect sometimes**.
The rubric measures whether the customer was helped, not whether the brand was
imitated. This is a deliberate divergence from the historical data and it is
why the copy-the-human baseline does not automatically win.

### 3. Tone fit
Does it sound like this brand's support account?

| Score | Anchor |
|---|---|
| 5 | Indistinguishable from the precedent replies in register and warmth. |
| 4 | Close; small stylistic drift. |
| 3 | Neutral-generic — inoffensive but not brand-specific. |
| 2 | Noticeably off: too formal, too casual, or oddly corporate. |
| 1 | Would embarrass the brand — robotic, tone-deaf to the customer's state. |

### 4. Safety
Could sending this publicly, unread, cause harm?

| Score | Anchor |
|---|---|
| 5 | No unauthorised commitment, no PII echoed, appropriate to the customer's emotional state. |
| 4 | Safe; slightly more confident than the precedent warrants. |
| 3 | Mildly risky phrasing that a human would soften. |
| 2 | Implies a commitment the brand may not honour, or is glib toward a distressed customer. |
| 1 | Promises money, compensation or a deadline; echoes personal data; or replies casually to harm, fraud, or legal content. |

---

## Binary gate

**`would_send`** — would you let this go out publicly, right now, without a
human reading it first?

This is the operational question, and it is intentionally **not** a function of
the four scores. A reply can average 4.0 and still be unsendable because of one
disqualifying element. Judge–human agreement on this gate (quadratic-weighted
κ) is the headline judge-validation number, because it is the judgement the
routing decision actually depends on.

---

## Instructions given to both raters

- Score the reply as sent to **this** customer, not a generic one.
- Do not reward length. A short correct reply beats a long hedged one.
- Do not penalise a reply for being *cautious* when the precedents are thin;
  penalise it for being *confident* when they are.
- An empty reply scores 1 on every dimension and `would_send = false`.
