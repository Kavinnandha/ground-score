# Decision log

The 12 non-obvious decisions that shaped the submission are below. Supporting
implementation details and experiment history remain in the repository's code,
results, and report.

## 1. Choose the brand by measurement

I profiled candidate brands by handoff rate, reply substance, intent diversity,
and multi-turn behaviour. I chose `AmazonHelp` rather than my initial guess,
`SpotifyCares`, because its replies contain more usable resolution patterns and
less routine deflection.

## 2. Correct the handoff metric before choosing the brand

The first profiler counted only DM-style deflection and badly understated
AmazonHelp's handoff rate. I added contact-page language such as “reach out to
us here”, checked the result against raw replies, and recorded the correction in
the report.

## 3. Hold the evaluation pool out of retrieval

Golden-set threads are assigned by a stable hash before labelling and are
excluded from the retrieval index. This prevents the system from retrieving the
exact conversation it is being evaluated on.

## 4. Sample the golden set without using predicted intents

The 150 examples use traffic-proportional, rare-intent, and adversarial strata
based on unsupervised clusters and surface heuristics. Sampling on classifier
predictions would hide the classes the classifier fails to discover.

## 5. Manually adjudicate the final labels

All 150 final intent, routing, and note fields were decided manually by the
project author against `taxonomy/intents.yaml`. Model outputs were weak-label
proposals only; the separate model pass is a sensitivity check, not the source
of the submitted gold labels.

## 6. Use one taxonomy for classification and annotation

`taxonomy/intents.yaml` is both the classifier's intent specification and the
labelling guideline. This avoids silently measuring prompt/specification drift
as if it were model error.

## 7. Route with deterministic safety rules before the LLM

Account compromise, safety/legal concerns, PII, unsupported drafts, and other
hard cases escalate before confidence thresholds are considered. Each such
decision carries an attributable rule name for auditability.

## 8. Report routing as coverage and false-auto rate

Auto-handling and escalation have very different costs, so one accuracy number
would conceal the important failure. The report therefore emphasizes coverage,
false-auto rate, escalation recall, and the coverage-versus-harm curve.

## 9. Require drafts to cite their precedents

The drafter must return `grounded_in` exemplar identifiers. An empty citation
list escalates; the citation is an auditable signal, while the judge separately
checks whether the text actually agrees with its evidence.

## 10. Score learned baselines out-of-fold on their fit split

The simple baseline is evaluated out-of-fold on dev rather than on the rows it
memorised. This prevents a nearest-neighbour lookup table from receiving an
artificial 1.00 score merely because it saw the evaluation rows during fitting.

## 11. Collect human reference scores before running the LLM judge

The blind human reference scores were written before judge outputs existed, and
the scoring tool refuses to run after judging unless explicitly overridden.
This keeps the judge-agreement result from being contaminated by anchoring.

## 12. Keep provider choice role-specific and expose fallback

The drafter and judge use separate role chains so the judge does not silently
grade its own generation lineage. If a provider fails, the serving model,
switch, and affected rows are recorded instead of pooling unlike judges under a
single opaque number.

## Borrowed / cited

- Customer Support on Twitter dataset: `thoughtvector/customer-support-on-twitter`
  (Kaggle, Thought Vector). A cleaned 8,496-thread subsample is committed.
- Models, used as-is with no fine-tuning: `qwen3:4b` (Alibaba Qwen team) for weak labels,
  classification and drafting; `gemma3:4b` (Google) for the blind second pass and judge
  fallback; `gemini-3.5-flash-lite` (Google, Gemini API) as the reply judge;
  `nomic-embed-text` (Nomic AI) for embeddings. Local models are served by Ollama.
- Libraries: scikit-learn (TF-IDF, logistic regression, KMeans, `cohen_kappa_score`,
  F1), SciPy (Spearman ρ), NumPy, pandas, PyYAML, rich, pytest, `google-genai`.
- Methods: percentile bootstrap confidence intervals (Efron & Tibshirani, 1993);
  Cohen's κ (Cohen, 1960) and quadratic-weighted κ (Cohen, 1968) for agreement;
  LLM-as-judge with a validated rubric and self-preference / verbosity probes, following
  the concerns raised in Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena" (NeurIPS 2023); retrieval-grounded few-shot prompting (dynamic
  exemplar selection by embedding similarity).
- The assignment permits AI coding assistants; the implementation uses them,
  but the final labels, report, and decision log are the author's own work.
