PY ?= python
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: help setup reproduce full test clean data corpus embeddings intents golden label \
        relabel second-annotator label-agreement tune replies eval eval-test \
        judge-human judge-agreement failures ui reference

help:
	@echo "make setup       install dependencies"
	@echo "make reproduce   regenerate headline results from committed caches (NO API key needed)"
	@echo "make test        run the test suite"
	@echo "make ui          browser UI for one message at a time (needs Ollama)"
	@echo ""
	@echo "Full rebuild (needs Ollama running, plus GEMINI_API_KEY for the judge):"
	@echo "  make full        data -> corpus -> embeddings -> intents"
	@echo "  make golden      sample 150 candidates with weak labels"
	@echo "  make label       adjudicate them by hand    (interactive)"
	@echo "  make relabel     blind re-label of 50 BY YOU, for intra-annotator kappa"
	@echo "  make second-annotator  blind re-label of the same 50 by a DIFFERENT model"
	@echo "  make label-agreement   kappa between the two passes"
	@echo "  make tune        fit routing thresholds on dev"
	@echo "  make replies     generate dev replies, WITHOUT judging them"
	@echo "  make judge-human blind-score those replies    (interactive, must precede eval)"
	@echo "  make eval        score every system on dev, judge included"
	@echo "  make judge-agreement   judge-vs-reference validation"
	@echo "  make failures    rank failure modes by frequency, with examples"
	@echo "  make reference   judge the brand's OWN replies on the same rows (no labels needed)"
	@echo "  make eval-test   score the test split ONCE"

setup:
	$(PY) -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# The headline path. Runs entirely from data/processed/ and cache/, both
# committed, so a reviewer needs no Kaggle account and no API key.
# ---------------------------------------------------------------------------
reproduce:
	$(PY) scripts/reproduce.py

test:
	$(PY) -m pytest tests/ -q

# Interactive single-message view of the pipeline. Stdlib http.server, so it
# adds no dependency; drafting still needs a live provider.
ui:
	$(PY) tools/ui.py

# ---------------------------------------------------------------------------
# Full rebuild from the raw dataset.
# ---------------------------------------------------------------------------
full: data corpus embeddings intents

data:
	$(PY) scripts/download_data.py

corpus:
	$(PY) scripts/build_dataset.py

embeddings:
	$(PY) -u scripts/build_embeddings.py

intents:
	$(PY) -m groundscore.discover_intents

golden:
	$(PY) scripts/sample_golden.py

label:
	$(PY) tools/label_cli.py

relabel:
	$(PY) tools/label_cli.py --relabel

# The other half of label quality. --relabel above measures the annotator
# against themselves; this measures them against a different annotator, which
# is the number the report actually needs. Runs on a local model by default so
# it does not spend the judge's daily hosted quota.
second-annotator:
	$(PY) -u tools/second_annotator.py --model gemma3:4b

label-agreement:
	$(PY) eval/label_agreement.py

tune:
	$(PY) eval/tune_thresholds.py

# Replies WITHOUT the judge. This target exists because of an ordering
# constraint: human reply scores have to be collected before the judge has
# produced an opinion of the same replies, or the human is anchored and the
# agreement number is worthless. `make eval` judges, so it cannot come first --
# but the replies must exist before a human can score them. Hence: replies ->
# judge-human -> eval. tools/score_replies_cli.py enforces this and will refuse
# to run in the wrong order.
replies:
	$(PY) eval/run_eval.py --split dev --no-judge

eval:
	$(PY) eval/run_eval.py --split dev

eval-test:
	$(PY) eval/run_eval.py --split test --final

judge-human:
	$(PY) tools/score_replies_cli.py --split dev

judge-agreement:
	$(PY) eval/judge_agreement.py --split dev

# The report has to name its top failure modes. Counting them beats picking
# them, which is how the interesting ones crowd out the common ones.
failures:
	$(PY) eval/failure_analysis.py --split dev --all-systems

# The brand's own historical reply, scored by the same blind judge on the same
# rows. Needs no golden labels, so it can run before adjudication -- it is the
# only reply-quality reference point in the project that is not self-referential.
reference:
	$(PY) eval/reference_replies.py --split dev

clean:
	rm -rf results/*.json results/*.md results/*.jsonl
	@echo "caches and labels left intact (delete cache/ by hand to force API calls)"
