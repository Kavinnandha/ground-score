PY ?= python
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: help setup reproduce full test clean data corpus embeddings intents golden label \
        relabel tune eval eval-test judge-human judge-agreement report-inputs

help:
	@echo "make setup       install dependencies"
	@echo "make reproduce   regenerate headline results from committed caches (NO API key needed)"
	@echo "make test        run the test suite"
	@echo ""
	@echo "Full rebuild (needs GEMINI_API_KEY, several hours of rate-limited calls):"
	@echo "  make full        data -> corpus -> embeddings -> intents"
	@echo "  make golden      sample 200 candidates with weak labels"
	@echo "  make label       adjudicate them by hand    (interactive)"
	@echo "  make tune        fit routing thresholds on dev"
	@echo "  make eval        score every system on dev"
	@echo "  make judge-human blind-score replies         (interactive)"
	@echo "  make judge-agreement   judge-vs-human validation"
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

tune:
	$(PY) eval/tune_thresholds.py

eval:
	$(PY) eval/run_eval.py --split dev

eval-test:
	$(PY) eval/run_eval.py --split test --final

judge-human:
	$(PY) tools/score_replies_cli.py --split dev

judge-agreement:
	$(PY) eval/judge_agreement.py --split dev

clean:
	rm -rf results/*.json results/*.md results/*.jsonl
	@echo "caches and labels left intact (delete cache/ by hand to force API calls)"
