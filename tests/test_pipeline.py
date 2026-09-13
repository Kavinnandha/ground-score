"""Tests for the properties that would silently corrupt the results if broken.

Not a coverage exercise. Each test here guards a specific way the headline
numbers could become wrong without anything visibly failing:

  * leakage    the agent retrieving the thread it is being scored on
  * splits     dev/test assignment drifting between runs
  * caching    a "reproduced" run quietly making live API calls
  * routing    a hard escalation rule being bypassed by a threshold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundscore import cleaning, config, llm  # noqa: E402
from groundscore.classify import Classification  # noqa: E402
from groundscore.draft import Draft  # noqa: E402
from groundscore.ingest import read_jsonl  # noqa: E402
from groundscore import route  # noqa: E402


# ---------------------------------------------------------------------------
# Cleaning / hashing
# ---------------------------------------------------------------------------

def test_normalise_strips_handles_and_urls_but_keeps_affect():
    text = "@AmazonHelp my order NEVER arrived!!! see https://t.co/x @115712"
    out = cleaning.normalise(text)
    assert "@AmazonHelp" not in out and "@115712" not in out
    assert "<URL>" in out
    # Caps and repeated punctuation are routing signal and must survive.
    assert "NEVER" in out and "!!!" in out


def test_deflection_detection():
    assert cleaning.is_deflection("Please send us a DM and we'll help")
    assert cleaning.is_deflection("Shoot us a message here")
    assert not cleaning.is_deflection("Try resetting the device and check again")


def test_stable_bucket_is_process_independent():
    """Python's builtin hash() is salted per process; ours must not be."""
    assert cleaning.stable_bucket("thread-123", 100) == cleaning.stable_bucket("thread-123", 100)
    assert cleaning.stable_bucket("thread-123", 100) == 43  # pinned: a change here moves every split


def test_dedupe_key_collapses_cosmetic_variants():
    assert cleaning.dedupe_key("Spotify is DOWN!!!") == cleaning.dedupe_key("spotify is down")
    assert cleaning.dedupe_key("order late") != cleaning.dedupe_key("order lost")


# ---------------------------------------------------------------------------
# Corpus integrity
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def threads():
    if not config.THREADS_PATH.exists():
        pytest.skip("corpus not built; run scripts/build_dataset.py")
    return read_jsonl(config.THREADS_PATH)


def test_every_thread_has_a_brand_reply(threads):
    assert threads
    assert all(t["brand_replies"] for t in threads)


def test_splits_are_reproducible_from_thread_id(threads):
    """Recomputing the split must reproduce what is on disk, exactly."""
    buckets = config.brand_config()["split"]["golden_pool_buckets"]
    for thread in threads:
        expected = (config.SPLIT_GOLDEN_POOL
                    if cleaning.stable_bucket(f"split:{thread['thread_id']}", 100) < buckets
                    else config.SPLIT_HISTORY)
        assert thread["split"] == expected, f"split drift on thread {thread['thread_id']}"


def test_no_leakage_between_retrieval_index_and_golden_pool(threads):
    """The property the whole evaluation rests on."""
    history = {t["thread_id"] for t in threads if t["split"] == config.SPLIT_HISTORY}
    golden = {t["thread_id"] for t in threads if t["split"] == config.SPLIT_GOLDEN_POOL}
    assert history and golden
    assert history.isdisjoint(golden)


def test_golden_labels_reference_pool_threads_only():
    if not config.GOLDEN_PATH.exists():
        pytest.skip("golden set not labelled yet")
    if not config.THREADS_PATH.exists():
        pytest.skip("corpus not built")
    pool = {t["thread_id"] for t in read_jsonl(config.THREADS_PATH)
            if t["split"] == config.SPLIT_GOLDEN_POOL}
    with config.GOLDEN_PATH.open(encoding="utf-8") as fh:
        labelled = [json.loads(line) for line in fh if line.strip()]
    assert labelled
    assert all(row["thread_id"] in pool for row in labelled)


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

def test_cache_key_is_stable_and_input_sensitive():
    base = llm.cache_key("m", "prompt", None, 0.0, None)
    assert base == llm.cache_key("m", "prompt", None, 0.0, None)
    assert base != llm.cache_key("m", "prompt!", None, 0.0, None)
    assert base != llm.cache_key("m2", "prompt", None, 0.0, None)
    assert base != llm.cache_key("m", "prompt", None, 0.7, None)
    assert base != llm.cache_key("m", "prompt", None, 0.0, "system")


def test_offline_miss_raises_rather_than_calling_out(monkeypatch):
    """A reproduce run with no key must fail loudly, never silently diverge."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GROUNDSCORE_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "_cache_get", lambda key: None)
    with pytest.raises(llm.OfflineCacheMiss):
        llm.complete("a prompt that is definitely not cached", model="test-model")


def test_offline_flag_blocks_local_backend_too(monkeypatch):
    """The local backend needs no key, so stripping keys is not enough.

    Without GROUNDSCORE_OFFLINE a cache miss during `make reproduce` would be
    served by a live Ollama call and the published numbers would silently stop
    matching the committed cache.
    """
    monkeypatch.setenv("GROUNDSCORE_OFFLINE", "1")
    monkeypatch.setenv("GROUNDSCORE_PROVIDER", "ollama")
    monkeypatch.setattr(llm, "_cache_get", lambda key: None)
    with pytest.raises(llm.OfflineCacheMiss):
        llm.complete("another definitely-uncached prompt", model="qwen3:4b")


def test_cache_key_separates_providers():
    """A cache built locally must never be replayed as if it were hosted."""
    local = llm.cache_key("m", "p", None, 0.0, None, prov="ollama")
    hosted = llm.cache_key("m", "p", None, 0.0, None, prov="gemini")
    assert local != hosted


# ---------------------------------------------------------------------------
# Routing rules
# ---------------------------------------------------------------------------

# Routing consults taxonomy.never_auto_names(), so these tests need the merged
# taxonomy. They are skipped -- not failed -- before it exists.
requires_taxonomy = pytest.mark.skipif(
    not config.TAXONOMY_PATH.exists(),
    reason="taxonomy/intents.yaml not built yet; run discover_intents + human merge",
)


def _ok_classification(intent="delivery_status", confidence=0.99, similarity=0.99):
    from groundscore.retrieve import Exemplar

    return Classification(
        intent=intent, confidence=confidence, rationale="",
        neighbours=[Exemplar("t1", "msg", "reply", similarity, False)],
    )


def _ok_draft():
    return Draft(reply="We can help with that - could you confirm your order date?",
                 grounded_in=["E1"])


@requires_taxonomy
def test_hard_rules_beat_high_confidence():
    """A confident classifier must not be able to auto-send a fraud report."""
    decision = route.route("my account was hacked and someone ordered a laptop",
                           _ok_classification(), _ok_draft(),
                           tau_confidence=0.0, tau_similarity=0.0, use_llm=False)
    assert decision.action == route.ESCALATE
    assert decision.triggered_rule == "account_compromise"


@requires_taxonomy
def test_pii_forces_escalation():
    decision = route.route("my card 4111 1111 1111 1111 was charged twice",
                           _ok_classification(), _ok_draft(),
                           tau_confidence=0.0, tau_similarity=0.0, use_llm=False)
    assert decision.action == route.ESCALATE
    assert decision.triggered_rule.startswith("pii_")


@requires_taxonomy
def test_ungrounded_draft_escalates():
    draft = Draft(reply="You'll get a full refund within 24 hours.", grounded_in=[])
    decision = route.route("where is my order", _ok_classification(), draft,
                           tau_confidence=0.0, tau_similarity=0.0, use_llm=False)
    assert decision.action == route.ESCALATE
    assert decision.triggered_rule == "draft_ungrounded"


@requires_taxonomy
def test_low_similarity_escalates():
    decision = route.route("where is my order",
                           _ok_classification(similarity=0.10), _ok_draft(),
                           tau_confidence=0.0, tau_similarity=0.75, use_llm=False)
    assert decision.action == route.ESCALATE
    assert decision.triggered_rule == "low_retrieval_similarity"


@requires_taxonomy
def test_clean_case_can_auto_when_llm_router_disabled():
    decision = route.route("where is my order please", _ok_classification(), _ok_draft(),
                           tau_confidence=0.5, tau_similarity=0.5, use_llm=False)
    assert decision.action == route.AUTO


@requires_taxonomy
def test_forced_escalation_cannot_be_swept_away():
    """The coverage curve must never auto-send something a hard rule forbids."""
    assert route.forced_escalation("someone hacked my account", _ok_classification(), _ok_draft())
    assert not route.forced_escalation("where is my parcel", _ok_classification(), _ok_draft())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_false_auto_rate_is_share_of_what_we_sent():
    sys.path.insert(0, str(ROOT))
    from eval import metrics

    gold = ["escalate", "auto", "auto", "auto"]
    pred = ["auto", "auto", "auto", "auto"]          # auto'd everything
    result = metrics.score_routing(gold, pred, resamples=50)
    assert result.coverage["value"] == 1.0
    assert result.false_auto_rate["value"] == pytest.approx(0.25)

    pred_all_escalate = ["escalate"] * 4
    safe = metrics.score_routing(gold, pred_all_escalate, resamples=50)
    assert safe.coverage["value"] == 0.0
    assert safe.false_auto_rate["value"] == 0.0      # sends nothing, harms nobody


def test_coverage_curve_respects_forced_escalation():
    sys.path.insert(0, str(ROOT))
    from eval import metrics

    gold = ["escalate", "auto"]
    curve = metrics.coverage_curve(gold, [1.0, 1.0], forced_escalate=[True, False])
    at_zero = next(p for p in curve if p["threshold"] == 0.0)
    assert at_zero["n_auto"] == 1  # the forced one never auto-sends


# --------------------------------------------------------------------------
# Baseline honesty
# --------------------------------------------------------------------------

def test_fitted_baselines_are_scored_out_of_fold_on_their_fit_split():
    """A learned baseline must never be graded on its own training rows.

    `build_systems` fits on dev, so scoring `--split dev` naively hands the
    TF-IDF nearest-neighbour baseline its own training set and it returns 1.00
    intent accuracy. That made the agent look far worse than it is against a
    baseline that had simply memorised the answers. Guards the out-of-fold path.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval import run_eval

    assert "simple_tfidf_nn" in run_eval.FITTED_BASELINES
    assert run_eval.OOF_FOLDS >= 2

    rows = [{"thread_id": str(i), "customer_msg": f"msg {i}", "intent": f"intent_{i % 3}"}
            for i in range(20)]
    seen: list[set[str]] = []

    class _Spy:
        def __init__(self, train_ids):
            self.train_ids = train_ids

        def run(self, items, on_progress=None):
            seen.append({tid for tid, _ in items} & self.train_ids)
            return [type("O", (), {"as_dict": lambda s, t=tid: {"thread_id": t}})()
                    for tid, _ in items]

    def fake_fit(name, messages, labels, retriever):
        return _Spy({r["thread_id"] for r in rows if r["customer_msg"] in set(messages)})

    original = run_eval._fit_baseline
    run_eval._fit_baseline = fake_fit
    try:
        out = run_eval.out_of_fold_outputs("simple_tfidf_nn", rows, None, k=5)
    finally:
        run_eval._fit_baseline = original

    # Every row predicted exactly once, and no fold ever predicted a row it trained on.
    assert sorted(o["thread_id"] for o in out) == sorted(r["thread_id"] for r in rows)
    assert all(not overlap for overlap in seen), f"train/test overlap in folds: {seen}"


# --------------------------------------------------------------------------
# Per-role provider chains
# --------------------------------------------------------------------------

def test_judge_role_defaults_to_a_different_vendor_than_the_drafter(monkeypatch):
    """The judge must not share the drafter's lineage by default.

    A model grading its own output cannot rule out self-preference, so the
    judge role has its own provider chain rather than inheriting the global one.
    """
    monkeypatch.setenv("GROUNDSCORE_PROVIDER", "anthropic")
    monkeypatch.delenv("GROUNDSCORE_ROLE_JUDGE", raising=False)
    monkeypatch.delenv("GROUNDSCORE_ROLE_FAST", raising=False)
    assert llm.role_chain("fast")[0] == "anthropic"
    assert llm.role_chain("judge")[0] == "gemini"
    assert llm.role_chain("judge")[0] != llm.role_chain("fast")[0]


def test_role_chain_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("GROUNDSCORE_ROLE_JUDGE", "gemini,nope")
    with pytest.raises(Exception, match="unknown provider"):
        llm.role_chain("judge")


def test_replay_checks_every_provider_in_the_chain(monkeypatch):
    """A cache built when the judge ran on one provider must still replay.

    Otherwise reordering the chain silently invalidates the committed cache and
    `make reproduce` starts demanding live calls for numbers already published.
    """
    monkeypatch.setenv("GROUNDSCORE_ROLE_JUDGE", "gemini,ollama")
    monkeypatch.setenv("GROUNDSCORE_OFFLINE", "1")
    wanted = llm.cache_key(llm.model_for("ollama", "judge"), "p", None, 0.0, None, prov="ollama")
    monkeypatch.setattr(llm, "_cache_get", lambda k: "cached!" if k == wanted else None)
    # gemini is first in the chain and has no entry; the ollama entry must win.
    assert llm.complete("p", model=llm.MODEL_JUDGE) == "cached!"


def test_provider_switch_is_recorded_not_silent(monkeypatch):
    """A mid-run fallback means one split was scored by two models.

    That has to appear in the results, because those rows cannot honestly be
    pooled into a single reply-quality number.
    """
    monkeypatch.setenv("GROUNDSCORE_ROLE_JUDGE", "gemini,ollama")
    monkeypatch.delenv("GROUNDSCORE_OFFLINE", raising=False)
    monkeypatch.setattr(llm, "_cache_get", lambda k: None)
    monkeypatch.setattr(llm, "_cache_put", lambda *a: None)
    monkeypatch.setattr(llm, "usable", lambda prov: True)
    monkeypatch.setattr(llm, "_pinned", {})
    monkeypatch.setattr(llm, "SERVING", {})
    monkeypatch.setattr(llm, "PROVIDER_EVENTS", [])

    def dispatch(prov, model, *a):
        if prov == "gemini":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    monkeypatch.setattr(llm, "_dispatch", dispatch)
    assert llm.complete("p", model=llm.MODEL_JUDGE) == "ok"
    assert llm.SERVING["judge"].startswith("ollama:")
    assert any("RESOURCE_EXHAUSTED" in e["reason"] for e in llm.PROVIDER_EVENTS)
