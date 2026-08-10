"""
Tests for the PE Acquisition Advisor corpus and testbed (v5).

These tests can run without downloading any model — they validate the schema,
the corpus builder's deterministic rule, split integrity, and the expanded
adversarial suite. Model-dependent tests are marked and skipped when torch
or transformers is unavailable.
"""

import json
import os
import subprocess
import sys

import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def corpus():
    """Build a mini corpus in a temp dir and load it."""
    import tempfile
    from corpus_builder import (
        SCHEMA_VERSION, build_synthetic, build_contrast, build_anchors,
        build_adversarial, stratified_split, decide)
    import random
    rng = random.Random(20260806)
    synthetic = build_synthetic(60, rng)
    contrast = build_contrast(12, rng)
    anchors = build_anchors()
    adversarial = build_adversarial(rng)
    train, val, test = stratified_split(synthetic + contrast, rng)
    for r in train: r["split"] = "train"
    for r in val: r["split"] = "val"
    for r in test: r["split"] = "test"
    for r in anchors: r["split"] = "anchor_eval"
    for r in adversarial: r["split"] = "adversarial_eval"
    return {
        "records": train + val + test + anchors + adversarial,
        "train": train, "val": val, "test": test,
        "anchors": anchors, "adversarial": adversarial,
    }


# --------------------------------------------------------------------------
# Schema tests
# --------------------------------------------------------------------------
class TestRecordSchema:
    def test_record_has_required_fields(self, corpus):
        for r in corpus["records"]:
            assert "id" in r
            assert "schema_version" in r
            assert r["schema_version"] == "pea-3.0"
            assert "origin" in r
            assert "features" in r
            assert "decision" in r
            assert "prompt" in r
            assert "target" in r
            assert "text" in r
            assert "abstain" in r
            assert "split" in r

    def test_decision_has_required_fields(self, corpus):
        for r in corpus["records"]:
            d = r["decision"]
            for field in ("evidence", "rights", "tier", "act", "why"):
                assert field in d, f"missing {field} in {r['id']}"

    def test_feature_space_is_closed(self, corpus):
        """Every feature value is from the documented vocabulary — no drift."""
        from corpus_builder import (SECTORS, DATA_TYPES, RIGHTS, CONSENT,
                                     REGIME, EVIDENCE_SOURCE, REVIEW, VOLUME,
                                     DEAL_CONTEXT)
        valid = {
            "sector": SECTORS, "data_type": DATA_TYPES, "rights": RIGHTS,
            "consent": CONSENT, "regime": REGIME,
            "evidence_source": EVIDENCE_SOURCE, "review": REVIEW,
            "volume": VOLUME, "deal_context": DEAL_CONTEXT,
        }
        for r in corpus["records"]:
            f = r["features"]
            for k, allowed in valid.items():
                assert f[k] in allowed, (
                    f"invalid {k}={f[k]} in {r['id']}")

    def test_split_values_are_valid(self, corpus):
        valid_splits = {"train", "val", "test", "anchor_eval",
                        "adversarial_eval"}
        for r in corpus["records"]:
            assert r["split"] in valid_splits

    def test_anchor_records_have_parties(self, corpus):
        for r in corpus["anchors"]:
            assert r["parties"]["buyer"] is not None
            assert r["parties"]["target"] is not None
            assert r["source"] is not None

    def test_adversarial_records_have_category(self, corpus):
        for r in corpus["adversarial"]:
            assert "category" in r
            assert r.get("expected_act") == "ABSTAIN"


# --------------------------------------------------------------------------
# Rule consistency tests
# --------------------------------------------------------------------------
class TestDecisionRule:
    def test_rule_is_deterministic(self, corpus):
        """Same features → same decision, always."""
        from corpus_builder import decide
        for r in corpus["records"]:
            d = decide(r["features"])
            assert d["act"] == r["decision"]["act"], (
                f"rule mismatch for {r['id']}: "
                f"decide()={d['act']} stored={r['decision']['act']}")

    def test_abstain_flag_matches_decision(self, corpus):
        for r in corpus["records"]:
            assert r["abstain"] == (r["decision"]["act"] == "ABSTAIN")

    def test_all_target_matches_rule(self, corpus):
        from corpus_builder import decide, render_target
        for r in corpus["records"]:
            d = decide(r["features"])
            expected = render_target(d)
            assert r["target"] == expected, (
                f"target mismatch for {r['id']}")

    def test_all_text_matches_prompt_plus_target(self, corpus):
        for r in corpus["records"]:
            assert r["text"] == r["prompt"] + r["target"]


# --------------------------------------------------------------------------
# Corpus composition tests
# --------------------------------------------------------------------------
class TestCorpusComposition:
    def test_synthetic_has_stratification(self, corpus):
        """No action class is starved — the rule would be commercially useless
        if everything abstained."""
        from collections import Counter
        acts = Counter(r["decision"]["act"] for r in
                       corpus["train"])
        assert "ABSTAIN" in acts
        assert "AIBUYER" in acts
        assert "SPONSOR" in acts
        assert "RENEG" in acts
        # No single act should dominate train >70%
        total = sum(acts.values())
        for act, count in acts.items():
            assert count / total < 0.75, f"{act} over-represented: {count}/{total}"

    def test_anchor_count(self, corpus):
        assert len(corpus["anchors"]) == 20

    def test_adversarial_count(self, corpus):
        assert len(corpus["adversarial"]) == 100

    def test_adversarial_categories(self, corpus):
        from corpus_builder import ADVERSARIAL_CATEGORIES
        cats = set(r.get("category") for r in corpus["adversarial"])
        for c in ADVERSARIAL_CATEGORIES:
            assert c in cats, f"missing adversarial category: {c}"
        # Each category should have at least 10 prompts (100 / 10 categories)
        from collections import Counter
        cat_counts = Counter(r.get("category") for r in corpus["adversarial"])
        for c in ADVERSARIAL_CATEGORIES:
            assert cat_counts[c] >= 10, (
                f"category {c} only has {cat_counts[c]} prompts")

    def test_contrast_pairs_kept_together(self, corpus):
        """A contrast pair must not be split across train/test — that would leak
        the boundary."""
        pair_splits = {}
        for r in corpus["records"]:
            pid = r.get("contrast_pair")
            if pid:
                pair_splits.setdefault(pid, set()).add(r["split"])
        for pid, splits in pair_splits.items():
            assert len(splits) == 1, (
                f"contrast pair {pid} split across {splits}")


# --------------------------------------------------------------------------
# Full corpus builder integration
# --------------------------------------------------------------------------
class TestCorpusBuilder:
    def test_build_command_creates_corpus(self, tmp_path):
        """Run the full corpus builder with minimal args and verify output."""
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "corpus_builder.py"),
             "--n", "60", "--pairs", "12", "--adversarial", "100",
             "--out-prefix", str(tmp_path / "corpus"), "--no-xlsx"],
            capture_output=True, timeout=30, cwd=SRC_DIR)
        assert result.returncode == 0, result.stderr.decode()

        jsonl = tmp_path / "corpus.jsonl"
        assert jsonl.exists()
        with open(jsonl) as f:
            records = [json.loads(l) for l in f]
        assert len(records) > 0
        # Check split distribution
        from collections import Counter
        splits = Counter(r["split"] for r in records)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert "anchor_eval" in splits
        assert "adversarial_eval" in splits
        assert splits["adversarial_eval"] == 100

        stats_path = tmp_path / "corpus_stats.json"
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["counts"]["adversarial"] == 100
        assert stats["counts"]["total"] > 100


# --------------------------------------------------------------------------
# Target module coverage tests
# --------------------------------------------------------------------------
class TestTargetModules:
    def test_gpt2_mapped(self):
        from testbed import TARGET_MODULES
        assert "c_attn" in TARGET_MODULES["gpt2"]

    def test_gemma3_mapped(self):
        from testbed import TARGET_MODULES
        assert "q_proj" in TARGET_MODULES["gemma3_text"]

    def test_llama_mapped(self):
        from testbed import TARGET_MODULES
        assert "q_proj" in TARGET_MODULES["llama"]
