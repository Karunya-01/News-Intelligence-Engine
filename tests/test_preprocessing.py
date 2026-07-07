import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_whitespace,
    fix_unicode,
    parse_labels,
    strip_html,
    truncate_for_bert,
)
from src.threshold_tuning import apply_thresholds, find_optimal_thresholds
from src.utils import LeakageGuardError, guard_against_leakage, load_config
from src.misinfo_scoring import clickbait_score, source_credibility_risk


def test_strip_html_removes_tags_and_entities():
    dirty = "Cybersecurity expert Isabel Stou&amp; called&nbsp;the vulnerability <b>'critical'</b>."
    clean = strip_html(dirty)
    assert "<b>" not in clean
    assert "&amp;" not in clean
    assert "critical" in clean


def test_parse_labels_handles_doubled_quotes():
    raw = '["Economy", "Technology", "Environment"]'
    assert parse_labels(raw) == ["Economy", "Technology", "Environment"]


def test_parse_labels_handles_garbage_gracefully():
    assert parse_labels("") == []
    assert parse_labels(None) == []


def test_truncate_for_bert_uses_headline_and_lead_sentences():
    headline = "Markets fall"
    body = "First sentence. Second sentence. Third sentence. Fourth sentence should be dropped."
    result = truncate_for_bert(headline, body, n_sentences=3)
    assert "Fourth sentence" not in result
    assert "First sentence" in result
    assert result.startswith("Markets fall")


def test_clean_whitespace_collapses_spaces():
    assert clean_whitespace("a   b\n\nc") == "a b c"


def test_leakage_guard_blocks_forbidden_columns():
    config = {"leakage_columns": ["summary_ref", "entities_ref", "mis_risk_label"]}
    df = pd.DataFrame({"summary_ref": ["x"], "headline": ["y"]})
    with pytest.raises(LeakageGuardError):
        guard_against_leakage(df, feature_columns=["summary_ref", "headline"], config=config)


def test_leakage_guard_allows_safe_columns():
    config = {"leakage_columns": ["summary_ref", "entities_ref", "mis_risk_label"]}
    df = pd.DataFrame({"headline": ["y"], "body_text": ["z"]})
    guard_against_leakage(df, feature_columns=["headline", "body_text"], config=config)  # should not raise


def test_find_optimal_thresholds_improves_or_matches_default():
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]])
    y_proba = np.array([[0.9, 0.2], [0.8, 0.3], [0.4, 0.7], [0.3, 0.9], [0.6, 0.6]])
    thresholds = find_optimal_thresholds(y_true, y_proba, ["A", "B"], grid=(0.1, 0.9, 0.05))
    preds = apply_thresholds(y_proba, ["A", "B"], thresholds)
    assert preds.shape == y_true.shape
    assert set(thresholds.keys()) == {"A", "B"}


def test_clickbait_score_ranks_hype_higher():
    hype = "You won't believe what happened next!!!"
    plain = "Central bank raises interest rates"
    assert clickbait_score(hype) > clickbait_score(plain)


def test_source_credibility_whitelist():
    whitelist = ["bbc.com", "reuters.com"]
    assert source_credibility_risk("bbc.com", whitelist) == 0.0
    assert source_credibility_risk("randomblog.net", whitelist) == 1.0


def test_config_loads_and_has_required_sections():
    config = load_config()
    for key in ["seed", "paths", "leakage_columns", "labels", "baseline", "transformer_classifier"]:
        assert key in config
