"""
misinfo_scoring.py
Deliverable #7: Misinformation signal scoring notebook backend.

Five signals (approach doc step 8), each independently offline-computable:
  (a) clickbait_score        — headline sentiment intensity / hype-word ratio
  (b) emotional_language     — emotionally-charged word ratio (NRC lexicon if
                                available, else a small built-in emotion
                                wordlist fallback)
  (c) source_credibility     — 1 - (domain in whitelist), i.e. higher = riskier
  (d) factual_density        — inverse of named-entity count per 100 words
                                (fewer grounded facts = higher risk)
  (e) quote_authenticity     — ratio of indirect claims to direct quotes
                                (more unverifiable claims = higher risk)

Combined into a weighted composite Mis-Risk Score in [0, 1], calibrated
against the ground-truth `mis_risk_label` column (never used as an input
feature — see guideline #1 / the leakage guard).

Run standalone:
    python -m src.misinfo_scoring
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.ner_model import extract_entities_rule_based
from src.utils import guard_against_leakage, load_config, resolve_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CLICKBAIT_PHRASES = [
    "you won't believe", "shocking", "this one trick", "what happened next",
    "will blow your mind", "number", "secret", "top", "never", "always",
    "guess what", "the truth about",
]

_EMOTION_WORDS = {
    "outrage", "furious", "terrified", "shocking", "devastating", "horrific",
    "amazing", "miracle", "disaster", "crisis", "panic", "fear", "rage",
    "heartbreaking", "explosive", "chaos", "catastrophe", "alarming",
}

_QUOTE_RE = re.compile(r"['\u2018\u2019][^'\u2018\u2019]{3,}['\u2018\u2019]")
_CLAIM_VERB_RE = re.compile(
    r"\b(said|claimed|alleged|reportedly|according to|sources say|suggested|warned)\b",
    re.IGNORECASE,
)


def clickbait_score(headline: str) -> float:
    """Higher = more clickbait-y. Rule-based hype-phrase + punctuation heuristic."""
    if not isinstance(headline, str) or not headline:
        return 0.0
    text = headline.lower()
    phrase_hits = sum(1 for p in _CLICKBAIT_PHRASES if p in text)
    punctuation_bonus = 0.15 if ("!" in headline or "?" in headline) else 0.0
    number_bonus = 0.1 if re.search(r"\d+", headline) else 0.0
    score = min(1.0, 0.2 * phrase_hits + punctuation_bonus + number_bonus)
    return score


def emotional_language_ratio(body: str) -> float:
    """Higher = more emotionally-charged. Falls back to a small builtin lexicon if NRCLex is absent."""
    if not isinstance(body, str) or not body.strip():
        return 0.0
    try:
        from nrclex import NRCLex

        nrc = NRCLex(body)
        freqs = nrc.affect_frequencies
        emotional = sum(v for k, v in freqs.items() if k not in ("positive", "negative", "anticip"))
        return float(min(1.0, emotional * 3))
    except ImportError:
        tokens = re.findall(r"[a-zA-Z']+", body.lower())
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in _EMOTION_WORDS)
        return float(min(1.0, hits / max(len(tokens), 1) * 20))


def source_credibility_risk(source_domain: str, whitelist: list[str]) -> float:
    """Returns RISK (0 = fully credible, 1 = not on whitelist / unknown)."""
    if not isinstance(source_domain, str) or not source_domain:
        return 1.0
    domain = source_domain.lower().strip()
    domain = domain.replace("www.", "")
    try:
        parsed = urlparse(domain if "://" in domain else f"http://{domain}")
        domain = parsed.netloc or parsed.path
    except ValueError:
        pass
    return 0.0 if domain in {d.lower() for d in whitelist} else 1.0


def factual_density_risk(body: str) -> float:
    """Higher = fewer grounded facts (risk). Uses the rule-based NER fallback for entity counts."""
    if not isinstance(body, str) or not body.strip():
        return 1.0
    entities = extract_entities_rule_based(body)
    n_words = max(len(body.split()), 1)
    entities_per_100 = len(entities) / n_words * 100
    # normalise: >=8 entities/100 words -> low risk (0), 0 entities -> high risk (1)
    risk = max(0.0, 1.0 - min(entities_per_100 / 8.0, 1.0))
    return float(risk)


def quote_authenticity_risk(body: str) -> float:
    """Higher = more unverifiable indirect claims relative to direct quotes (risk)."""
    if not isinstance(body, str) or not body.strip():
        return 0.5
    n_quotes = len(_QUOTE_RE.findall(body))
    n_claims = len(_CLAIM_VERB_RE.findall(body))
    if n_quotes + n_claims == 0:
        return 0.5
    return float(n_claims / (n_quotes + n_claims))


def composite_mis_risk_score(row: pd.Series, weights: dict, whitelist: list[str]) -> dict:
    signals = {
        "clickbait": clickbait_score(row.get("headline", "")),
        "emotional_language": emotional_language_ratio(row.get("body_text", "")),
        "source_credibility": source_credibility_risk(row.get("source_domain", ""), whitelist),
        "factual_density": factual_density_risk(row.get("body_text", "")),
        "quote_authenticity": quote_authenticity_risk(row.get("body_text", "")),
    }
    composite = sum(weights[k] * v for k, v in signals.items())
    signals["mis_risk_score"] = float(min(max(composite, 0.0), 1.0))
    return signals


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def run(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    df = pd.read_csv(resolve_path(config["paths"]["train_csv"]))

    # mis_risk_label is evaluation-only ground truth for calibration, never a feature.
    guard_against_leakage(df, feature_columns=["headline", "body_text", "source_domain"], config=config)

    cfg = config["misinfo_scoring"]
    scored = df.apply(lambda r: composite_mis_risk_score(r, cfg["weights"], cfg["credible_domain_whitelist"]), axis=1)
    scored_df = pd.json_normalize(scored)
    result = pd.concat([df[["article_id", "headline", "source_domain"]].reset_index(drop=True), scored_df], axis=1)

    # Calibration vs human ground truth (mis_risk_label), where available.
    labeled = df["mis_risk_label"].notna()
    if labeled.sum() > 0:
        y_true = df.loc[labeled, "mis_risk_label"].values
        y_pred = scored_df.loc[labeled, "mis_risk_score"].values
        bscore = brier_score(y_true, y_pred)
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        logger.info("Calibration vs human mis_risk_label (n=%d): Brier=%.4f, Pearson r=%.4f",
                    labeled.sum(), bscore, corr)

    top10 = result.sort_values("mis_risk_score", ascending=False).head(10)
    logger.info("Top-10 highest-risk articles:\n%s",
                top10[["article_id", "headline", "mis_risk_score"]].to_string(index=False))

    out_dir = resolve_path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "misinfo_scores.csv", index=False)
    logger.info("Wrote misinformation scores for %d articles to artifacts/misinfo_scores.csv", len(result))
    return result


if __name__ == "__main__":
    run()
