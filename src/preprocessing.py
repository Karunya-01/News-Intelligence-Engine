"""
preprocessing.py
Deliverable #2: Text preprocessing pipeline.

Handles:
  - HTML tag stripping (headline/body_text contain ~8% HTML noise)
  - Unicode normalisation / mojibake repair (encoding noise ~5% of body_text)
  - Programmatic language detection & filtering via langdetect (the raw
    `language` column has ~10% nulls / wrong labels and cannot be trusted)
  - Boilerplate / scrape-noise removal (ads, nav fragments in `scrape_noise`)
  - Label parsing (JSON-ish string -> list[str])
  - Truncation strategy for the 512-token BERT limit (headline + first N
    sentences, per the approach doc)

Run standalone:
    python -m src.preprocessing
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic langdetect
    _HAS_LANGDETECT = True
except ImportError:  # pragma: no cover - keeps the pipeline runnable offline
    _HAS_LANGDETECT = False
    logging.getLogger(__name__).warning(
        "langdetect not installed; falling back to a crude ASCII/stopword heuristic. "
        "`pip install langdetect` for production-quality detection (required by guideline #3)."
    )

    _EN_STOPWORDS = {
        "the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with",
        "as", "was", "at", "by", "an", "be", "has", "have", "it", "its",
    }

    def _fallback_detect_language(text: str) -> str:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        if not tokens:
            return "unk"
        ascii_ratio = sum(t.isascii() for t in tokens) / len(tokens)
        stopword_hits = sum(t in _EN_STOPWORDS for t in tokens[:60])
        if ascii_ratio > 0.9 and stopword_hits >= 3:
            return "en"
        return "unk"

try:
    import ftfy

    _HAS_FTFY = True
except ImportError:  # pragma: no cover - keeps the pipeline runnable offline
    _HAS_FTFY = False
    logging.getLogger(__name__).warning(
        "ftfy not installed; mojibake repair will be skipped. `pip install ftfy` for full fidelity."
    )

from src.utils import guard_against_leakage, load_config, resolve_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities (&amp;, &nbsp;, &#x27; etc.)."""
    if not isinstance(text, str) or not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def fix_unicode(text: str) -> str:
    """Repair mojibake (e.g. 'w�ich' / smart-quote artifacts) and normalise unicode."""
    if not isinstance(text, str) or not text:
        return ""
    if _HAS_FTFY:
        text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    return text


def remove_boilerplate(text: str, scrape_noise: Optional[str] = None) -> str:
    """
    Strip leftover nav/ad fragments. Uses the scrape_noise column when present,
    plus a small set of generic boilerplate patterns as a fallback.
    """
    if not isinstance(text, str):
        return ""
    if isinstance(scrape_noise, str) and scrape_noise.strip():
        for fragment in scrape_noise.split("|"):
            fragment = fragment.strip()
            if fragment:
                text = text.replace(fragment, " ")
    generic_patterns = [
        r"subscribe now.*?(\.|$)",
        r"advertisement",
        r"click here to.*?(\.|$)",
        r"share this article",
        r"related articles?:.*?(\.|$)",
    ]
    for pattern in generic_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def detect_language_safe(text: str) -> Optional[str]:
    """
    Programmatic language detection (guideline #3): the raw `language`
    column is unreliable (~10% nulls/wrong), so this is the source of truth.
    """
    if not isinstance(text, str) or len(text.strip()) < 20:
        return None
    if not _HAS_LANGDETECT:
        return _fallback_detect_language(text)
    try:
        return detect(text)
    except LangDetectException:
        return None


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def truncate_for_bert(headline: str, body: str, n_sentences: int = 3) -> str:
    """
    Approach doc step 5: use headline + first N sentences as the model input,
    which comfortably respects the 512-token BERT limit while keeping the
    highest-signal part of the article.
    """
    sentences = split_sentences(body)
    lead = " ".join(sentences[:n_sentences])
    headline = headline or ""
    return clean_whitespace(f"{headline}. {lead}")


def parse_labels(raw_labels) -> list[str]:
    """
    labels column is stored as a JSON-list-looking string with doubled quotes
    from the CSV export, e.g.  ["Economy", "Technology"]  -> parse robustly.
    """
    if isinstance(raw_labels, list):
        return raw_labels
    if not isinstance(raw_labels, str) or not raw_labels.strip():
        return []
    candidate = raw_labels.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # fall back to a permissive extraction of quoted tokens
        return re.findall(r'"([^"]+)"', candidate)


def preprocess_dataframe(
    df: pd.DataFrame,
    config: dict,
    text_col: str = "body_text",
    headline_col: str = "headline",
    require_min_words: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one dataframe. Returns a NEW dataframe
    with added columns: clean_headline, clean_body, detected_language,
    model_input (truncated headline+lead), body_word_count, is_kept.
    """
    guard_against_leakage(df, feature_columns=[headline_col, text_col], config=config)

    out = df.copy()

    out["clean_headline"] = out[headline_col].fillna("").apply(strip_html).apply(fix_unicode)
    out["clean_headline"] = out["clean_headline"].apply(clean_whitespace)

    scrape_noise_col = out["scrape_noise"] if "scrape_noise" in out.columns else pd.Series([None] * len(out))
    out["clean_body"] = out[text_col].fillna("").apply(strip_html).apply(fix_unicode)
    out["clean_body"] = [
        remove_boilerplate(t, sn) for t, sn in zip(out["clean_body"], scrape_noise_col)
    ]
    out["clean_body"] = out["clean_body"].apply(clean_whitespace)

    if config["preprocessing"]["language_detect"]:
        out["detected_language"] = out["clean_body"].apply(detect_language_safe)
    else:
        out["detected_language"] = out.get("language")

    out["body_word_count"] = out["clean_body"].apply(lambda t: len(t.split()))

    out["model_input"] = [
        truncate_for_bert(h, b) for h, b in zip(out["clean_headline"], out["clean_body"])
    ]

    if "labels" in out.columns:
        out["labels_parsed"] = out["labels"].apply(parse_labels)

    allowed = set(config["preprocessing"]["allowed_languages"])
    min_words = config["preprocessing"]["min_body_words"]
    keep = out["detected_language"].isin(allowed)
    if require_min_words:
        keep &= out["body_word_count"] >= min_words
    out["is_kept"] = keep

    return out


def run(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    train_path = resolve_path(config["paths"]["train_csv"])
    df = pd.read_csv(train_path)
    logger.info("Loaded %d raw rows from %s", len(df), train_path)

    processed = preprocess_dataframe(df, config)
    kept = processed[processed["is_kept"]]
    logger.info(
        "Kept %d / %d rows after language filtering + min-word filtering (%.1f%%)",
        len(kept), len(processed), 100 * len(kept) / len(processed),
    )

    out_dir = resolve_path(config["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_processed.parquet"
    try:
        processed.to_parquet(out_path, index=False)
        logger.info("Wrote processed dataframe to %s", out_path)
    except (ImportError, ValueError):
        csv_fallback = out_dir / "train_processed.csv"
        processed.to_csv(csv_fallback, index=False)
        logger.warning(
            "pyarrow/fastparquet not available; wrote CSV fallback to %s instead", csv_fallback
        )


if __name__ == "__main__":
    run()
