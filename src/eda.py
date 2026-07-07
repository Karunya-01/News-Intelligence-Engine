"""
eda.py
Deliverable #1: EDA & label analysis.

Produces (saved to artifacts/eda/):
  - label_cooccurrence_heatmap.png
  - label_density_report.txt (avg labels/article, rare combos)
  - headline_vs_body_length.png
  - noise_audit.txt (HTML tags, encoding errors, scrape fragments found)
  - entity_type_distribution.png (uses the NER-labelled eval split, since
    the training set has no gold entities — see guideline #1)

Run standalone:
    python -m src.eda
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import parse_labels, preprocess_dataframe
from src.utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def label_cooccurrence_matrix(label_lists: pd.Series, categories: list[str]) -> pd.DataFrame:
    """Build a symmetric co-occurrence count matrix across the fixed topic categories."""
    mat = pd.DataFrame(0, index=categories, columns=categories, dtype=int)
    for labels in label_lists:
        labels = [l for l in labels if l in categories]
        for a in labels:
            mat.loc[a, a] += 1
        for a, b in combinations(sorted(set(labels)), 2):
            mat.loc[a, b] += 1
            mat.loc[b, a] += 1
    return mat


def plot_label_cooccurrence(mat: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis")
    plt.title("Label Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def label_density_report(label_lists: pd.Series) -> dict:
    n_labels = label_lists.apply(len)
    combo_counter = Counter(tuple(sorted(l)) for l in label_lists)
    rare_combos = [c for c, cnt in combo_counter.items() if cnt <= 3]
    return {
        "avg_labels_per_article": float(n_labels.mean()),
        "median_labels_per_article": float(n_labels.median()),
        "max_labels_per_article": int(n_labels.max()),
        "min_labels_per_article": int(n_labels.min()),
        "n_unique_label_combinations": len(combo_counter),
        "n_rare_combinations_leq3": len(rare_combos),
        "most_common_combinations": combo_counter.most_common(10),
    }


def plot_headline_vs_body_length(df: pd.DataFrame, out_path: Path) -> None:
    headline_len = df["clean_headline"].str.split().apply(len)
    body_len = df["clean_body"].str.split().apply(len)
    plt.figure(figsize=(7, 6))
    plt.scatter(headline_len, body_len, alpha=0.3, s=10)
    plt.xlabel("Headline length (words)")
    plt.ylabel("Body length (words)")
    plt.title(f"Headline vs Body length (corr={headline_len.corr(body_len):.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_body_length_distribution(df: pd.DataFrame, out_path: Path, max_bert_tokens: int) -> None:
    word_counts = df["body_word_count"]
    plt.figure(figsize=(8, 5))
    sns.histplot(word_counts, bins=40, kde=True)
    # rough token estimate ~1.3 tokens/word for BERT-style tokenizers
    est_token_cutoff = int(max_bert_tokens / 1.3)
    plt.axvline(est_token_cutoff, color="red", linestyle="--",
                label=f"~{max_bert_tokens}-token limit (~{est_token_cutoff} words)")
    plt.legend()
    plt.title("Article body length distribution")
    plt.xlabel("Word count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def noise_audit(raw_df: pd.DataFrame) -> dict:
    html_hits = raw_df["body_text"].fillna("").str.contains(r"<[a-zA-Z/][^>]*>").sum()
    entity_hits = raw_df["body_text"].fillna("").str.contains(r"&[a-zA-Z#0-9]+;").sum()
    null_body = raw_df["body_text"].isna().sum()
    null_language = raw_df["language"].isna().sum() if "language" in raw_df.columns else None
    scrape_noise_present = raw_df["scrape_noise"].notna().sum() if "scrape_noise" in raw_df.columns else None
    return {
        "rows_with_html_tags": int(html_hits),
        "rows_with_html_entities": int(entity_hits),
        "rows_with_null_body": int(null_body),
        "rows_with_null_language": None if null_language is None else int(null_language),
        "rows_with_scrape_noise_flag": None if scrape_noise_present is None else int(scrape_noise_present),
    }


def entity_type_distribution(ner_eval_df: pd.DataFrame, out_path: Path) -> None:
    """Entity type distribution from the gold NER eval split (entities_ref column)."""
    counter: Counter = Counter()
    for raw in ner_eval_df["entities_ref"].dropna():
        try:
            entities = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for ent in entities:
            counter[ent.get("label", "UNKNOWN")] += 1
    if not counter:
        logger.warning("No parsable entities found for entity type distribution plot.")
        return
    types, counts = zip(*counter.most_common())
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(types), y=list(counts))
    plt.title("Entity type distribution (gold NER eval split)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    out_dir = resolve_path("artifacts") / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(resolve_path(config["paths"]["train_csv"]))
    processed = preprocess_dataframe(raw_df, config)
    kept = processed[processed["is_kept"]].copy()
    kept["labels_parsed"] = kept["labels"].apply(parse_labels)

    categories = config["labels"]["topic_categories"]
    mat = label_cooccurrence_matrix(kept["labels_parsed"], categories)
    plot_label_cooccurrence(mat, out_dir / "label_cooccurrence_heatmap.png")

    density = label_density_report(kept["labels_parsed"])
    with open(out_dir / "label_density_report.txt", "w") as f:
        json.dump(density, f, indent=2, default=str)
    logger.info("Label density: %.2f labels/article on average", density["avg_labels_per_article"])

    plot_headline_vs_body_length(kept, out_dir / "headline_vs_body_length.png")
    plot_body_length_distribution(kept, out_dir / "body_length_distribution.png",
                                   config["preprocessing"]["max_bert_tokens"])

    audit = noise_audit(raw_df)
    with open(out_dir / "noise_audit.txt", "w") as f:
        json.dump(audit, f, indent=2)
    logger.info("Noise audit: %s", audit)

    ner_eval_path = resolve_path(config["paths"]["eval_ner_csv"])
    if ner_eval_path.exists():
        ner_df = pd.read_csv(ner_eval_path)
        entity_type_distribution(ner_df, out_dir / "entity_type_distribution.png")

    logger.info("EDA artifacts written to %s", out_dir)


if __name__ == "__main__":
    run()
