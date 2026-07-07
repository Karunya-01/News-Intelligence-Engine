from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import parse_labels, preprocess_dataframe
from src.threshold_tuning import apply_thresholds, find_optimal_thresholds
from src.utils import guard_against_leakage, load_config, resolve_path, set_global_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultilabelDataset:
    X_train_text: list
    X_val_text: list
    X_test_text: list
    Y_train: np.ndarray
    Y_val: np.ndarray
    Y_test: np.ndarray
    mlb: MultiLabelBinarizer


def build_dataset(config: dict) -> MultilabelDataset:
    raw_df = pd.read_csv(resolve_path(config["paths"]["train_csv"]))
    guard_against_leakage(raw_df, feature_columns=["headline", "body_text"], config=config)

    processed = preprocess_dataframe(raw_df, config)
    kept = processed[processed["is_kept"]].copy()
    kept["labels_parsed"] = kept["labels"].apply(parse_labels)

    categories = config["labels"]["topic_categories"]
    kept["labels_parsed"] = kept["labels_parsed"].apply(lambda ls: [l for l in ls if l in categories])
    kept = kept[kept["labels_parsed"].apply(len) > 0]

    mlb = MultiLabelBinarizer(classes=categories)
    Y = mlb.fit_transform(kept["labels_parsed"])
    X = kept["model_input"].tolist()

    val_size = config["split"]["val_size"]
    test_size = config["split"]["test_size"]
    seed = config["seed"]

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(val_size + test_size), random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=relative_test, random_state=seed
    )
    return MultilabelDataset(X_train, X_val, X_test, Y_train, Y_val, Y_test, mlb)


def manual_tfidf_example(corpus: list[str], term: str) -> None:
    """Illustrates the manual TF-IDF formula quoted in the approach doc."""
    n_docs = len(corpus)
    df_count = sum(1 for doc in corpus if term in doc.lower().split())
    if df_count == 0:
        logger.info("Term '%s' not found in corpus for manual TF-IDF demo.", term)
        return
    idf = np.log(n_docs / df_count)
    doc0_terms = corpus[0].lower().split()
    tf = doc0_terms.count(term) / max(len(doc0_terms), 1)
    logger.info("Manual TF-IDF demo for term=%r: TF(doc0)=%.5f IDF=%.5f TF-IDF=%.5f",
                term, tf, idf, tf * idf)


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label = {
        label_names[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
        for i in range(len(label_names))
    }
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "hamming_loss": float(h_loss),
        "jaccard_samples": float(jaccard),
        "per_label": per_label,
    }


def train_tfidf_logreg(ds: MultilabelDataset, config: dict):
    cfg = config["baseline"]
    vectorizer = TfidfVectorizer(
        max_features=cfg["tfidf_max_features"],
        ngram_range=tuple(cfg["tfidf_ngram_range"]),
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(ds.X_train_text)
    X_val = vectorizer.transform(ds.X_val_text)

    clf = OneVsRestClassifier(LogisticRegression(C=cfg["logreg_C"], max_iter=1000, class_weight="balanced"))
    clf.fit(X_train, ds.Y_train)

    # Guideline #2: per-label threshold tuning instead of a global 0.5 cutoff.
    y_proba = clf.predict_proba(X_val)
    grid = tuple(config["transformer_classifier"]["threshold_search_grid"])
    thresholds = find_optimal_thresholds(ds.Y_val, y_proba, list(ds.mlb.classes_), grid)
    preds = apply_thresholds(y_proba, list(ds.mlb.classes_), thresholds)

    metrics = evaluate_multilabel(ds.Y_val, preds, list(ds.mlb.classes_))
    metrics["per_label_thresholds"] = thresholds
    return vectorizer, clf, metrics


def train_tfidf_svm(ds: MultilabelDataset, config: dict):
    cfg = config["baseline"]
    vectorizer = TfidfVectorizer(
        max_features=cfg["tfidf_max_features"],
        ngram_range=tuple(cfg["tfidf_ngram_range"]),
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(ds.X_train_text)
    X_val = vectorizer.transform(ds.X_val_text)

    clf = OneVsRestClassifier(LinearSVC(C=cfg["svm_C"], class_weight="balanced"))
    clf.fit(X_train, ds.Y_train)
    preds = clf.predict(X_val)
    metrics = evaluate_multilabel(ds.Y_val, preds, list(ds.mlb.classes_))
    return vectorizer, clf, metrics


def train_word2vec_mlp(ds: MultilabelDataset, config: dict):
    """
    Word2Vec averaged embeddings + MLP baseline. Requires gensim; falls back
    gracefully with a clear message if it isn't installed (e.g. offline env).
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        logger.warning("gensim not installed — skipping Word2Vec+MLP baseline. `pip install gensim`.")
        return None, None, None

    from sklearn.neural_network import MLPClassifier

    cfg = config["baseline"]
    tokenized_train = [t.lower().split() for t in ds.X_train_text]
    w2v = Word2Vec(
        sentences=tokenized_train,
        vector_size=cfg["word2vec_dim"],
        window=cfg["word2vec_window"],
        min_count=cfg["word2vec_min_count"],
        seed=config["seed"],
        workers=1,
    )

    def embed(text: str) -> np.ndarray:
        tokens = [t for t in text.lower().split() if t in w2v.wv]
        if not tokens:
            return np.zeros(cfg["word2vec_dim"])
        return np.mean([w2v.wv[t] for t in tokens], axis=0)

    X_train = np.vstack([embed(t) for t in ds.X_train_text])
    X_val = np.vstack([embed(t) for t in ds.X_val_text])

    clf = OneVsRestClassifier(
        MLPClassifier(hidden_layer_sizes=tuple(cfg["mlp_hidden_layers"]), random_state=config["seed"],
                      max_iter=300)
    )
    clf.fit(X_train, ds.Y_train)
    preds = clf.predict(X_val)
    metrics = evaluate_multilabel(ds.Y_val, preds, list(ds.mlb.classes_))
    return w2v, clf, metrics


def run(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    set_global_seed(config["seed"])

    ds = build_dataset(config)
    logger.info("Dataset sizes -> train: %d, val: %d, test: %d",
                len(ds.X_train_text), len(ds.X_val_text), len(ds.X_test_text))

    manual_tfidf_example(ds.X_train_text, "economy")

    results = []

    _, _, lr_metrics = train_tfidf_logreg(ds, config)
    results.append({"model": "TF-IDF + LogisticRegression",
                     **{k: v for k, v in lr_metrics.items() if k not in ("per_label", "per_label_thresholds")}})
    logger.info("LogReg micro-F1=%.4f macro-F1=%.4f", lr_metrics["micro_f1"], lr_metrics["macro_f1"])

    _, _, svm_metrics = train_tfidf_svm(ds, config)
    results.append({"model": "TF-IDF + LinearSVC", **{k: v for k, v in svm_metrics.items() if k != "per_label"}})
    logger.info("SVM micro-F1=%.4f macro-F1=%.4f", svm_metrics["micro_f1"], svm_metrics["macro_f1"])

    w2v_result = train_word2vec_mlp(ds, config)
    if w2v_result[-1] is not None:
        w2v_metrics = w2v_result[-1]
        results.append({"model": "Word2Vec + MLP", **{k: v for k, v in w2v_metrics.items() if k != "per_label"}})
        logger.info("Word2Vec+MLP micro-F1=%.4f macro-F1=%.4f", w2v_metrics["micro_f1"], w2v_metrics["macro_f1"])

    comparison = pd.DataFrame(results)
    out_dir = resolve_path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "baseline_comparison.csv", index=False)
    logger.info("\n%s", comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    run()
