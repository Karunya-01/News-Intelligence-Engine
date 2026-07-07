# %% [markdown]
# # 03 — Baseline Classification (TF-IDF + LR / SVM, Word2Vec + MLP)

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()))

from src.baseline_models import (
    build_dataset,
    manual_tfidf_example,
    train_tfidf_logreg,
    train_tfidf_svm,
    train_word2vec_mlp,
)
from src.utils import load_config, set_global_seed

config = load_config()
set_global_seed(config["seed"])

# %% [markdown]
# ## Build the leakage-safe multilabel dataset

# %%
ds = build_dataset(config)
print(f"train={len(ds.X_train_text)} val={len(ds.X_val_text)} test={len(ds.X_test_text)}")
print("Labels:", list(ds.mlb.classes_))

# %% [markdown]
# ## Manual TF-IDF worked example (from the approach doc formula)
# TF(t,d) = count(t,d) / len(d)   ·   IDF(t) = log(N / df(t))   ·   TF-IDF = TF × IDF

# %%
manual_tfidf_example(ds.X_train_text, "economy")

# %% [markdown]
# ## Baseline 1: TF-IDF + Logistic Regression (One-vs-Rest) + per-label thresholds

# %%
lr_vectorizer, lr_model, lr_metrics = train_tfidf_logreg(ds, config)
print(f"micro-F1={lr_metrics['micro_f1']:.4f}  macro-F1={lr_metrics['macro_f1']:.4f}  "
      f"Hamming={lr_metrics['hamming_loss']:.4f}  Jaccard={lr_metrics['jaccard_samples']:.4f}")
lr_metrics["per_label_thresholds"]

# %% [markdown]
# ## Baseline 2: TF-IDF + Linear SVM (One-vs-Rest)

# %%
svm_vectorizer, svm_model, svm_metrics = train_tfidf_svm(ds, config)
print(f"micro-F1={svm_metrics['micro_f1']:.4f}  macro-F1={svm_metrics['macro_f1']:.4f}  "
      f"Hamming={svm_metrics['hamming_loss']:.4f}  Jaccard={svm_metrics['jaccard_samples']:.4f}")

# %% [markdown]
# ## Baseline 3: Word2Vec averaged embeddings + MLP
# (requires `pip install gensim` — skipped automatically if unavailable)

# %%
w2v_model, mlp_model, w2v_metrics = train_word2vec_mlp(ds, config)
if w2v_metrics:
    print(f"micro-F1={w2v_metrics['micro_f1']:.4f}  macro-F1={w2v_metrics['macro_f1']:.4f}")
else:
    print("gensim not available in this environment — install it to run this baseline.")

# %% [markdown]
# ## Comparison table

# %%
import pandas as pd

rows = [{"model": "TF-IDF + LogReg", **{k: v for k, v in lr_metrics.items() if k not in ("per_label", "per_label_thresholds")}},
        {"model": "TF-IDF + LinearSVC", **{k: v for k, v in svm_metrics.items() if k != "per_label"}}]
if w2v_metrics:
    rows.append({"model": "Word2Vec + MLP", **{k: v for k, v in w2v_metrics.items() if k != "per_label"}})
pd.DataFrame(rows)
