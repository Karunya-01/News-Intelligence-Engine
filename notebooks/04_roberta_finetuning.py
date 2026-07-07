import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()))

from src.classifier_transformer import finetune_roberta_classifier
from src.utils import load_config

config = load_config()

# %% [markdown]
# ## Fine-tune
# This single call: tokenizes with the shared RoBERTa tokenizer, builds a
# sigmoid multilabel head with label-weighted BCEWithLogitsLoss (to handle
# imbalance), trains for `config["transformer_classifier"]["epochs"]`
# epochs, then re-derives predictions using **per-label F1-optimal
# thresholds** (guideline #2) instead of a global 0.5 cutoff.

# %%
model, tokenizer, thresholds, final_metrics = finetune_roberta_classifier(config)

# %%
print(f"Final micro-F1: {final_metrics['micro_f1']:.4f}")
print(f"Final macro-F1: {final_metrics['macro_f1']:.4f}")
print(f"Hamming loss:   {final_metrics['hamming_loss']:.4f}")
print(f"Jaccard score:  {final_metrics['jaccard_samples']:.4f}")

# %% [markdown]
# ## Per-label thresholds found

# %%
thresholds

# %% [markdown]
# ## Per-label precision / recall / F1 (confusion matrix substitute for multilabel)

# %%
import pandas as pd

pd.DataFrame(final_metrics["per_label"]).T

# %% [markdown]
# ## SHAP token attribution (optional deliverable item)
# Requires `pip install shap`. Explains which tokens pushed a prediction
# toward / away from a given label for a sample article.

# %%
# import shap
# explainer = shap.Explainer(model, tokenizer)
# shap_values = explainer(["Central bank raises interest rates amid inflation fears."])
# shap.plots.text(shap_values[0])

# %% [markdown]
# Model artefact is saved to `models/model_roberta_v1/` per the versioning
# convention in the project guidelines (`model_roberta_v1.pt` etc).
