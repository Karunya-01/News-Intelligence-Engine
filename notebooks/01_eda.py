# %% [markdown]
# # 01 — EDA & Label Analysis
# Open this file in VS Code with the Python extension and run cells with
# `Shift+Enter` (each `# %%` is a cell, shown in the Interactive Window).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()))

import pandas as pd
from src.eda import (
    entity_type_distribution,
    label_cooccurrence_matrix,
    label_density_report,
    noise_audit,
    plot_body_length_distribution,
    plot_headline_vs_body_length,
    plot_label_cooccurrence,
)
from src.preprocessing import parse_labels, preprocess_dataframe
from src.utils import load_config, resolve_path

config = load_config()

# %% [markdown]
# ## Load raw data & run the preprocessing pipeline

# %%
raw_df = pd.read_csv(resolve_path(config["paths"]["train_csv"]))
print(raw_df.shape)
raw_df.head()

# %%
processed = preprocess_dataframe(raw_df, config)
kept = processed[processed["is_kept"]].copy()
kept["labels_parsed"] = kept["labels"].apply(parse_labels)
print(f"Kept {len(kept)} / {len(processed)} rows after language + min-word filtering")

# %% [markdown]
# ## Label co-occurrence heatmap

# %%
categories = config["labels"]["topic_categories"]
mat = label_cooccurrence_matrix(kept["labels_parsed"], categories)
mat

# %%
out_dir = resolve_path("artifacts/eda")
out_dir.mkdir(parents=True, exist_ok=True)
plot_label_cooccurrence(mat, out_dir / "label_cooccurrence_heatmap.png")
print("Saved:", out_dir / "label_cooccurrence_heatmap.png")

# %% [markdown]
# ## Label density & rare combinations

# %%
density = label_density_report(kept["labels_parsed"])
density

# %% [markdown]
# ## Headline vs body length correlation

# %%
plot_headline_vs_body_length(kept, out_dir / "headline_vs_body_length.png")
plot_body_length_distribution(kept, out_dir / "body_length_distribution.png",
                               config["preprocessing"]["max_bert_tokens"])
print("Saved length plots to", out_dir)

# %% [markdown]
# ## Noise audit (HTML tags, encoding errors, scrape fragments)

# %%
audit = noise_audit(raw_df)
audit

# %% [markdown]
# ## Entity type distribution (gold NER eval split)

# %%
ner_df = pd.read_csv(resolve_path(config["paths"]["eval_ner_csv"]))
entity_type_distribution(ner_df, out_dir / "entity_type_distribution.png")
print("Saved entity type distribution plot")

# %% [markdown]
# **Note on this dataset:** the label/summary/entity columns in this
# capstone's data are synthetically generated for practice and are only
# loosely correlated with article content (e.g. label assignment looks
# close to random, and `entities_ref` spans are noisy). Expect much
# stronger real-world numbers once you point this pipeline at genuine
# news data — the pipeline logic itself is what's being graded.
