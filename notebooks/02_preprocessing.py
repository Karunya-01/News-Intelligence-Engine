# %% [markdown]
# # 02 — Preprocessing Walkthrough
# Demonstrates each preprocessing step from `src/preprocessing.py` on real
# sample rows before running the full pipeline.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()))

import pandas as pd
from src.preprocessing import (
    clean_whitespace,
    detect_language_safe,
    fix_unicode,
    parse_labels,
    preprocess_dataframe,
    remove_boilerplate,
    strip_html,
    truncate_for_bert,
)
from src.utils import load_config, resolve_path

config = load_config()
raw_df = pd.read_csv(resolve_path(config["paths"]["train_csv"]))
raw_df.head(3)

# %% [markdown]
# ## Step-by-step on one noisy row

# %%
sample = raw_df.iloc[1]
print("RAW:\n", sample["body_text"][:300])

# %%
step1 = strip_html(sample["body_text"])
print("AFTER strip_html:\n", step1[:300])

# %%
step2 = fix_unicode(step1)
print("AFTER fix_unicode:\n", step2[:300])

# %%
step3 = remove_boilerplate(step2, sample.get("scrape_noise"))
step3 = clean_whitespace(step3)
print("AFTER remove_boilerplate + clean_whitespace:\n", step3[:300])

# %%
lang = detect_language_safe(step3)
print("Detected language:", lang)

# %%
model_input = truncate_for_bert(sample["headline"], step3, n_sentences=3)
print("Model input (headline + first 3 sentences):\n", model_input)

# %% [markdown]
# ## Run the full pipeline over the whole training set

# %%
processed = preprocess_dataframe(raw_df, config)
print(f"Kept {processed['is_kept'].sum()} / {len(processed)} rows")
processed[["headline", "clean_headline", "detected_language", "body_word_count", "is_kept"]].head(10)

# %% [markdown]
# ## Save processed data for downstream notebooks

# %%
out_dir = resolve_path(config["paths"]["processed_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
try:
    processed.to_parquet(out_dir / "train_processed.parquet", index=False)
    print("Saved parquet.")
except (ImportError, ValueError):
    processed.to_csv(out_dir / "train_processed.csv", index=False)
    print("pyarrow unavailable — saved CSV instead.")
