# %% [markdown]
# # 06 — Abstractive Summarisation (extractive baseline -> T5/BART fine-tune)

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()))

import pandas as pd
from src.summarizer import (
    bert_score_f1,
    enforce_entity_grounding,
    extractive_baseline_summary,
    rouge_scores,
)
from src.utils import load_config, resolve_path

config = load_config()
eval_df = pd.read_csv(resolve_path(config["paths"]["eval_summary_csv"]))
eval_df.head(3)

# %% [markdown]
# ## Extractive baseline (offline-safe) + entity-grounding enforcement

# %%
predictions, grounded_flags = [], []
for body in eval_df["body_text"].fillna(""):
    raw_summary = extractive_baseline_summary(body, n_sentences=3)
    grounded_summary, was_grounded = enforce_entity_grounding(raw_summary, body)
    predictions.append(grounded_summary)
    grounded_flags.append(was_grounded)

eval_df["generated_summary"] = predictions
eval_df["was_already_grounded"] = grounded_flags
eval_df[["headline", "generated_summary", "summary_ref"]].head(5)

# %% [markdown]
# ## Evaluate: ROUGE-1/2/L + BERTScore (guideline #5 — never ROUGE alone)
# Requires `pip install rouge-score bert-score`.

# %%
rouge = rouge_scores(predictions, eval_df["summary_ref"].fillna("").tolist())
bscore = bert_score_f1(predictions, eval_df["summary_ref"].fillna("").tolist())
print("ROUGE:", rouge)
print("BERTScore F1:", bscore)
print(f"Entity-grounded without repair: {sum(grounded_flags)}/{len(grounded_flags)}")

# %% [markdown]
# ## Fine-tune facebook/bart-base (or t5-small) — heavy, needs internet+GPU
# See `src/summarizer.finetune_seq2seq()` for the model/tokenizer setup.
# The missing piece to fill in here is the training Dataset: tokenize
# `(model_input, summary_ref)` pairs with the shared tokenizer, respecting
# `config["summarization"]["max_input_length"]` / `max_target_length`.

# %%
# from src.summarizer import finetune_seq2seq
# tokenizer, model, training_args = finetune_seq2seq(config)
# ... build tokenized train/val Datasets, then:
# trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=..., eval_dataset=...,
#                           compute_metrics=lambda p: {"rouge": rouge_scores(...), "bertscore": bert_score_f1(...)})
# trainer.train()

# %% [markdown]
# ## Sample output gallery (article -> summary side-by-side)

# %%
for _, row in eval_df.head(5).iterrows():
    print("HEADLINE:", row["headline"])
    print("GENERATED:", row["generated_summary"])
    print("REFERENCE:", row["summary_ref"])
    print("-" * 100)
