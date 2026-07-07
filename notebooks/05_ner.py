# %% [markdown]
# # 05 — Named Entity Recognition: spaCy baseline vs fine-tuned BERT-NER

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()))

import pandas as pd
from src.ner_model import (
    evaluate_ner,
    extract_entities_rule_based,
    load_gold_entities,
)
from src.utils import load_config, resolve_path

config = load_config()
eval_df = pd.read_csv(resolve_path(config["paths"]["eval_ner_csv"]))
gold = load_gold_entities(eval_df)

# %% [markdown]
# ## Offline smoke-test baseline (dependency-free rule-based extractor)
# This is NOT the graded NER model — it exists purely so the evaluation
# harness below is exercised and verified without any model download.

# %%
rule_based_preds = [extract_entities_rule_based(t) for t in eval_df["body_text"].fillna("")]
rule_based_report = evaluate_ner(gold, rule_based_preds)
pd.DataFrame(rule_based_report).T

# %% [markdown]
# ## spaCy baseline (`en_core_web_trf`)
# Requires `pip install spacy` and `python -m spacy download en_core_web_trf`.

# %%
try:
    from src.ner_model import extract_entities_spacy

    spacy_preds = extract_entities_spacy(eval_df["body_text"].fillna("").tolist(),
                                          model_name=config["ner"]["spacy_baseline_model"])
    spacy_report = evaluate_ner(gold, spacy_preds)
    display_df = pd.DataFrame(spacy_report).T
except (ImportError, OSError) as e:
    print(f"spaCy baseline unavailable in this environment: {e}")
    display_df = None
display_df

# %% [markdown]
# ## Fine-tuned BERT-NER (`dslim/bert-base-NER` + custom LAW/EVENT types)
# **Heavy step.** Requires `pip install transformers torch seqeval` +
# internet. See `src/ner_model.finetune_bert_ner()` for the model/label
# setup — you'll need to convert `entities_ref` character spans into
# token-level BIO tags before calling `Trainer.train()`. A minimal
# span-to-BIO helper:

# %%
def spans_to_bio(text: str, tokenizer, spans: list[dict]) -> list[str]:
    """Convert character-offset entity spans into BIO tags aligned to tokenizer offsets."""
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True)
    tags = ["O"] * len(encoding["input_ids"])
    for span in spans:
        # NOTE: this demo data's entities_ref does not always include
        # start/end offsets — if yours doesn't either, re-locate the span
        # text in `text` first (str.find) to recover offsets.
        start = text.find(span["text"])
        if start == -1:
            continue
        end = start + len(span["text"])
        started = False
        for i, (tok_start, tok_end) in enumerate(encoding["offset_mapping"]):
            if tok_start == tok_end:
                continue
            if tok_start >= start and tok_end <= end:
                tags[i] = f"{'B' if not started else 'I'}-{span['label']}"
                started = True
    return tags

# %%
# from src.ner_model import finetune_bert_ner
# tokenizer, model, training_args = finetune_bert_ner(config)
# ... build a tokenized Dataset using spans_to_bio() above, then:
# trainer = Trainer(model=model, args=training_args, train_dataset=..., eval_dataset=...)
# trainer.train()

# %% [markdown]
# Target from the project brief: **>0.79 NER-F1** (entity-level, strict
# span match) on the fine-tuned model — the rule-based/spaCy numbers above
# are baselines to beat, not the deliverable itself.
