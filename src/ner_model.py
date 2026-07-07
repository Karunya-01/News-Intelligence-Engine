from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import pandas as pd

from src.utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. spaCy baseline (requires `pip install spacy` + model download)
# ---------------------------------------------------------------------------
def extract_entities_spacy(texts: list[str], model_name: str = "en_core_web_trf") -> list[list[dict]]:
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy is not installed. `pip install spacy` and "
            f"`python -m spacy download {model_name}` to use this baseline."
        ) from e
    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise OSError(
            f"spaCy model '{model_name}' not found locally. "
            f"Run: python -m spacy download {model_name}"
        ) from e

    results = []
    for doc in nlp.pipe(texts):
        results.append([{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                         for ent in doc.ents])
    return results


# ---------------------------------------------------------------------------
# 2. Dependency-free rule-based fallback (offline-safe baseline)
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}\b"
)
_MONEY_RE = re.compile(r"\$\d[\d,.]*\s?(?:B|M|K|billion|million|thousand)?", re.IGNORECASE)
_ORG_SUFFIX_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+){0,3}\s"
    r"(?:Inc|Group|Ministry|Department|Police Department|Organisation|Organization|Bank|University))\b"
)
_LAW_RE = re.compile(r"\b(The\s)?[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,4}\s(?:Act|Bill)\s\d{4}\b")
_PERSON_RE = re.compile(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b")
_GPE_RE = re.compile(r"\b([A-Z][a-z]+(?:town|ville|haven|stad|burg|land|city)?,\s[A-Z][a-zA-Z\s]+)\b")


def extract_entities_rule_based(text: str) -> list[dict]:
    """
    Lightweight heuristic NER used only as an offline-safe fallback / smoke
    test. NOT a substitute for the spaCy or fine-tuned BERT-NER models —
    those are the deliverables that count toward the >0.79 NER-F1 target.
    """
    entities = []
    for pattern, label in [
        (_DATE_RE, "DATE"),
        (_MONEY_RE, "MONEY"),
        (_LAW_RE, "LAW"),
        (_ORG_SUFFIX_RE, "ORG"),
        (_GPE_RE, "GPE"),
        (_PERSON_RE, "PERSON"),
    ]:
        for m in pattern.finditer(text):
            entities.append({"text": m.group(0).strip(), "label": label, "start": m.start(), "end": m.end()})
    # de-duplicate overlapping spans, keep first match
    entities = sorted(entities, key=lambda e: e["start"])
    deduped, last_end = [], -1
    for ent in entities:
        if ent["start"] >= last_end:
            deduped.append(ent)
            last_end = ent["end"]
    return deduped


# ---------------------------------------------------------------------------
# 3. Entity-level evaluation (strict span match)
# ---------------------------------------------------------------------------
def evaluate_ner(gold_entities: list[list[dict]], pred_entities: list[list[dict]]) -> dict:
    """
    Entity-level precision / recall / F1 per type, STRICT match on
    (normalised text, label) pairs — a pragmatic proxy for strict span
    match when gold/pred come from different tokenizers with different
    character offsets.
    """
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gold_list, pred_list in zip(gold_entities, pred_entities):
        gold_set = {(e["text"].strip().lower(), e["label"]) for e in gold_list}
        pred_set = {(e["text"].strip().lower(), e["label"]) for e in pred_list}

        for item in pred_set:
            label = item[1]
            if item in gold_set:
                stats[label]["tp"] += 1
            else:
                stats[label]["fp"] += 1
        for item in gold_set:
            if item not in pred_set:
                stats[item[1]]["fn"] += 1

    report = {}
    all_tp = all_fp = all_fn = 0
    for label, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        all_tp += tp
        all_fp += fp
        all_fn += fn

    micro_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    micro_r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    report["_micro_avg"] = {"precision": micro_p, "recall": micro_r, "f1": micro_f1}
    return report


def load_gold_entities(df: pd.DataFrame, column: str = "entities_ref") -> list[list[dict]]:
    gold = []
    for raw in df[column]:
        if not isinstance(raw, str):
            gold.append([])
            continue
        try:
            gold.append(json.loads(raw))
        except json.JSONDecodeError:
            gold.append([])
    return gold


# ---------------------------------------------------------------------------
# 4. Fine-tuning entry point (requires transformers/torch + internet + GPU)
# ---------------------------------------------------------------------------
def finetune_bert_ner(config: dict):
    """
    Fine-tunes dslim/bert-base-NER with two additional custom entity types
    (LAW, EVENT) as required by the approach doc. This is a heavy job —
    run it on a machine with a GPU and internet access; it is not executed
    automatically by this repo's test suite.

    Expected data format: token-level BIO tags derived from `entities_ref`
    spans. See notebooks/05_ner.py for the full worked example including
    span-to-BIO conversion and the Hugging Face Trainer setup.
    """
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    cfg = config["ner"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer_model_name"])
    base_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    custom_labels = []
    for t in cfg["custom_entity_types"]:
        custom_labels += [f"B-{t}", f"I-{t}"]
    label_list = base_labels + custom_labels
    label2id = {l: i for i, l in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        cfg["transformer_model_name"],
        num_labels=len(label_list),
        id2label={i: l for l, i in label2id.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="models/ner_bert",
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=config["seed"],
    )
    logger.info(
        "finetune_bert_ner() is a scaffold — plug in a tokenized Dataset and "
        "call Trainer(...).train(). See notebooks/05_ner.py for the full pipeline."
    )
    return tokenizer, model, training_args


def run(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    eval_path = resolve_path(config["paths"]["eval_ner_csv"])
    df = pd.read_csv(eval_path)

    gold = load_gold_entities(df)
    preds = [extract_entities_rule_based(t) for t in df["body_text"].fillna("")]

    report = evaluate_ner(gold, preds)
    logger.info("Rule-based NER baseline (offline smoke test) — micro F1: %.4f",
                report["_micro_avg"]["f1"])
    for label, m in report.items():
        if label == "_micro_avg":
            continue
        logger.info("  %-10s P=%.3f R=%.3f F1=%.3f support=%d", label, m["precision"], m["recall"], m["f1"], m["support"])

    out_dir = resolve_path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ner_rule_based_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    run()
