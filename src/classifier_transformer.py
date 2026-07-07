from __future__ import annotations

import logging

import numpy as np

from src.baseline_models import build_dataset, evaluate_multilabel
from src.threshold_tuning import apply_thresholds, find_optimal_thresholds
from src.utils import load_config, set_global_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedBCETrainer:
    """
    Thin wrapper documenting the label-weighted BCEWithLogitsLoss setup.
    Plugs into a HuggingFace `Trainer` by overriding `compute_loss`.
    """

    def __init__(self, pos_weight: "np.ndarray"):
        import torch

        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        import torch
        from torch.nn import BCEWithLogitsLoss

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def compute_pos_weight(y_train: np.ndarray) -> np.ndarray:
    """Inverse label frequency, used to counter class imbalance in the BCE loss."""
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_counts = np.clip(pos_counts, 1, None)  # avoid div-by-zero for empty labels
    return neg_counts / pos_counts


def finetune_roberta_classifier(config: dict):
    """
    Full fine-tuning pipeline scaffold. Requires torch/transformers/datasets
    and internet access to pull `roberta-base` weights + tokenizer.
    """
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    cfg = config["transformer_classifier"]
    set_global_seed(config["seed"])

    ds = build_dataset(config)
    label_names = list(ds.mlb.classes_)
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tokenize(batch_texts: list[str]):
        return tokenizer(batch_texts, truncation=True, padding="max_length", max_length=cfg["max_length"])

    train_encodings = tokenize(ds.X_train_text)
    val_encodings = tokenize(ds.X_val_text)

    train_dataset = Dataset.from_dict({**train_encodings, "labels": ds.Y_train.tolist()})
    val_dataset = Dataset.from_dict({**val_encodings, "labels": ds.Y_val.tolist()})

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="models/roberta_classifier",
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        seed=config["seed"],
    )

    pos_weight = compute_pos_weight(ds.Y_train)
    weighted_trainer_mixin = WeightedBCETrainer(pos_weight)

    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return weighted_trainer_mixin.compute_loss(model, inputs, return_outputs)

    def compute_metrics(eval_pred):
        import torch

        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        # global 0.5 for the Trainer's running metric only; final numbers use
        # the per-label threshold search below.
        preds = (probs >= 0.5).astype(int)
        metrics = evaluate_multilabel(labels, preds, label_names)
        return {"f1_micro": metrics["micro_f1"], "f1_macro": metrics["macro_f1"]}

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting RoBERTa fine-tuning (%d epochs) — this needs a GPU for reasonable runtime.",
                cfg["epochs"])
    trainer.train()

    # Final evaluation with per-label F1-optimal thresholds (guideline #2).
    import torch

    val_logits = trainer.predict(val_dataset).predictions
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
    thresholds = find_optimal_thresholds(ds.Y_val, val_probs, label_names, tuple(cfg["threshold_search_grid"]))
    final_preds = apply_thresholds(val_probs, label_names, thresholds)
    final_metrics = evaluate_multilabel(ds.Y_val, final_preds, label_names)

    logger.info("Final (per-label-threshold) micro-F1=%.4f macro-F1=%.4f (target: >0.81 micro-F1)",
                final_metrics["micro_f1"], final_metrics["macro_f1"])

    model.save_pretrained("models/model_roberta_v1")
    tokenizer.save_pretrained("models/model_roberta_v1")
    return model, tokenizer, thresholds, final_metrics


if __name__ == "__main__":
    config = load_config()
    finetune_roberta_classifier(config)
