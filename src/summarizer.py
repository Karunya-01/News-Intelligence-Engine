from __future__ import annotations

import logging
import re

import pandas as pd

from src.ner_model import extract_entities_rule_based
from src.utils import guard_against_leakage, load_config, resolve_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def extractive_baseline_summary(body: str, n_sentences: int = 3) -> str:
    """
    TF-IDF-weighted sentence-position extractive summary: scores each
    sentence by average TF-IDF weight of its terms (computed over the
    document's own sentences) plus a lead-bias, then keeps the top-N in
    their original order. Fully offline, no model download required.
    """
    sentences = _split_sentences(body)
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:n_sentences])

    scores = tfidf.sum(axis=1).A1
    lead_bias = [1.0 / (i + 1) for i in range(len(sentences))]
    combined = [0.7 * s + 0.3 * lb for s, lb in zip(scores, lead_bias)]

    top_indices = sorted(range(len(sentences)), key=lambda i: combined[i], reverse=True)[:n_sentences]
    top_indices.sort()
    return " ".join(sentences[i] for i in top_indices)


def enforce_entity_grounding(summary: str, article: str) -> tuple[str, bool]:
    """
    Approach doc constraint: summaries must include at least one Person,
    Organisation, or Location entity from the source article. If the
    generated summary has none, append the highest-confidence article
    entity as a lightweight repair (a placeholder for a proper constrained
    decoding setup in the transformer pipeline).
    """
    grounding_types = {"PERSON", "ORG", "GPE", "LOC"}
    summary_entities = {e["text"].lower() for e in extract_entities_rule_based(summary)}
    article_entities = [e for e in extract_entities_rule_based(article) if e["label"] in grounding_types]

    is_grounded = any(e["text"].lower() in summary_entities for e in article_entities) if article_entities else True
    if is_grounded or not article_entities:
        return summary, is_grounded

    repaired = f"{summary} This involves {article_entities[0]['text']}."
    return repaired, False


def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not installed — skipping ROUGE. `pip install rouge-score`.")
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(scores[k].fmeasure)
    return {k: sum(v) / len(v) if v else 0.0 for k, v in agg.items()}


def bert_score_f1(predictions: list[str], references: list[str]) -> float:
    try:
        from bert_score import score as bertscore
    except ImportError:
        logger.warning("bert-score not installed — skipping BERTScore. `pip install bert-score`.")
        return float("nan")
    _, _, f1 = bertscore(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def finetune_seq2seq(config: dict):
    """
    Fine-tunes facebook/bart-base (or t5-small) on (model_input, summary_ref)
    pairs. Heavy job — run with internet + GPU. See notebooks/06_summarization.py
    for the full worked Trainer setup including the entity-grounding loss
    penalty and generation-time constrained decoding hook.
    """
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainingArguments,
    )

    cfg = config["summarization"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])

    training_args = Seq2SeqTrainingArguments(
        output_dir="models/summarizer",
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        predict_with_generate=True,
        generation_max_length=cfg["max_target_length"],
        generation_num_beams=cfg["num_beams"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=config["seed"],
    )
    logger.info(
        "finetune_seq2seq() is a scaffold — plug in a tokenized (article, "
        "summary) Dataset and call Seq2SeqTrainer(...).train(). "
        "See notebooks/06_summarization.py."
    )
    return tokenizer, model, training_args


def run(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    eval_path = resolve_path(config["paths"]["eval_summary_csv"])
    df = pd.read_csv(eval_path)
    guard_against_leakage(df, feature_columns=["headline", "body_text"], config=config)

    predictions, grounded_flags = [], []
    for body in df["body_text"].fillna(""):
        summary = extractive_baseline_summary(body, n_sentences=3)
        grounded_summary, was_already_grounded = enforce_entity_grounding(summary, body)
        predictions.append(grounded_summary)
        grounded_flags.append(was_already_grounded)

    references = df["summary_ref"].fillna("").tolist()
    rouge = rouge_scores(predictions, references)
    bscore = bert_score_f1(predictions, references) if any(references) else float("nan")

    logger.info("Extractive baseline (offline) — ROUGE: %s | BERTScore F1: %.4f | entity-grounded: %d/%d",
                rouge, bscore if bscore == bscore else -1, sum(grounded_flags), len(grounded_flags))

    out_dir = resolve_path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = df[["article_id", "headline"]].copy()
    result["generated_summary"] = predictions
    result["reference_summary"] = references
    result["was_already_entity_grounded"] = grounded_flags
    result.to_csv(out_dir / "summarization_baseline_results.csv", index=False)
    logger.info("Wrote sample summaries to artifacts/summarization_baseline_results.csv")
    return result


if __name__ == "__main__":
    run()
