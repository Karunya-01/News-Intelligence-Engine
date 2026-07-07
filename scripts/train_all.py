"""
scripts/train_all.py
Runs every offline-safe component of the pipeline end to end, in order,
and writes all artifacts to artifacts/. This intentionally SKIPS the heavy
transformer fine-tuning steps (RoBERTa classifier, BERT-NER, BART/T5
summarizer) since those need internet + ideally a GPU — run those
separately via:

    python -m src.classifier_transformer
    python -m src.ner_model        (see finetune_bert_ner)
    python -m src.summarizer       (see finetune_seq2seq)

Usage:
    python scripts/train_all.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baseline_models, eda, misinfo_scoring, preprocessing, summarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== [1/5] Preprocessing ===")
    preprocessing.run()

    logger.info("=== [2/5] EDA ===")
    eda.run()

    logger.info("=== [3/5] Baseline classification models ===")
    baseline_models.run()

    logger.info("=== [4/5] Extractive summarisation baseline ===")
    summarizer.run()

    logger.info("=== [5/5] Misinformation scoring ===")
    misinfo_scoring.run()

    logger.info("Done. See artifacts/ for all outputs.")
    logger.info(
        "Heavy transformer components (RoBERTa classifier, BERT-NER, "
        "BART/T5 summarizer) are not run automatically — see notebooks/ "
        "04, 05 (fine-tune section) and 06 (fine-tune section)."
    )


if __name__ == "__main__":
    main()
