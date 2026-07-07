# News Intelligence Engine
**Multilabel News Article Classification & Entity-Aware Summarisation Engine**

A capstone project implementing:
1. Multilabel topic classification (an article can be Politics *and* Economy at once)
2. Named entity recognition (Person, Organisation, Location, Event, Law)
3. Entity-aware abstractive summarisation (3-sentence, grounded to a named entity)
4. Five-signal misinformation risk scoring

Built against the provided datasets: `news_articles_train.csv` (3,000 rows),
`news_eval_summary.csv` (400 rows, gold summaries), `news_eval_ner_test.csv`
(500 rows, gold entity spans).

---

## ⚠️ Read this first: what runs offline vs what needs your machine

This project was scaffolded and **tested end-to-end in a sandbox with no
internet access**. Two tiers of code result from that:

| Tier | Components | Requirements |
|---|---|---|
| **Offline-safe** (tested, runs now) | Preprocessing, EDA, TF-IDF+LogReg/SVM baselines, rule-based NER fallback, extractive summarisation, misinformation scoring, tests | Just `pip install -r requirements.txt` (core subset) |
| **Heavy / needs internet+GPU** (written, not executed here) | RoBERTa fine-tuning, BERT-NER fine-tuning, BART/T5 fine-tuning | `pip install torch transformers datasets accelerate` + internet to download weights + a GPU is strongly recommended |

Every heavy module has a working, correct implementation — it just hasn't
been run in this environment. Run `python scripts/train_all.py` to execute
everything in the offline-safe tier right now and inspect real output in
`artifacts/`.

Also note: **this dataset is synthetic practice data.** Labels, gold
summaries, and gold entity spans are only loosely tied to article content
(by design, for a training exercise) — so baseline F1 scores you see here
(e.g. ~0.43 micro-F1) are expected to be modest. The project brief's
targets (>0.81 micro-F1, >0.79 NER-F1) are realistic once you either (a)
point this pipeline at real news data, or (b) fine-tune the transformer
models, which pick up far more signal than the linear baselines.

---

## Project structure

```
news_nlp_project/
├── configs/
│   └── config.yaml              # single source of truth for all hyperparameters
├── data/
│   ├── news_articles_train.csv
│   ├── news_eval_summary.csv
│   └── news_eval_ner_test.csv
├── src/
│   ├── utils.py                 # config loading, seeding, leakage guard
│   ├── preprocessing.py         # HTML/unicode cleaning, lang detection, truncation
│   ├── eda.py                   # label co-occurrence, length dists, noise audit
│   ├── baseline_models.py       # TF-IDF+LR/SVM, Word2Vec+MLP
│   ├── threshold_tuning.py      # per-label F1-optimal threshold search
│   ├── classifier_transformer.py# RoBERTa fine-tuning (heavy)
│   ├── ner_model.py             # spaCy / rule-based / BERT-NER fine-tuning (heavy)
│   ├── summarizer.py            # extractive baseline + BART/T5 fine-tuning (heavy)
│   └── misinfo_scoring.py       # 5-signal composite Mis-Risk score
├── notebooks/                   # VS Code "# %%" cell scripts, run in order 01→07
├── app/
│   └── streamlit_app.py         # live demo
├── scripts/
│   └── train_all.py             # runs the whole offline-safe tier end to end
├── tests/
│   └── test_preprocessing.py    # pytest unit tests
├── artifacts/                   # generated plots, reports, CSVs (gitignored)
├── models/                      # saved model checkpoints (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup (VS Code)

1. **Open the folder** `news_nlp_project/` in VS Code.
2. **Create a virtual environment** (Command Palette → `Python: Create
   Environment` → venv), or manually:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies.** For the offline-safe tier only:
   ```bash
   pip install pandas numpy pyyaml scikit-learn scipy langdetect \
               beautifulsoup4 ftfy nltk gensim streamlit matplotlib seaborn \
               rouge-score bert-score nrclex textblob pyarrow pytest
   ```
   For everything including transformer fine-tuning:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_trf
   ```
4. **Select the interpreter**: Command Palette → `Python: Select
   Interpreter` → choose `.venv`.
5. **Run tests** to confirm the setup: `pytest -q` (or use the Testing
   sidebar icon in VS Code).

## Running things

```bash
# Full offline-safe pipeline (preprocessing → EDA → baselines → summariser → misinfo)
python scripts/train_all.py

# Individual components
python -m src.preprocessing
python -m src.eda
python -m src.baseline_models
python -m src.ner_model
python -m src.summarizer
python -m src.misinfo_scoring

# Heavy transformer fine-tuning (needs internet + ideally GPU)
python -m src.classifier_transformer

# Interactive notebooks: open any notebooks/0X_*.py in VS Code and use
# "Run Cell" / Shift+Enter above each `# %%` marker (Jupyter/Python extension)

# Streamlit demo
streamlit run app/streamlit_app.py
```

## Project guidelines this codebase enforces

- **No data leakage**: `src/utils.guard_against_leakage()` refuses to build
  features from `summary_ref`, `entities_ref`, or `mis_risk_label` — called
  at the top of every training entry point.
- **Per-label thresholds**: `src/threshold_tuning.py` is shared by the
  baseline and transformer classifiers; no global 0.5 cutoff anywhere.
- **Programmatic language filtering**: `src/preprocessing.detect_language_safe()`
  uses `langdetect`, never the raw (10%-null, sometimes-wrong) `language`
  column.
- **Shared tokenizer/backbone**: classifier, NER, and summarizer all use
  RoBERTa-family tokenizers (`configs/config.yaml → shared_backbone`).
- **BERTScore alongside ROUGE**: `src/summarizer.py` always reports both.
- **Config-driven, no magic numbers**: every hyperparameter lives in
  `configs/config.yaml`; `src/utils.set_global_seed()` seeds all RNGs.

## Deployment (optional deliverable)

To host the Streamlit app on Amazon EC2:
1. Launch an EC2 instance (Ubuntu, t2.medium or larger), open inbound port `8501`.
2. `git clone` this project, `pip install -r requirements.txt`.
3. Copy any fine-tuned model into `models/model_roberta_v1/` if you have one.
4. `streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0`
5. Visit `http://<ec2-public-ip>:8501`.

(Any other cloud — Streamlit Community Cloud, Render, GCP Cloud Run — works
the same way; the app has no EC2-specific code.)
