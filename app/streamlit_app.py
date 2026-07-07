from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.baseline_models import build_dataset, train_tfidf_logreg
from src.misinfo_scoring import composite_mis_risk_score
from src.ner_model import extract_entities_rule_based
from src.summarizer import enforce_entity_grounding, extractive_baseline_summary
from src.utils import load_config

st.set_page_config(page_title="News Intelligence Engine", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    config = load_config()
    model_dir = Path("models/model_roberta_v1")

    if model_dir.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return {"kind": "transformer", "model": model, "tokenizer": tokenizer}, config

    # Fallback: train the lightweight baseline on the fly.
    ds = build_dataset(config)
    vectorizer, clf, metrics = train_tfidf_logreg(ds, config)
    return {"kind": "baseline", "vectorizer": vectorizer, "model": clf,
            "label_names": list(ds.mlb.classes_), "metrics": metrics}, config


def predict_topics(text: str, bundle: dict) -> list[tuple[str, float]]:
    if bundle["kind"] == "baseline":
        proba = bundle["model"].predict_proba(bundle["vectorizer"].transform([text]))[0]
        return sorted(zip(bundle["label_names"], proba), key=lambda x: -x[1])
    else:
        import torch

        tokenizer, model = bundle["tokenizer"], bundle["model"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        proba = torch.sigmoid(logits)[0].numpy()
        label_names = [model.config.id2label[i] for i in range(len(proba))]
        return sorted(zip(label_names, proba), key=lambda x: -x[1])


def main() -> None:
    st.title("📰 News Intelligence Engine")
    st.caption("Multilabel classification · NER · entity-aware summarisation · misinformation scoring")

    bundle, config = load_models()
    if bundle["kind"] == "baseline":
        st.info(
            "No fine-tuned RoBERTa checkpoint found at `models/model_roberta_v1` — "
            "using the TF-IDF + LogisticRegression baseline instead. "
            "Run `python -m src.classifier_transformer` (needs internet + GPU) to use the full model.",
            icon="ℹ️",
        )

    col1, col2 = st.columns(2)
    with col1:
        headline = st.text_input("Headline", "Central Bank raises interest rates amid inflation fears")
        source_domain = st.text_input("Source domain", "bbc.com")
    with col2:
        body_text = st.text_area(
            "Article body",
            "The central bank announced a 0.5% interest rate hike on Tuesday, citing "
            "persistent inflation pressures. Finance Minister Elena Cruz said the move "
            "was necessary to stabilise prices. Markets reacted negatively, with the "
            "stock index falling 2% in early trading. Analysts at Whitmore & Chen "
            "warned that further hikes could slow economic growth.",
            height=180,
        )

    if st.button("Analyse article", type="primary"):
        with st.spinner("Analysing..."):
            full_text = f"{headline}. {body_text}"

            st.subheader("🏷️ Topic labels")
            topics = predict_topics(full_text, bundle)
            topic_df = pd.DataFrame(topics, columns=["Label", "Probability"])
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

            st.subheader("🧾 Named entities")
            entities = extract_entities_rule_based(body_text)
            if entities:
                st.dataframe(pd.DataFrame(entities), use_container_width=True, hide_index=True)
            else:
                st.write("No entities detected by the rule-based fallback extractor.")

            st.subheader("📝 Entity-aware summary")
            raw_summary = extractive_baseline_summary(body_text, n_sentences=3)
            grounded_summary, was_grounded = enforce_entity_grounding(raw_summary, body_text)
            st.write(grounded_summary)
            st.caption("✅ Entity-grounded" if was_grounded else "⚠️ Entity grounding was auto-repaired")

            st.subheader("⚠️ Misinformation risk signals")
            row = pd.Series({"headline": headline, "body_text": body_text, "source_domain": source_domain})
            signals = composite_mis_risk_score(
                row,
                config["misinfo_scoring"]["weights"],
                config["misinfo_scoring"]["credible_domain_whitelist"],
            )
            risk_score = signals.pop("mis_risk_score")
            st.metric("Composite Mis-Risk Score", f"{risk_score:.2f}", help="0 = low risk, 1 = high risk")
            st.bar_chart(pd.Series(signals))


if __name__ == "__main__":
    main()
