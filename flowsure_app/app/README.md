# FlowSure Intent Classifier

AI-powered Streamlit app that classifies customer support messages into 27 intents for FlowSure insurance.

## How it works

```
Customer text → ONNX model (sklearn pipeline) → intent label + confidence scores
                                               → intent_mapping.json → category + priority
```

The model is a scikit-learn TF-IDF + classifier pipeline exported to ONNX format. Given raw text it returns:
- **Intent** — one of 27 support intents (e.g. `cancel_order`, `payment_issue`)
- **Category** — high-level group (ACCOUNT, ORDER, PAYMENT, REFUND, …)
- **Priority** — `high` / `medium` / `low` routing signal
- **Confidence** — probability score (0–1) for the predicted intent
- **Top 3** — the three most likely intents with scores

## Installation

```bash
cd flowsure_app
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## App layout

| Tab | Description |
|-----|-------------|
| 🎯 Classify Ticket | Single message classification with confidence meter, top-3 chart, and priority badge |
| 📊 Batch Classification | Upload CSV (`text` column) or paste lines; get category/priority charts + downloadable results |

## Cloud Model — RAG Response Generator

Tab 3 adds a Retrieval-Augmented Generation (RAG) pipeline that suggests a draft reply for any customer message.

```
Customer text → SentenceTransformer (all-MiniLM-L6-v2)
             → FAISS index (50,000 vectors)
             → top-3 similar historical tickets
             → Claude Sonnet (LLM) or best-match fallback
             → suggested response
```

### Setup

Copy `.env.example` to `.env` and add your Anthropic API key:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

The app works without a key — it falls back to the best-matching historical support answer.

## Full Pipeline (Tab 3 — bottom section)

The "Full Pipeline" section runs both models on one input and shows the combined output:

| Step | Model | Output |
|------|-------|--------|
| 1. Edge | ONNX (TF-IDF + classifier) | Intent, Category, Priority |
| 2. Cloud | FAISS + Claude Sonnet | Suggested response |

## File structure

```
flowsure_app/
├── app.py                   # Streamlit UI (3 tabs)
├── classifier.py            # IntentClassifier (ONNX inference)
├── rag_engine.py            # RAGEngine (FAISS + LLM)
├── requirements.txt
├── .env                     # API key (not committed)
├── .env.example             # Template
└── models/
    ├── baseline_model.onnx  # Trained model (1.53 MB)
    ├── intent_mapping.json  # intent → [category, priority]
    ├── faiss_index.bin      # FAISS vector index (50K embeddings)
    └── texts.parquet        # customer_text + support_text + company
```
