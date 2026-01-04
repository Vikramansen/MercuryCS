# MercuryCS

A multilingual, RAG-based customer support agent designed for e-commerce.

![System Architecture](architecture.png)

## Overview

MercuryCS handles customer queries in any language by translating them to English, retrieving relevant context from a knowledge base, and generating a grounded response. It prioritizes **correctness** and **safety** over conversational flow, using strict intent classification and hard grounding to prevent hallucinations.

**Tech Stack:** FastAPI, SentenceTransformers (all-MiniLM-L6-v2), LangDetect, DeepTranslator.

## Quick Start

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the API:**
    ```bash
    uvicorn api.main:app --reload
    ```

3.  **Test the endpoint:**
    ```bash
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"query": "Where is my order?", "user_id": "test_user"}'
    ```

## Evaluation

We track faithfulness, latency, and fallback rates to ensure reliability.

Run the evaluation suite:
```bash
python run_evals.py
```

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for the latest performance metrics.

## Technical Decisions

*   **Translate-Process-Translate**: We translate all inputs to English to maintain a single, high-quality knowledge base and embedding space. This adds ~400ms latency but significantly reduces complexity.
*   **Embedding-Based Classification**: We use cosine similarity against a set of canonical examples for intent classification. It's faster and more deterministic than LLM-based classification for this specific domain.
*   **Hard Grounding**: If the retrieval score is below threshold, the system refuses to answer. This is a deliberate choice to avoid liability in e-commerce scenarios.
