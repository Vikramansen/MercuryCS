# MercuryCS

**MercuryCS** is a multilingual, evaluation-driven conversational AI system for e-commerce customer support, designed to mirror TikTok Shopping use cases.

## Architecture

```mermaid
graph TD
    User[User] -->|Query (Any Lang)| API[FastAPI /chat]
    API -->|Detect Lang| LangDetect[Language Detector]
    LangDetect -->|Translate to EN| Translator[Translator]
    Translator -->|Query (EN)| Classifier[Intent Classifier]
    
    Classifier -->|Intent: Unsupported| Fallback[Fallback Generator]
    Classifier -->|Intent: Supported| Retriever[RAG Retriever]
    
    Retriever -->|Retrieve Context| KnowledgeBase[(FAQ Docs)]
    KnowledgeBase -->|Context Chunks| Generator[Response Generator]
    
    Generator -->|Generate Answer (EN)| TranslatorBack[Translator]
    Fallback -->|Refusal (EN)| TranslatorBack
    
    TranslatorBack -->|Response (Original Lang)| API
    API -->|Final Response| User
```

## Key Features
- **Intent Classification**: Strictly categorizes queries into `product_inquiry`, `order_issue`, `return_refund`, or `unsupported`.
- **Hard Grounding**: Rejects responses if no relevant context is retrieved. No "best guess" or hallucinations.
- **Multilingual Flow**: Seamlessly handles non-English queries by translating to English for processing and translating the response back.
- **Evaluation-First**: Built with a suite of evaluation scripts to measure faithfulness, latency, and fallback rates.

## Tradeoffs
1.  **Translation Latency vs. Native Support**: We use a translate-process-translate approach.
    *   *Pro*: Allows using a single, high-quality English embedding model and knowledge base. Simplifies maintenance.
    *   *Con*: Adds latency (2x translation calls) and potential translation errors.
2.  **Strict Grounding vs. Helpfulness**: We prioritize correctness over conversational flow.
    *   *Pro*: Eliminates hallucinations and liability (critical for e-commerce).
    *   *Con*: System may seem "dumber" if it refuses to answer slightly ambiguous queries that a human could guess.
3.  **Embedding-Based Classification vs. LLM Classification**:
    *   *Pro*: Much faster and cheaper. Deterministic behavior for known examples.
    *   *Con*: Less flexible with nuance than a large LLM.

## Failure Cases
1.  **Translation Loss**: If the user's query loses meaning during translation (e.g., slang, idioms), the intent classifier may fail.
    *   *Mitigation*: Use higher quality translation models or fine-tune on domain-specific multilingual data.
2.  **Context Splitting**: If the answer spans multiple chunks that aren't retrieved together, the generator may miss the full answer.
    *   *Mitigation*: Implement sliding window chunking or hierarchical retrieval.
3.  **Ambiguous Intents**: Queries like "Can I change it?" could be an order issue or a return.
    *   *Mitigation*: Implement multi-turn dialogue to ask clarifying questions (not in current scope).

## Why Evaluation Matters More Than Model Size
In production systems, **reliability is the product**. A 100B parameter model that hallucinates a return policy 1% of the time is worse than a 1B parameter model that admits ignorance.
- **Faithfulness**: We measure if the answer is strictly derived from the retrieved docs.
- **Fallback Rate**: We track how often we safely refuse. A high fallback rate on supported intents indicates retrieval failure; a low fallback rate on unsupported intents indicates safety failure.
- **Latency**: E-commerce users are impatient. We optimize for p99 latency to ensure consistent experience.

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run API
```bash
uvicorn api.main:app --reload
```

### Run Evaluations
```bash
python eval/faithfulness.py
python eval/latency.py
python eval/fallback_rate.py
```
