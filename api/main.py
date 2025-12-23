import time
import uvicorn
from fastapi import FastAPI, HTTPException
from api.schemas import ChatRequest, ChatResponse
from model.intent_classifier import IntentClassifier
from model.retriever import Retriever
from model.generator import Generator

app = FastAPI(title="MercuryCS API")

# Initialize components
# In a real app, we might want to lazy load or use dependency injection
print("Loading models...")
classifier = IntentClassifier()
retriever = Retriever()
generator = Generator()
print("Models loaded.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    # 1. Detect Language
    original_query = request.query
    lang = generator.detect_language(original_query)
    
    # 2. Translate to English if needed
    if lang != "en":
        query_en = generator.translate(original_query, "en")
    else:
        query_en = original_query
        
    # 3. Classify Intent
    intent_result = classifier.predict(query_en)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    
    retrieved_docs = []
    
    # 4. Handle Intent
    if intent == "unsupported" or intent == "fallback":
        response_en = generator.generate(query_en, [], "en") # Will trigger fallback
    else:
        # 5. Retrieve Context
        retrieved_docs = retriever.retrieve(query_en)
        
        # 6. Generate Response
        response_en = generator.generate(query_en, retrieved_docs, "en")
        
    # 7. Translate Response back
    if lang != "en":
        final_response = generator.translate(response_en, lang)
    else:
        final_response = response_en
        
    end_time = time.time()
    latency = (end_time - start_time) * 1000
    
    return ChatResponse(
        response=final_response,
        detected_language=lang,
        intent=intent,
        confidence=confidence,
        retrieved_context=retrieved_docs,
        latency_ms=latency
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
