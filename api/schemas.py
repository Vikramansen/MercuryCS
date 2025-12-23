from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    detected_language: str
    intent: str
    confidence: float
    retrieved_context: List[str] = []
    latency_ms: float
