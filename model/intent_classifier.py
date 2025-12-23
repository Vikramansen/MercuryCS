import json
import os
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class IntentClassifier:
    def __init__(self, intents_path: str = "data/intents.json", model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.intents = self._load_intents(intents_path)
        self.embeddings, self.labels = self._embed_intents()
        self.confidence_threshold = 0.6 # Strict threshold

    def _load_intents(self, path: str) -> List[Dict]:
        with open(path, 'r') as f:
            data = json.load(f)
        return data['intents']

    def _embed_intents(self) -> Tuple[np.ndarray, List[str]]:
        examples = []
        labels = []
        for intent in self.intents:
            for example in intent['examples']:
                examples.append(example)
                labels.append(intent['name'])
        
        embeddings = self.model.encode(examples)
        return embeddings, labels

    def predict(self, query: str) -> Dict[str, any]:
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.labels[best_idx]

        if best_score < self.confidence_threshold:
            return {"intent": "fallback", "confidence": float(best_score)}
        
        return {"intent": best_intent, "confidence": float(best_score)}

if __name__ == "__main__":
    # Simple test
    classifier = IntentClassifier(intents_path="../data/intents.json")
    print(classifier.predict("I want to return my shoes"))
    print(classifier.predict("What is the weather?"))
