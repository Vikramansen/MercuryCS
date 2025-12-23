import os
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, docs_dir: str = "data/faq_docs", model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs_dir = docs_dir
        self.chunks, self.chunk_embeddings = self._load_and_embed_docs()

    def _load_and_embed_docs(self) -> Tuple[List[str], np.ndarray]:
        chunks = []
        for filename in os.listdir(self.docs_dir):
            if filename.endswith(".txt"):
                path = os.path.join(self.docs_dir, filename)
                with open(path, 'r') as f:
                    content = f.read()
                    # Simple splitting by double newline (paragraphs)
                    file_chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
                    chunks.extend(file_chunks)
        
        if not chunks:
            return [], np.array([])
            
        embeddings = self.model.encode(chunks)
        return chunks, embeddings

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if len(self.chunks) == 0:
            return []
            
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Optional: Add a relevance threshold here too
            if similarities[idx] > 0.3:
                results.append(self.chunks[idx])
        
        return results

if __name__ == "__main__":
    retriever = Retriever(docs_dir="../data/faq_docs")
    results = retriever.retrieve("How long does shipping take?")
    for r in results:
        print(f"- {r}")
