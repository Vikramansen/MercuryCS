import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, util

class FaithfulnessEvaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def evaluate(self, answer: str, context: list[str]) -> float:
        """
        Evaluates faithfulness by checking semantic similarity between answer and context.
        In a production system, this would use an LLM-as-a-judge (e.g., GPT-4) to verify 
        if the answer is logically entailed by the context.
        """
        if not context:
            return 0.0
        
        # Combine context chunks
        full_context = " ".join(context)
        
        # Compute cosine similarity
        embeddings = self.model.encode([answer, full_context])
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        return score

if __name__ == "__main__":
    evaluator = FaithfulnessEvaluator()
    
    # Test case 1: Faithful
    ctx = ["Standard shipping takes 3-5 business days."]
    ans = "You can expect your delivery in 3-5 business days."
    score = evaluator.evaluate(ans, ctx)
    print(f"Faithful Score: {score:.4f}")
    
    # Test case 2: Hallucination
    ctx = ["We do not offer refunds."]
    ans = "You can get a full refund within 30 days."
    score = evaluator.evaluate(ans, ctx)
    print(f"Hallucination Score: {score:.4f}")
