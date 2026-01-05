import os
from typing import List, Optional
from deep_translator import GoogleTranslator
from langdetect import detect
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Generator:
    def __init__(self, prompts_dir: str = "prompts", model_name: str = "google/flan-t5-small"):
        self.prompts_dir = prompts_dir
        self.response_template = self._load_prompt("response.txt")
        self.fallback_template = self._load_prompt("fallback.txt")
        
        print(f"Loading generator model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model loaded.")

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.prompts_dir, filename)
        with open(path, 'r') as f:
            return f.read()

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except:
            return "en"

    def translate(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text
        try:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def generate(self, query: str, context: List[str], target_lang: str = "en") -> str:
        # Hard grounding check
        if not context:
            return self.translate(self.fallback_template, target_lang)

        context_str = "\n\n".join(context)
        # Flan-T5 works best with a simple instruction format
        # modifying the template logic slightly to fit the model better if needed,
        # but for now, we'll construct a prompt that fits the model's instruction tuning.
        
        # Construct input text for the model
        input_text = f"Answer based on context:\n{context_str}\n\nQuestion: {query}"
        
        # Tokenize and generate
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=128, 
                num_beams=4, 
                early_stopping=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if target_lang != "en":
            response = self.translate(response, target_lang)
            
        return response

if __name__ == "__main__":
    # Ensure correct path when running directly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_path = os.path.join(base_dir, "prompts")
    
    gen = Generator(prompts_dir=prompts_path)
    print("Test Generation:")
    print(gen.generate("shipping time", ["Standard shipping takes 3-5 days."], "en"))
