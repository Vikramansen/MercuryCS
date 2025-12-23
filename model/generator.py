import os
from typing import List, Optional
from deep_translator import GoogleTranslator
from langdetect import detect

class Generator:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.response_template = self._load_prompt("response.txt")
        self.fallback_template = self._load_prompt("fallback.txt")

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
        prompt = self.response_template.format(context=context_str, query=query)

        # SIMULATED LLM CALL
        # In a real system, this would call OpenAI/Gemini/etc.
        # Here we simulate a grounded response.
        
        response = f"[Generated Answer based on context]: Based on the information provided, {context[0][:100]}..."
        
        # If we want to be slightly more realistic without an LLM:
        # We can just return the most relevant chunk as the answer.
        response = f"Here is the relevant information: {context[0]}"

        if target_lang != "en":
            response = self.translate(response, target_lang)
            
        return response

if __name__ == "__main__":
    gen = Generator(prompts_dir="../prompts")
    print(gen.generate("shipping time", ["Standard shipping takes 3-5 days."], "es"))
