import os
from dotenv import load_dotenv
from google import genai

from core.base_model import BaseModel
from core.registry import register_model

class Gemini25Model(BaseModel):
    name = "gemini-2.5"

    def load(self):
        load_dotenv()  # load .env

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in .env")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

        print("Gemini 2.5 model loaded")

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


# 🔥 Register model (one line)
register_model(Gemini25Model)
