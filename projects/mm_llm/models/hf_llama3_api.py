import os
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

from core.base_model import BaseModel
from core.registry import register_model


class HFLlama3Model(BaseModel):
    name = "hf-llama3-8b"

    def load(self):
        # 🔥 Load .env once here
        load_dotenv()

        hf_api_key = os.getenv("HF_API_KEY")
        if not hf_api_key:
            raise RuntimeError("HF_API_KEY not found in .env")

        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.client = InferenceClient(
            model=self.model_id,
            token=hf_api_key
        )

        self.system_prompt = (
            "You are a strict JSON-only assistant. "
            "Do not explain. Do not add markdown. "
            "Return ONLY valid JSON."
        )

        print("HF LLaMA-3 8B (API) loaded")

    def generate(self, prompt: str) -> str:
        retries = 1

        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.2
                )

                return response.choices[0].message.content

            except HfHubHTTPError as e:
                if e.response and e.response.status_code == 429:
                    if attempt < retries:
                        time.sleep(20)
                        continue
                    return None
                raise e


# 🔥 Register model
register_model(HFLlama3Model)
