from core.base_model import BaseModel
from core.registry import register_model

class DummyModel(BaseModel):
    name = "dummy-model"

    def load(self):
        print("Dummy model loaded")

    def generate(self, prompt: str) -> str:
        return f"[Dummy reply to]: {prompt}"

# 🔥 REGISTER MODEL (one line)
register_model(DummyModel)
