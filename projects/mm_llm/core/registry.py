from typing import Dict, Type
from core.base_model import BaseModel

_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(model_cls: Type[BaseModel]):
    """
    Register a model class.
    Called once per model.
    """
    _MODEL_REGISTRY[model_cls.name] = model_cls

def list_models():
    return list(_MODEL_REGISTRY.keys())

def load_model(name: str) -> BaseModel:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found")

    model = _MODEL_REGISTRY[name]()
    model.load()
    return model
