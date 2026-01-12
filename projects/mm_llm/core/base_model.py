from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Every model MUST follow this contract.
    Core system depends ONLY on this.
    """

    name: str  # unique model name

    @abstractmethod
    def load(self):
        """Load weights / connect to API"""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Take prompt, return text output"""
        pass
