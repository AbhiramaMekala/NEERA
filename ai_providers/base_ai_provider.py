from abc import ABC, abstractmethod

class BaseAIProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generates text from a given prompt."""
        pass
