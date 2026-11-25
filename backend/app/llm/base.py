from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

class BaseLLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider, e.g. 'groq' or 'ollama'."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> Tuple[str, str, Optional[int]]:
        """
        Generate response from LLM.
        Returns: (answer_text, model_used, token_count)
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Return list of models supported or available on this provider."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if provider is online and ready for inference."""
        pass
