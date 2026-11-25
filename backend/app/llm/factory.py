import logging
from typing import Dict, List, Any, Optional

from app.core.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

class LLMProviderFactory:
    _instance: Optional["LLMProviderFactory"] = None

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {
            "groq": GroqProvider(),
            "ollama": OllamaProvider(),
        }
        # Always enforce groq as the default working provider unless explicitly configured otherwise
        self.active_provider_name: str = settings.default_provider or "groq"
        self.active_model_name: str = (
            settings.default_groq_model
            if self.active_provider_name == "groq"
            else settings.default_ollama_model
        )

    @classmethod
    def get_instance(cls) -> "LLMProviderFactory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_provider_healthy(self, name: Optional[str] = None) -> bool:
        p = self.get_provider(name)
        return p.is_healthy()

    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        provider_key = (name or self.active_provider_name).lower().strip()
        if provider_key not in self._providers:
            raise ValueError(f"Unknown LLM provider: '{provider_key}'. Supported: {list(self._providers.keys())}")
        return self._providers[provider_key]

    def set_active_selection(self, provider: str, model: str):
        provider_key = provider.lower().strip()
        if provider_key not in self._providers:
            raise ValueError(f"Unknown LLM provider: '{provider}'")
        self.active_provider_name = provider_key
        self.active_model_name = model
        logger.info(f"Active LLM switched to {provider_key} -> {model}")

    def list_all_models(self) -> List[Dict[str, Any]]:
        all_models = []
        for name, provider in self._providers.items():
            models = provider.get_available_models()
            for m in models:
                m["is_selected"] = (
                    m["provider"] == self.active_provider_name and m["id"] == self.active_model_name
                )
                all_models.append(m)
        return all_models
