import logging
from typing import List, Tuple, Optional, Dict, Any
from groq import Groq, APIError, AuthenticationError, RateLimitError

from app.core.config import settings
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key
        self._client: Optional[Groq] = None
        if self.api_key:
            try:
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

    @property
    def provider_name(self) -> str:
        return "groq"

    def is_healthy(self) -> bool:
        return bool(self._client and self.api_key)

    def get_available_models(self) -> List[Dict[str, Any]]:
        models = []
        # Query active models dynamically if key is valid
        if self._client:
            try:
                remote_models = self._client.models.list()
                remote_ids = {m.id for m in remote_models.data}
                for m_id in settings.available_groq_models:
                    models.append({
                        "id": m_id,
                        "name": f"Groq - {m_id}",
                        "provider": "groq",
                        "is_available": m_id in remote_ids or True,
                        "is_default": m_id == settings.default_groq_model,
                        "notes": "Cloud-hosted ultra-fast inference",
                    })
                return models
            except Exception as e:
                logger.warning(f"Error listing Groq models: {e}")

        # Fallback to configured list
        for m_id in settings.available_groq_models:
            models.append({
                "id": m_id,
                "name": f"Groq - {m_id}",
                "provider": "groq",
                "is_available": bool(self.api_key),
                "is_default": m_id == settings.default_groq_model,
                "notes": "Cloud-hosted ultra-fast inference",
            })
        return models

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Tuple[str, str, Optional[int]]:
        if not self._client or not self.api_key:
            raise ValueError(
                "Groq API Key is not configured. Please set GROQ_API_KEY in the environment or backend/.env file."
            )

        target_model = model or settings.default_groq_model

        # Fallback cascade in case the requested model is restricted/missing
        fallback_cascade = [target_model]
        for fb in settings.available_groq_models:
            if fb not in fallback_cascade:
                fallback_cascade.append(fb)

        last_exc = None
        for current_model in fallback_cascade:
            try:
                response = self._client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else None
                return answer, current_model, token_count

            except APIError as e:
                if "model_not_found" in str(e) or getattr(e, "status_code", 0) == 404:
                    logger.warning(f"Model '{current_model}' unavailable on Groq. Trying next in cascade...")
                    last_exc = e
                    continue
                raise e

        raise RuntimeError(f"All attempted Groq models failed. Last error: {last_exc}")
