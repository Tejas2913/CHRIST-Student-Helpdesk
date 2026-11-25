import logging
from typing import List, Tuple, Optional, Dict, Any
import httpx

from app.core.config import settings
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = 60.0

    @property
    def provider_name(self) -> str:
        return "ollama"

    def is_healthy(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> List[Dict[str, Any]]:
        pulled_models = set()
        is_server_up = False

        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    is_server_up = True
                    data = res.json()
                    pulled_models = {m.get("name") for m in data.get("models", [])}
        except Exception:
            pass

        models = []
        for m_id in settings.available_ollama_models:
            is_pulled = m_id in pulled_models or f"{m_id}:latest" in pulled_models
            notes = "Local model (Colab evaluated)"
            if m_id == "qwen2.5:1.5b":
                notes = "⭐ Provisional Winner from Notebook (Decision score: 0.729, 1.13s latency)"
            elif m_id == "llama3.2:1b":
                notes = "Smallest baseline (Decision score: 0.544)"
            elif m_id == "gemma2:2b":
                notes = "Google candidate (Decision score: 0.492)"

            models.append({
                "id": m_id,
                "name": f"Ollama - {m_id}",
                "provider": "ollama",
                "is_available": is_server_up and is_pulled,
                "is_default": m_id == settings.default_ollama_model,
                "notes": notes if is_server_up else f"{notes} (Ollama server offline at {self.base_url})",
            })

        return models

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> Tuple[str, str, Optional[int]]:
        target_model = model or settings.default_ollama_model

        if not self.is_healthy():
            raise ConnectionError(
                f"Ollama local inference server is not running at {self.base_url}. "
                "Please start Ollama using 'ollama serve' and pull the model with "
                f"'ollama pull {target_model}' or switch provider to Groq."
            )

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code != 200:
                    raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")

                data = res.json()
                answer = data.get("message", {}).get("content", "")
                eval_count = data.get("eval_count")
                return answer, target_model, eval_count

        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running?"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")
