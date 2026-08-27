"""
Ollama Client for Local Legal RAG generation, query rewriting, and analysis.
Communicates with Ollama API endpoint (http://localhost:11434) with sync/async support.
"""
import logging
from typing import List, Dict, Any, Optional

import httpx
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with local Ollama LLMs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.temperature = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

        logger.info(
            f"OllamaClient initialized: base_url={self.base_url}, "
            f"model={self.model}, temp={self.temperature}"
        )

    def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List all available models in local Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, httpx.RequestError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Synchronous generation via Ollama /api/generate.
        """
        url = f"{self.base_url}/api/generate"
        temp = temperature if temperature is not None else self.temperature

        options: Dict[str, Any] = {"temperature": temp}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def agenerate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Asynchronous generation via Ollama /api/generate.
        """
        url = f"{self.base_url}/api/generate"
        temp = temperature if temperature is not None else self.temperature

        options: Dict[str, Any] = {"temperature": temp}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Asynchronous chat completion via Ollama /api/chat.
        """
        url = f"{self.base_url}/api/chat"
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

