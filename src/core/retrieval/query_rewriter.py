"""
Query Rewriter – converts natural language queries into formal legal queries.
Uses Gemini 2.5 Flash with cached execution, zero thinking latency, and fallback.
"""
import asyncio
import logging
import time
from typing import Optional

import vertexai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)

from src.settings import settings

vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        cache_ttl: int = None,
    ):
        self.model_name = model_name or getattr(settings, "REWRITER_MODEL", "gemini-2.5-flash")
        self.temperature = temperature if temperature is not None else 0.0
        self.cache_ttl = cache_ttl or getattr(settings, "REWRITER_CACHE_TTL_SECONDS", 3600)
        self._cache = {}
        self._model = None

        # Permissive safety settings for legal contracts
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    @property
    def model(self) -> GenerativeModel:
        if self._model is None:
            self._model = GenerativeModel(
                model_name=self.model_name,
                system_instruction=(
                    "Tu es un expert juridique spécialisé dans les contrats photovoltaïques et baux civils. "
                    "Ta tâche unique est de reformuler la question de l'utilisateur en une requête de recherche "
                    "juridique formelle, précise et concise pour interroger une base vectorielle. "
                    "Ne réponds pas à la question. Ne rajoute pas d'explication. Renvoie uniquement la question reformulée."
                ),
            )
        return self._model

    def _get_cached(self, query: str) -> Optional[str]:
        if query in self._cache:
            ts, rewritten = self._cache[query]
            if time.time() - ts < self.cache_ttl:
                return rewritten
            del self._cache[query]
        return None

    def _set_cache(self, query: str, rewritten: str) -> None:
        self._cache[query] = (time.time(), rewritten)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_llm(self, query: str) -> str:
        prompt = f"Question brute : {query}\nQuestion juridique formulée :"

        generation_config = GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=1024,
            thinking_config={"thinking_budget": 0},  # Disable thinking to prevent empty MAX_TOKENS aborts
        )

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=self.safety_settings,
        )

        if not response.candidates or not response.candidates[0].content.parts:
            logger.warning(f"Empty candidate returned. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'None'}")
            return query

        return response.text.strip()

    async def rewrite(self, query: str) -> str:
        cached = self._get_cached(query)
        if cached is not None:
            return cached

        try:
            loop = asyncio.get_running_loop()
            rewritten = await loop.run_in_executor(None, self._call_llm, query)
            if rewritten and len(rewritten) > 3:
                self._set_cache(query, rewritten)
                logger.info(f"Query rewrite: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed ({e}). Falling back to original query.")

        return query