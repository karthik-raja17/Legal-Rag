"""
Query Rewriter – converts natural language queries into formal legal queries.
Uses local Ollama with cached execution, fast latency, and fallback.
"""
import asyncio
import logging
import time
from typing import Optional

from src.core.llm.ollama_client import OllamaClient
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        cache_ttl: Optional[int] = None,
    ):
        self.ollama = ollama_client or OllamaClient(
            model=model_name or settings.REWRITER_MODEL,
            temperature=temperature if temperature is not None else settings.REWRITER_TEMPERATURE,
        )
        self.cache_ttl = cache_ttl or settings.REWRITER_CACHE_TTL_SECONDS
        self._cache = {}

        self.system_prompt = (
            "You are a legal contract search specialist. "
            "Your sole task is to rewrite the user's natural language question into a formal, concise, "
            "and precise legal search query suitable for hybrid dense and keyword retrieval across contract clauses. "
            "Do not answer the question. Do not provide explanations or commentary. Return ONLY the rewritten query."
        )

    def _get_cached(self, query: str) -> Optional[str]:
        if query in self._cache:
            ts, rewritten = self._cache[query]
            if time.time() - ts < self.cache_ttl:
                return rewritten
            del self._cache[query]
        return None

    def _set_cache(self, query: str, rewritten: str) -> None:
        self._cache[query] = (time.time(), rewritten)

    async def rewrite(self, query: str) -> str:
        cached = self._get_cached(query)
        if cached is not None:
            return cached

        prompt = f"Raw query: {query}\nFormal legal search query:"

        try:
            rewritten = await self.ollama.agenerate(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.1,
                max_tokens=128,
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten and len(rewritten) > 3:
                self._set_cache(query, rewritten)
                logger.info(f"Query rewrite: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed ({e}). Falling back to original query.")

        return query