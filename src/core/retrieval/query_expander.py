"""
Query expansion using local Ollama LLM with caching and observability.
Generates multiple legal-specific reformulations of the user query to improve retrieval recall.
"""
import asyncio
import logging
import re
import time
from typing import List, Optional, Dict, Tuple

from src.core.llm.ollama_client import OllamaClient
from src.config.settings import settings

logger = logging.getLogger(__name__)


class _TTLCache:
    """Thread-safe async cache with TTL."""
    def __init__(self, maxsize: int = 128, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, List[str]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[List[str]]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            timestamp, value = entry
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None
            return value

    async def set(self, key: str, value: List[str]):
        async with self._lock:
            if len(self._cache) >= self.maxsize:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = (time.time(), value)


class QueryExpander:
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model_name: Optional[str] = None,
        num_variants: Optional[int] = None,
        temperature: Optional[float] = None,
        enabled: Optional[bool] = None,
        cache_ttl_seconds: int = 3600,
    ):
        self.ollama = ollama_client or OllamaClient(
            model=model_name or settings.OLLAMA_MODEL,
            temperature=temperature if temperature is not None else settings.QUERY_EXPANSION_TEMPERATURE,
        )
        self.num_variants = num_variants or settings.QUERY_EXPANSION_NUM_VARIANTS
        self.enabled = enabled if enabled is not None else settings.ENABLE_QUERY_EXPANSION
        self._cache = _TTLCache(maxsize=128, ttl_seconds=cache_ttl_seconds)

        self.prompt_template = (
            "Vous êtes un expert juridique spécialisé dans les contrats. "
            "Reformulez la question suivante en {num_variants} variantes de recherche différentes, "
            "en utilisant un vocabulaire juridique varié, en gardant le même sens. "
            "Retournez uniquement les variantes, une par ligne, sans numérotation "
            "et sans commentaire.\n\n"
            "Question originale : {query}\n\n"
            "Variantes :"
        )

    async def expand(self, query: str) -> List[str]:
        if not self.enabled:
            return [query]

        query = (query or "").strip()
        if len(query) < 3:
            return [query]

        cache_key = query.lower()
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self.prompt_template.format(
            num_variants=self.num_variants - 1,
            query=query,
        )

        try:
            raw_resp = await self.ollama.agenerate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=256,
            )
            variants = []
            for line in raw_resp.strip().split("\n"):
                line = line.strip()
                cleaned = re.sub(r"^\s*[\d\-•*]+\.?\s*", "", line).strip()
                if cleaned and len(cleaned) > 3:
                    variants.append(cleaned)

            unique = []
            seen = set()
            for v in [query] + variants:
                k = v.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    unique.append(v.strip())
                if len(unique) >= self.num_variants:
                    break

            result = unique
            await self._cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Returning original query.")
            return [query]

    def close(self):
        pass