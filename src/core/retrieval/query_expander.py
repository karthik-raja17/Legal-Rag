"""
Query expansion/rewriting using Gemini with production‑grade retries, caching, and observability.
Generates multiple legal‑specific reformulations of the user query to improve retrieval recall.
"""
import asyncio
import logging
import time
from typing import List, Optional, Dict, Tuple
from functools import lru_cache
import functools

from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted, GoogleAPICallError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import GenerationConfig

from src.settings import settings

logger = logging.getLogger(__name__)

# Simple in‑memory cache with TTL
class _TTLCache:
    """Thread‑safe (async‑safe) cache with TTL and max size."""
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
            # Evict oldest if over maxsize (simple FIFO)
            if len(self._cache) >= self.maxsize:
                # Remove first (oldest) entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = (time.time(), value)


class QueryExpander:
    """
    Generates query variants using Gemini to improve retrieval coverage.

    Features:
        - Configurable number of variants and temperature.
        - TTL‑based caching to avoid redundant calls.
        - Resilient with retries and timeouts.
        - Query truncation to avoid token limits.
        - Logging of latency and cache metrics.

    Example:
        expander = QueryExpander()
        variants = await expander.expand(
            "Quelles sont les obligations de confidentialité ?"
        )
        # -> ["original query", "variante 1", "variante 2", ...]
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_variants: Optional[int] = None,
        temperature: Optional[float] = None,
        enabled: Optional[bool] = None,
        max_query_length: int = 2000,
        cache_ttl_seconds: int = 3600,
        cache_max_size: int = 128,
        prompt_template: Optional[str] = None,
    ):
        self.model_name = model_name or settings.VERTEX_AI_LLM_MODEL
        self.num_variants = num_variants or getattr(settings, "QUERY_EXPANSION_NUM_VARIANTS", 3)
        self.temperature = temperature or getattr(settings, "QUERY_EXPANSION_TEMPERATURE", 0.2)
        self.enabled = enabled if enabled is not None else getattr(settings, "ENABLE_QUERY_EXPANSION", True)
        self.max_query_length = max_query_length
        self.prompt_template = prompt_template or (
            "Vous êtes un expert juridique spécialisé dans les contrats énergétiques. "
            "Reformulez la question suivante en {num_variants} variantes différentes, "
            "en utilisant un vocabulaire juridique varié, en gardant le même sens "
            "mais en changeant la formulation. "
            "Retournez uniquement les variantes, une par ligne, sans numérotation "
            "et sans commentaire supplémentaire.\n\n"
            "Question originale : {query}\n\n"
            "Variantes :"
        )

        self._model = None
        self._cache = _TTLCache(maxsize=cache_max_size, ttl_seconds=cache_ttl_seconds)
        logger.info(
            f"QueryExpander initialized: model={self.model_name}, "
            f"num_variants={self.num_variants}, temperature={self.temperature}, "
            f"enabled={self.enabled}, cache_ttl={cache_ttl_seconds}s"
        )

    @property
    def model(self) -> GenerativeModel:
        if self._model is None:
            self._model = GenerativeModel(self.model_name)
        return self._model

    async def expand(self, query: str) -> List[str]:
        """
        Generate query variants. Returns a list of strings, including the original.
        If expansion is disabled or fails, returns a list containing only the original query.
        """
        if not self.enabled:
            logger.debug("Query expansion disabled; returning original query.")
            return [query]

        query = (query or "").strip()
        if len(query) < 3:
            logger.debug("Query too short; returning original.")
            return [query]

        # Truncate query to avoid token limits
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]
            logger.warning(f"Query truncated to {self.max_query_length} characters.")

        # Check cache
        cache_key = query.lower()  # case-insensitive
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for query: '{query[:50]}...'")
            return cached

        try:
            variants = await self._call_gemini(query)
        except Exception as e:
            logger.error(f"Query expansion failed after retries: {e}", exc_info=True)
            return [query]  # fallback

        # Ensure at least the original is present
        if not variants or not any(v.strip() for v in variants):
            logger.warning("Gemini returned empty variants; falling back to original.")
            return [query]

        # Deduplicate (case-insensitive) and limit to num_variants
        unique = []
        seen = set()
        for v in [query] + variants:
            key = v.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(v.strip())
            if len(unique) >= self.num_variants:
                break

        result = unique
        await self._cache.set(cache_key, result)
        logger.info(f"Generated {len(result)} query variants (cached): {result[:3]}...")
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ServiceUnavailable, ResourceExhausted, GoogleAPICallError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_gemini(self, query: str) -> List[str]:
        """
        Call Gemini with retries and a timeout. Offloads the synchronous call to a thread.
        Returns a list of variant strings (without the original).
        """
        prompt = self.prompt_template.format(
            num_variants=self.num_variants - 1,  # we already have the original
            query=query
        )

        generation_config = GenerationConfig(
            temperature=self.temperature,
            # Optionally set max_output_tokens to avoid long responses
            max_output_tokens=256,
        )

        loop = asyncio.get_running_loop()
        start_time = time.time()

        # Use asyncio.timeout to prevent hanging indefinitely
        try:
                async with asyncio.timeout(10.0):  # 10 seconds timeout
                    # Use functools.partial to bundle kwargs for run_in_executor
                    func = functools.partial(
                        self.model.generate_content,
                        prompt,
                        generation_config=generation_config,
                    )
                    response = await loop.run_in_executor(None, func)
        except asyncio.TimeoutError:
                logger.error("Gemini call timed out after 10s.")
                raise TimeoutError("Gemini generation timed out.")

        elapsed = time.time() - start_time
        logger.info(f"Gemini call completed in {elapsed:.2f}s")

        # Extract response text
        if not response or not response.text:
            logger.warning("Gemini returned empty response.")
            return []

        # Parse variants
        variants = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading numbering like "1.", "2-", etc.
            import re
            cleaned = re.sub(r"^\s*[\d\-•*]+\.?\s*", "", line)
            if cleaned:
                variants.append(cleaned)

        if not variants:
            logger.warning("No variants parsed from Gemini response.")
            return []

        return variants

    def close(self):
        """Release resources (no-op for this client)."""
        pass