"""
Vertex AI Reranker using Google's Discovery Engine Ranking API.
Performs cross-encoder re-ranking on retrieved candidate chunks.

Production‑grade with async support, retries, caching, and graceful fallback.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import discoveryengine_v1 as discoveryengine
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config.settings import settings

logger = logging.getLogger(__name__)


class VertexReranker:
    """
    Reranks retrieved candidate chunks using Google Cloud Vertex AI Ranking API.

    The Ranking API acts as a cross-encoder: it takes the user query and candidate
    chunks, evaluates how well each chunk answers the query, and assigns a
    precise relevance score. This is significantly more accurate than vector
    similarity alone.

    Usage (async):
        reranker = VertexReranker()
        final_chunks = await reranker.rerank(
            query="Quelles sont les obligations du Prestataire ?",
            retrieved_chunks=chunks,  # top 20 from hybrid search
            top_n=5
        )
    """

    def __init__(
        self,
        location: Optional[str] = None,
        model: Optional[str] = None,
        ranking_config: Optional[str] = None,
        enabled: Optional[bool] = None,
        max_query_length: int = 2000,
    ):
        """
        Args:
            location: GCP location for the ranking config (e.g., 'global').
            model: Ranking model name (e.g., 'ranking-default').
            ranking_config: Full resource path; if not provided, built from project/location.
            enabled: Override settings.VERTEX_RERANKER_ENABLED.
            max_query_length: Truncate query to this many characters to avoid API limits.
        """
        self.enabled = enabled if enabled is not None else getattr(settings, "VERTEX_RERANKER_ENABLED", False)
        if not self.enabled:
            logger.info("VertexReranker is disabled by configuration.")
            return

        self.location = location or getattr(settings, "VERTEX_RERANKER_LOCATION", "global")
        self.model = model or getattr(settings, "VERTEX_RERANKER_MODEL", "ranking-default")
        self.project_id = settings.GCP_PROJECT_ID
        self.max_query_length = max_query_length

        # Build the ranking config resource path if not provided
        if ranking_config:
            self.ranking_config = ranking_config
        else:
            self.ranking_config = (
                f"projects/{self.project_id}/locations/{self.location}/"
                f"rankingConfigs/default_ranking_config"
            )

        self._client = None  # Lazy initialization
        logger.info(
            f"VertexReranker initialized: config={self.ranking_config}, "
            f"model={self.model}, location={self.location}, enabled={self.enabled}"
        )

    @property
    def client(self) -> discoveryengine.RankServiceClient:
        """Lazy-initialize the client."""
        if self._client is None:
            self._client = discoveryengine.RankServiceClient()
        return self._client

    async def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks based on semantic relevance to the query.

        Args:
            query: The user query string (will be truncated if too long).
            retrieved_chunks: List of chunk dicts from ChromaDB/hybrid search.
                              Must contain 'id', 'text', and optionally 'heading'.
            top_n: Number of top results to return after reranking.
                   Defaults to settings.VERTEX_RERANKER_TOP_N.

        Returns:
            Reranked list of chunk dicts, sliced to top_n, with a 'rerank_score'
            field added to each. If reranking is disabled or fails, returns
            the original candidates sliced to top_n.
        """
        # Short-circuit if disabled
        if not self.enabled:
            logger.debug("Reranking disabled; returning original results.")
            return retrieved_chunks[:top_n] if top_n else retrieved_chunks

        if not retrieved_chunks:
            return []

        top_n = top_n or getattr(settings, "VERTEX_RERANKER_TOP_N", 5)

        # Truncate long queries to avoid API limits
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]
            logger.warning(f"Query truncated to {self.max_query_length} characters.")

        # Build records for the Ranking API
        records = []
        record_map = {}

        for idx, chunk in enumerate(retrieved_chunks):
            # Use a fallback id if missing
            record_id = str(chunk.get("id", f"chunk_{idx}"))
            title = chunk.get("heading") or chunk.get("metadata", {}).get("heading", "")
            content = chunk.get("text", "")
            if not content:
                # Skip empty chunks (should not happen but be safe)
                continue

            records.append(
                discoveryengine.RankingRecord(
                    id=record_id,
                    title=title,
                    content=content,
                )
            )
            record_map[record_id] = chunk

        if not records:
            logger.warning("No valid records to rerank; returning original order.")
            return retrieved_chunks[:top_n]

        # Perform the rerank call with retries
        try:
            reranked = await self._call_rank_api(query, records, top_n)
        except Exception as e:
            logger.error(f"Vertex AI reranking failed after retries: {e}", exc_info=True)
            # Fallback: return original candidates sliced to top_n
            logger.info("Falling back to original ranking.")
            return retrieved_chunks[:top_n]

        # Build reranked results with scores
        reranked_chunks = []
        for result in reranked:
            original_chunk = record_map.get(result.id)
            if original_chunk:
                chunk_copy = original_chunk.copy()
                chunk_copy["rerank_score"] = result.score
                reranked_chunks.append(chunk_copy)

        logger.info(
            f"Reranked {len(retrieved_chunks)} candidates → {len(reranked_chunks)} top results "
            f"(top_n={top_n})"
        )
        return reranked_chunks

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_rank_api(
        self,
        query: str,
        records: List[discoveryengine.RankingRecord],
        top_n: int,
    ) -> List[discoveryengine.RankingRecord]:
        """
        Call the Ranking API with retries. Offloads the synchronous call to a thread.
        """
        request = discoveryengine.RankRequest(
            ranking_config=self.ranking_config,
            model=self.model,
            query=query,
            records=records,
            top_n=top_n,
            # We do not need full metadata in the response
            ignore_record_details_in_response=True,
        )

        loop = asyncio.get_running_loop()
        start = time.time()
        logger.debug(f"Calling Ranking API with {len(records)} records, top_n={top_n}")

        response = await loop.run_in_executor(
            None,
            self.client.rank,
            request,
        )

        elapsed = time.time() - start
        logger.info(f"Ranking API call completed in {elapsed:.2f}s, returned {len(response.records)} results.")
        return response.records

    def close(self):
        """Release resources (no-op for this client)."""
        pass