"""
Local Cross-Encoder reranker for retrieved candidate chunks.
Performs semantic cross-encoder re-ranking or score-based ordering.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)


class LocalReranker:
    """
    Local Reranker that scores and re-ranks candidate passages using cross-encoders.
    """

    def __init__(self, model_name: Optional[str] = None, enabled: Optional[bool] = None):
        self.enabled = enabled if enabled is not None else (settings.RERANKER_TYPE != "none")
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model = None

    def _get_model(self):
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading local CrossEncoder model '{self.model_name}'...")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder model '{self.model_name}': {e}. Using RRF score ordering.")
                self.enabled = False
        return self._model

    async def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks using local CrossEncoder.
        """
        if not retrieved_chunks:
            return []

        top_n = top_n or settings.HYBRID_TOP_K

        if not self.enabled:
            return retrieved_chunks[:top_n]

        model = self._get_model()
        if model is None:
            return retrieved_chunks[:top_n]

        pairs = [[query, chunk.get("text", "")] for chunk in retrieved_chunks]
        
        loop = asyncio.get_running_loop()
        try:
            scores = await loop.run_in_executor(None, model.predict, pairs)
            scored_chunks = []
            for chunk, score in zip(retrieved_chunks, scores):
                c = chunk.copy()
                c["rerank_score"] = float(score)
                scored_chunks.append(c)

            scored_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return scored_chunks[:top_n]
        except Exception as e:
            logger.warning(f"Local cross-encoder rerank failed: {e}. Returning original candidates.")
            return retrieved_chunks[:top_n]

    def close(self):
        pass