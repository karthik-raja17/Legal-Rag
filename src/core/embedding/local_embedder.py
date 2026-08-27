"""
Local Sentence-Transformers Embedder for Legal RAG.
Runs all-MiniLM-L6-v2 (or configured model) locally on CPU / CUDA with normalized embeddings.
"""
import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.config.settings import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Local embedding generator using sentence-transformers.
    Produces 384-dimensional normalized vectors with 'all-MiniLM-L6-v2'.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Initializing LocalEmbedder with model '{self.model_name}' on device '{self.device}'"
        )
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            try:
                # Try loading from local cache first to avoid network timeout delays
                self._model = SentenceTransformer(
                    self.model_name, device=self.device, local_files_only=True
                )
            except Exception:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model '{self.model_name}' loaded successfully.")
        return self._model

    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate normalized embeddings for a list of document strings.
        """
        if not texts:
            return []

        bs = batch_size or self.batch_size
        logger.debug(f"Generating embeddings for {len(texts)} texts (batch_size={bs})...")

        # Encode texts with L2 normalization (for cosine similarity via inner product)
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate normalized embedding for a single query string.
        """
        if not query:
            raise ValueError("Query string cannot be empty")

        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()

