"""
Local Sentence-Transformers Embedder for Legal RAG.
Runs Qwen/Qwen3-Embedding-0.6B with MRL (512-dim truncation) or any configured model with L2 normalization.
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
    Supports Matryoshka Representation Learning (MRL) dimension truncation and L2 normalization.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        if device:
            self.device = device
        elif settings.EMBEDDING_DEVICE:
            self.device = settings.EMBEDDING_DEVICE
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Initializing LocalEmbedder: model='{self.model_name}', dim={self.dimension}, device='{self.device}'"
        )
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            try:
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
        Generate normalized embeddings for a list of document strings with MRL truncation.
        """
        if not texts:
            return []

        bs = batch_size or self.batch_size
        logger.debug(f"Generating embeddings for {len(texts)} texts (batch_size={bs})...")

        encode_kwargs = {
            "batch_size": bs,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.dimension:
            encode_kwargs["truncate_dim"] = self.dimension

        embeddings = self.model.encode(texts, **encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate normalized embedding for a single query string with MRL truncation.
        """
        if not query:
            raise ValueError("Query string cannot be empty")

        encode_kwargs = {
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.dimension:
            encode_kwargs["truncate_dim"] = self.dimension

        embedding = self.model.encode(query, **encode_kwargs)
        return embedding.tolist()
