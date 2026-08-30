"""
Local SentenceTransformer Embedder.
Generates dense vector embeddings using local Hugging Face / SentenceTransformer models
with Matryoshka Representation Learning (MRL) dimension truncation and L2 normalization.
"""
import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from src.config.settings import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Local embedding generator using SentenceTransformer.
    Embeds documents and queries on GPU (CUDA) or CPU.
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
        self.device = device or settings.EMBEDDING_DEVICE
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None

        logger.info(
            f"Initializing LocalEmbedder: model='{self.model_name}', "
            f"dim={self.dimension}, device='{self.device}'"
        )

    @property
    def model(self) -> SentenceTransformer:
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
        Uses torch.inference_mode() to minimize VRAM usage.
        """
        if not texts:
            return []

        bs = batch_size or self.batch_size or 16
        logger.debug(f"Generating embeddings for {len(texts)} texts (batch_size={bs})...")

        encode_kwargs = {
            "batch_size": bs,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.dimension:
            encode_kwargs["truncate_dim"] = self.dimension

        try:
            with torch.inference_mode():
                embeddings = self.model.encode(texts, **encode_kwargs)
            return embeddings.tolist()
        except torch.AcceleratorError as e:
            logger.warning(f"CUDA OOM during embedding generation. Clearing cache and retrying on CPU/batch... ({e})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Retry with smaller batch or CPU
            encode_kwargs["batch_size"] = 8
            with torch.inference_mode():
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

        with torch.inference_mode():
            embedding = self.model.encode(query, **encode_kwargs)
        return embedding.tolist()
