"""
Indexer service: Chunks parsed documents, generates embeddings (if missing),
and indexes them into ChromaDB.
"""
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import requests
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from google.api_core import exceptions as gcp_exceptions

from src.core.parser.chunker import DocumentChunker
from src.core.indexer.chroma_client import ChromaClient
from src.core.parser.pdf_parser import ParsedDocument
from src.core.embedding.bge_client import BGEEmbedderClient
from src.settings import settings

logger = logging.getLogger(__name__)


class Indexer:
    """
    Production‑grade indexer for legal documents.

    Features:
        - Lazy initialization of embedding model and Chroma client.
        - Batch embedding generation to reduce API calls and improve throughput.
        - Automatic retries for transient errors.
        - Configuration via settings (batch sizes, timeout).
        - Graceful handling of missing embeddings (generates them on the fly).
        - Close method for resource cleanup.
    """

    def __init__(self):
        self.chunker = DocumentChunker()
        self.chroma = ChromaClient()
        self.bge_embedder = BGEEmbedderClient()

        # Batch size for embedding API calls (configurable)
        self.embedding_batch_size = getattr(settings, "EMBEDDING_BATCH_SIZE", 16)
        # Batch size for ChromaDB inserts (configurable)
        self.chroma_batch_size = getattr(settings, "CHROMA_BATCH_SIZE", 100)

        logger.info(
            f"Indexer initialized: embedding_batch_size={self.embedding_batch_size}, "
            f"chroma_batch_size={self.chroma_batch_size}"
        )

    def _estimate_input_tokens(self, text: str, title: str) -> int:
        """
        Conservative token estimate for a single chunk.

        Uses a chars-per-token ratio of 2.5 (more conservative than the chunker's 3.3).
        This helps keep batches under the BGE model's token limit (8192 tokens per input).
        """
        chars_per_token_safety = 2.5
        return int((len(text) + len(title)) / chars_per_token_safety) + 8

    def _make_token_aware_batches(
        self,
        chunks: List[Dict[str, Any]],
        title: str,
        max_tokens_per_request: int = 4000,
        max_items_per_request: int = 16,
    ) -> List[List[Dict[str, Any]]]:
        max_items = max_items_per_request or 16
        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_tokens = 0

        for chunk in chunks:
            # Estimate tokens using the actual text that will be embedded
            # (heading + text)
            text_with_heading = self._prepare_text_for_embedding(chunk)
            est = self._estimate_input_tokens(text_with_heading, title)

            if est > max_tokens_per_request:
                if current_batch:
                    batches.append(current_batch)
                    current_batch, current_tokens = [], 0
                batches.append([chunk])
                continue

            would_exceed_tokens = current_tokens + est > max_tokens_per_request
            would_exceed_items = len(current_batch) >= max_items

            if current_batch and (would_exceed_tokens or would_exceed_items):
                batches.append(current_batch)
                current_batch, current_tokens = [], 0

            current_batch.append(chunk)
            current_tokens += est

        if current_batch:
            batches.append(current_batch)

        return batches

    def _prepare_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare the text for embedding by prefixing the section heading if available.
        This adds structural context to the dense vector.
        """
        heading = chunk.get("heading", "")
        text = chunk.get("text", "")
        if heading:
            # DEBUG: uncomment to verify headings are being prepended
            # logger.info(f"Prepared text (first 100 chars): {heading} | {text[:100]}")
            return f"{heading} | {text}"
        return text

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(
            (
                gcp_exceptions.ServerError,
                gcp_exceptions.TooManyRequests,
                gcp_exceptions.DeadlineExceeded,
                gcp_exceptions.ResourceExhausted,
                ConnectionError,
                TimeoutError,
                requests.exceptions.HTTPError,   # <-- ADD THIS
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _generate_embeddings_from_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Process in smaller batches with delay
        all_embeddings = []
        batch_size = 8  # smaller than 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.bge_embedder.embed_documents(batch, batch_size=batch_size)
            all_embeddings.extend(embeddings)
            # Wait between batches to avoid overwhelming the embedder
            if i + batch_size < len(texts):
                time.sleep(0.5)
        return all_embeddings

    def index_document(self, parsed_doc: ParsedDocument, document_id: str) -> int:
        """
        Chunk the document, generate embeddings, and index into ChromaDB.

        Args:
            parsed_doc: ParsedDocument object from the parser.
            document_id: Unique identifier for the document.

        Returns:
            Number of chunks indexed.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If embedding generation or ChromaDB insertion fails.
        """
        if not parsed_doc or not document_id:
            raise ValueError("parsed_doc and document_id are required.")

        structure = parsed_doc.structure
        if not structure:
            logger.warning(f"Document {document_id} has no structure; skipping indexing.")
            return 0

        # 1. Chunk the document
        chunks = self.chunker.chunk_document(structure)
        if not chunks:
            logger.warning(f"No chunks generated for document {document_id}; skipping indexing.")
            return 0

        # 2. Always regenerate embeddings (ignore any existing)
        chunks_without_emb = chunks  # all chunks

        # 3. Generate embeddings
        if chunks_without_emb:
            logger.info(
                f"Generating embeddings for {len(chunks_without_emb)} chunks "
                f"(document {document_id})"
            )
            title = parsed_doc.metadata.get("title", document_id)

            # Token-aware batching
            token_batches = self._make_token_aware_batches(chunks_without_emb, title)
            logger.info(
                f"Document {document_id}: {len(chunks_without_emb)} chunks packed into "
                f"{len(token_batches)} token-aware batches."
            )

            for batch_num, batch in enumerate(token_batches, start=1):
                # Prepare texts with heading prefix
                texts = [self._prepare_text_for_embedding(chunk) for chunk in batch]
                try:
                    batch_embeddings = self._generate_embeddings_from_texts(texts)
                    for chunk, emb_vec in zip(batch, batch_embeddings):
                        chunk["embedding"] = emb_vec
                    logger.info(
                        f"Generated embeddings for batch {batch_num}/{len(token_batches)} "
                        f"({len(batch)} chunks) (doc {document_id})"
                    )
                except Exception as e:
                    logger.error(
                        f"Batch {batch_num}/{len(token_batches)} embedding generation failed "
                        f"for document {document_id}: {e}"
                    )
                    raise RuntimeError(
                        f"Failed to generate embeddings for document {document_id}"
                    ) from e

        # 4. Prepare data for indexing
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for idx, chunk in enumerate(chunks):
            emb = chunk.get("embedding")
            if not emb:
                logger.error(
                    f"Chunk missing embedding after generation for document {document_id}; skipping."
                )
                continue

            chunk_id = f"{document_id}_{chunk['section_id']}_{idx}"
            ids.append(chunk_id)
            embeddings.append(emb)

            metadata = {
                "section_id": chunk.get("section_id", "unknown"),
                "heading": (chunk.get("heading", "") or "")[:200],
                "breadcrumb": chunk.get("breadcrumb", ""),
                "level": chunk.get("level", 0),
                "page": chunk.get("page", 1),
                "clause_type": chunk.get("clause_type", "general"),
                "document_id": document_id,
            }
            metadatas.append(metadata)
            # CRITICAL FIX: Store the prepared text (with heading prefix) in ChromaDB
            documents.append(self._prepare_text_for_embedding(chunk))

        if not ids:
            logger.warning(f"No valid chunks to index for document {document_id}")
            return 0

        # 5. Index into ChromaDB (batched)
        total = len(ids)
        for start in range(0, total, self.chroma_batch_size):
            end = min(start + self.chroma_batch_size, total)
            try:
                self.chroma.add_documents(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents[start:end],
                )
                logger.info(
                    f"Indexed batch {start // self.chroma_batch_size + 1}/"
                    f"{(total - 1) // self.chroma_batch_size + 1} "
                    f"for doc {document_id}"
                )
            except Exception as e:
                logger.error(f"ChromaDB batch insertion failed: {e}")
                raise RuntimeError(f"Failed to index document {document_id}") from e

        logger.info(f"Successfully indexed {len(ids)} chunks for document {document_id}")
        return len(ids)

    def close(self) -> None:
        """Close any persistent connections."""
        if hasattr(self.chroma, "close"):
            try:
                self.chroma.close()
                logger.info("Chroma client closed.")
            except Exception as e:
                logger.warning(f"Error closing Chroma client: {e}")