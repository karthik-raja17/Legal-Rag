"""
ChromaDB client for the Legal RAG system.
Handles connection, collection management, and document operations.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from requests.exceptions import RequestException, ConnectionError, Timeout

from src.config.settings import settings

logger = logging.getLogger(__name__)


def is_retryable_exception(exception: Exception) -> bool:
    """Retry on transient network errors AND stale collection errors."""
    if isinstance(exception, (ConnectionError, Timeout, RequestException)):
        return True
    if hasattr(exception, "status_code") and 500 <= exception.status_code < 600:
        return True
    if hasattr(exception, "__cause__") and isinstance(exception.__cause__, RequestException):
        return True
    # Retry if the collection doesn't exist – we can refresh and retry.
    if isinstance(exception, InvalidCollectionException):
        return True
    return False


class ChromaClient:
    """
    Production‑grade ChromaDB client with:
        - Lazy connection initialization.
        - Collection creation with dot_product distance (optimized for normalized embeddings).
        - Batch insertion with retry logic on transient errors.
        - Query with metadata filtering.
        - Graceful error handling and logging.
        - Delete by document_id for reindexing.
    """

    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the ChromaDB client.

        Args:
            collection_name: Name of the collection to use. Defaults to settings.CHROMA_COLLECTION.
        """
        self.host = settings.CHROMA_HOST
        self.port = settings.CHROMA_PORT
        self.collection_name = collection_name or settings.CHROMA_COLLECTION

        self._client: Optional[chromadb.HttpClient] = None
        self._collection: Optional[chromadb.Collection] = None

        logger.info(f"ChromaClient initialized for {self.host}:{self.port}/{self.collection_name}")

    @property
    def client(self) -> chromadb.HttpClient:
        """Expose the underlying HTTP client, lazily connecting if needed."""
        return self._get_client()

    def _get_client(self) -> chromadb.HttpClient:
        """Lazily initialize the HTTP client."""
        if self._client is None:
            try:
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(allow_reset=True),
                )
                logger.info(f"Connected to ChromaDB at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB: {e}")
                raise RuntimeError(f"Cannot connect to ChromaDB at {self.host}:{self.port}") from e
        return self._client

    def set_search_ef(self, ef_search: int) -> None:
        """
        Update the HNSW search_ef parameter on an existing collection
        without re‑indexing.
        """
        if self._collection is None:
            raise RuntimeError("Collection not initialised")
        metadata = self._collection.metadata or {}
        metadata["hnsw:search_ef"] = ef_search
        self._collection.modify(metadata=metadata)
        logger.info(f"Updated search_ef to {ef_search} on collection '{self.collection_name}'")

    def _get_collection(self, force_refresh: bool = False) -> chromadb.Collection:
        """
        Get the collection object. If force_refresh is True, bypass cache and
        re-fetch from the server.
        """
        if self._collection is not None and not force_refresh:
            return self._collection

        client = self._get_client()
        try:
            self._collection = client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' fetched.")
        except InvalidCollectionException:
            # Collection doesn't exist – create it.
            logger.info(f"Collection '{self.collection_name}' does not exist. Creating...")
            self._create_collection(client)
            self._collection = client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection: {e}")
            raise

        # Update search_ef if needed
        desired_ef = getattr(settings, "HNSW_EF_SEARCH", 500)
        current_metadata = self._collection.metadata or {}
        current_ef = current_metadata.get("hnsw:search_ef")
        if current_ef != desired_ef:
            self.set_search_ef(desired_ef)

        return self._collection

    def _create_collection(self, client) -> None:
        """
        Create the collection with HNSW parameters from settings.
        """
        # HNSW configuration from environment / settings
        hnsw_config = {
            "hnsw:space": getattr(settings, "HNSW_SPACE", "ip"),
            "hnsw:construction_ef": getattr(settings, "HNSW_EF_CONSTRUCTION", 400),
            "hnsw:M": getattr(settings, "HNSW_M", 64),
            "hnsw:search_ef": getattr(settings, "HNSW_EF_SEARCH", 500),
        }
        metadata = {
            "description": "Legal contracts collection",
            **hnsw_config,
        }
        try:
            client.create_collection(
                name=self.collection_name,
                metadata=metadata,
                get_or_create=False,  # we already checked it doesn't exist
            )
            logger.info(f"Created collection '{self.collection_name}' with HNSW config: {hnsw_config}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def get_all_chunks(self, limit: int = 1000, offset: int = 0, include: List[str] = None):
        return self._collection.get(limit=limit, offset=offset, include=include or [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def get_all_ids(self, limit: int = 10_000_000) -> List[str]:
        """
        Fetch all document IDs from the collection (paginated).
        Used by the hybrid retriever for version hashing.
        """
        all_ids = []
        offset = 0
        batch_size = 1000
        while True:
            results = self._get_collection().get(
                limit=min(batch_size, limit),
                offset=offset,
                include=[]  # only need IDs
            )
            ids = results.get("ids", [])
            if not ids:
                break
            all_ids.extend(ids)
            offset += len(ids)
            if len(ids) < batch_size or len(all_ids) >= limit:
                break
        return all_ids

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the ChromaDB collection with retry logic.
        Always fetches the latest collection reference to avoid stale UUIDs.
        """
        if not ids:
            return

        n = len(ids)
        if len(embeddings) != n:
            raise ValueError(f"IDs count ({n}) does not match embeddings count ({len(embeddings)})")
        if metadatas and len(metadatas) != n:
            raise ValueError(f"IDs count ({n}) does not match metadatas count ({len(metadatas)})")
        if documents and len(documents) != n:
            raise ValueError(f"IDs count ({n}) does not match documents count ({len(documents)})")

        # Validate embedding dimensions (all same length)
        if embeddings:
            dim = len(embeddings[0])
            for idx, emb in enumerate(embeddings):
                if not isinstance(emb, list):
                    raise ValueError(f"Embedding at index {idx} is not a list")
                if len(emb) != dim:
                    raise ValueError(f"Embedding dimension mismatch: {len(emb)} vs {dim} at index {idx}")

        try:
            # CRITICAL FIX: Always force refresh to get the latest collection by name
            collection = self._get_collection(force_refresh=True)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.info(f"Added {n} documents to collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise RuntimeError(f"ChromaDB insertion failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_documents: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the collection with a single embedding vector.

        Args:
            query_embedding: The query embedding vector.
            n_results: Number of results to return.
            filter_metadata: Optional metadata filter (e.g., {"document_id": "test_001"}).
            include_documents: Whether to include the document text in results.

        Returns:
            Dict with keys: ids, distances, metadatas, documents.
            All lists are flattened (single query).
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        include = ["metadatas", "distances"]
        if include_documents:
            include.append("documents")

        try:
            collection = self._get_collection()
            where_filter = filter_metadata if filter_metadata else None
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=include,
            )
            # Flatten results (single query)
            return {
                "ids": results["ids"][0] if results.get("ids") else [],
                "distances": results["distances"][0] if results.get("distances") else [],
                "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
                "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise RuntimeError(f"ChromaDB query failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def delete_by_filter(self, filter_metadata: Dict[str, Any]) -> int:
        """
        Delete documents matching a metadata filter (e.g., {'document_id': 'doc_123'}).

        Args:
            filter_metadata: Metadata dict to match.

        Returns:
            Number of deleted documents.
        """
        if not filter_metadata:
            raise ValueError("filter_metadata cannot be empty")

        try:
            collection = self._get_collection()
            # First, get the IDs of matching documents
            results = collection.get(where=filter_metadata, include=[])
            ids = results.get("ids", [])
            if not ids:
                logger.info(f"No documents found matching filter {filter_metadata}")
                return 0

            # Delete by IDs
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents matching filter {filter_metadata}")
            return len(ids)
        except Exception as e:
            logger.error(f"Delete by filter failed: {e}")
            raise RuntimeError(f"ChromaDB delete failed: {e}") from e

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a specific document.

        Args:
            document_id: The document ID (from metadata).

        Returns:
            Number of deleted chunks.
        """
        return self.delete_by_filter({"document_id": document_id})

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dict with collection metadata and count, or {"error": str} on failure.
        """
        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "metadata": collection.metadata,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def delete_collection(self) -> None:
        """Delete the entire collection (use with caution)."""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if ChromaDB is reachable and the collection exists.
        Returns True if healthy.
        """
        try:
            self._get_client()
            self._get_collection()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the connection and release resources."""
        self._client = None
        self._collection = None
        logger.info("ChromaClient closed")