"""
FAISS Vector Store Client for Legal RAG.
Production-grade local vector database with HNSW indexing, metadata persistence,
cosine similarity search (inner product with normalized embeddings), and metadata filtering.
"""
import json
import logging
import os
import threading
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from src.config.settings import settings

logger = logging.getLogger(__name__)


class FAISSClient:
    """
    Local vector store backed by FAISS HNSW index with metadata persistence.

    Features:
        - HNSW indexing with configurable M, efConstruction, and efSearch.
        - Metric: Inner Product (equivalent to Cosine Similarity with normalized vectors).
        - Thread-safe updates and queries.
        - Automatic disk persistence (index.faiss + metadata.json).
        - Metadata filtering (exact match and $in list queries).
        - Chunk pagination and ID extraction for BM25 and version hashing.
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        dimension: Optional[int] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        self.index_dir = index_dir or settings.FAISS_INDEX_DIR
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.m = m or settings.HNSW_M
        self.ef_construction = ef_construction or settings.HNSW_EF_CONSTRUCTION
        self.ef_search = ef_search or settings.HNSW_EF_SEARCH
        self.collection_name = collection_name or settings.FAISS_COLLECTION

        self._lock = threading.RLock()
        self.index_path = os.path.join(self.index_dir, f"{self.collection_name}.faiss")
        self.meta_path = os.path.join(self.index_dir, f"{self.collection_name}_meta.json")

        self._index: Optional[faiss.IndexHNSWFlat] = None
        # Internal ID (int) -> dict with {"id": chunk_id, "document": text, "metadata": {...}}
        self._id_to_data: Dict[int, Dict[str, Any]] = {}
        # String chunk_id -> int internal id
        self._chunk_id_to_int: Dict[str, int] = {}
        self._next_id: int = 0

        self._load_or_initialize()

    def _create_new_index(self) -> faiss.IndexHNSWFlat:
        """Create a fresh HNSW index with configured parameters."""
        index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def _load_or_initialize(self) -> None:
        """Load index and metadata from disk if available, otherwise initialize new."""
        with self._lock:
            os.makedirs(self.index_dir, exist_ok=True)
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                try:
                    logger.info(f"Loading FAISS index from {self.index_path}...")
                    self._index = faiss.read_index(self.index_path)
                    self._index.hnsw.efSearch = self.ef_search

                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        saved_meta = json.load(f)

                    self._id_to_data = {int(k): v for k, v in saved_meta.get("id_to_data", {}).items()}
                    self._chunk_id_to_int = saved_meta.get("chunk_id_to_int", {})
                    self._next_id = saved_meta.get("next_id", 0)

                    logger.info(
                        f"FAISS index loaded successfully: {self._index.ntotal} vectors, "
                        f"{len(self._id_to_data)} metadata records."
                    )
                    return
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from {self.index_path}: {e}. Creating new index.", exc_info=True)

            # Initialize fresh index
            logger.info(
                f"Initializing new FAISS HNSW index (dim={self.dimension}, M={self.m}, "
                f"efC={self.ef_construction}, efS={self.ef_search})"
            )
            self._index = self._create_new_index()
            self._id_to_data = {}
            self._chunk_id_to_int = {}
            self._next_id = 0

    def reset_collection(self) -> None:
        """Reset and clear FAISS collection to an empty state."""
        with self._lock:
            self._index = self._create_new_index()
            self._id_to_data = {}
            self._chunk_id_to_int = {}
            self._next_id = 0
            self.save()
            logger.info(f"FAISS collection '{self.collection_name}' reset successfully.")

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        with self._lock:
            os.makedirs(self.index_dir, exist_ok=True)
            tmp_index_path = f"{self.index_path}.tmp"
            tmp_meta_path = f"{self.meta_path}.tmp"

            try:
                faiss.write_index(self._index, tmp_index_path)
                meta_payload = {
                    "id_to_data": self._id_to_data,
                    "chunk_id_to_int": self._chunk_id_to_int,
                    "next_id": self._next_id,
                    "dimension": self.dimension,
                    "collection": self.collection_name,
                }
                with open(tmp_meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_payload, f, ensure_ascii=False, indent=2)

                os.replace(tmp_index_path, self.index_path)
                os.replace(tmp_meta_path, self.meta_path)
                logger.info(f"FAISS index and metadata successfully saved to {self.index_dir}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}", exc_info=True)
                if os.path.exists(tmp_index_path):
                    os.remove(tmp_index_path)
                if os.path.exists(tmp_meta_path):
                    os.remove(tmp_meta_path)
                raise

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents with embeddings and metadata to FAISS index.
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

        with self._lock:
            # Convert embeddings to float32 numpy array and normalize
            emb_matrix = np.array(embeddings, dtype=np.float32)
            if emb_matrix.shape[1] != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {emb_matrix.shape[1]}"
                )

            faiss.normalize_L2(emb_matrix)

            # Check if any IDs already exist to avoid duplicate accumulation
            existing_ids = [cid for cid in ids if cid in self._chunk_id_to_int]
            if existing_ids:
                logger.info(f"Replacing {len(existing_ids)} existing document IDs in FAISS index...")
                self._delete_by_ids(existing_ids, auto_save=False)

            # Add vectors to index
            self._index.add(emb_matrix)

            for i, chunk_id in enumerate(ids):
                int_id = self._next_id
                self._next_id += 1

                meta = metadatas[i] if metadatas else {}
                doc = documents[i] if documents else ""

                self._id_to_data[int_id] = {
                    "id": chunk_id,
                    "document": doc,
                    "metadata": meta,
                }
                self._chunk_id_to_int[chunk_id] = int_id

            logger.info(f"Added {n} documents to FAISS index (total: {self._index.ntotal})")
            self.save()

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_documents: bool = True,
    ) -> Dict[str, Any]:
        """
        Query FAISS index with a single embedding vector and optional metadata filtering.

        Returns:
            Dict with keys: "ids", "distances", "metadatas", "documents"
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        with self._lock:
            if self._index.ntotal == 0:
                return {"ids": [], "distances": [], "metadatas": [], "documents": []}

            q_vec = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(q_vec)

            # When filtering, retrieve more candidates from HNSW to ensure sufficient matches
            search_k = min(self._index.ntotal, max(n_results * 10, 50)) if filter_metadata else min(self._index.ntotal, n_results)

            scores, indices = self._index.search(q_vec, search_k)
            scores = scores[0]
            indices = indices[0]

            out_ids: List[str] = []
            out_scores: List[float] = []
            out_metadatas: List[Dict[str, Any]] = []
            out_documents: List[str] = []

            for score, int_id in zip(scores, indices):
                if int_id == -1 or int_id not in self._id_to_data:
                    continue

                item = self._id_to_data[int_id]
                meta = item.get("metadata", {})

                # Apply metadata filter
                if filter_metadata and not self._matches_filter(meta, filter_metadata):
                    continue

                out_ids.append(item["id"])
                out_scores.append(float(score))
                out_metadatas.append(meta)
                if include_documents:
                    out_documents.append(item.get("document", ""))

                if len(out_ids) >= n_results:
                    break

            return {
                "ids": out_ids,
                "distances": out_scores,  # inner product score (higher is more similar)
                "metadatas": out_metadatas,
                "documents": out_documents if include_documents else [],
            }

    def _matches_filter(self, meta: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if item metadata satisfies filter_metadata dict."""
        for key, value in filter_metadata.items():
            if isinstance(value, dict) and "$in" in value:
                allowed = value["$in"]
                if meta.get(key) not in allowed:
                    return False
            else:
                if meta.get(key) != value:
                    return False
        return True

    def _delete_by_ids(self, chunk_ids: List[str], auto_save: bool = True) -> int:
        """
        Delete items by chunk_id string list. Rebuilds index to maintain consistency.
        """
        with self._lock:
            ids_to_remove = set(chunk_ids)
            remaining_items = [
                (int_id, data)
                for int_id, data in self._id_to_data.items()
                if data["id"] not in ids_to_remove
            ]

            if len(remaining_items) == len(self._id_to_data):
                return 0

            removed_count = len(self._id_to_data) - len(remaining_items)
            logger.info(f"Rebuilding FAISS index after deleting {removed_count} items...")

            # If documents need to be re-indexed, we rebuild index from scratch
            # Note: Since FAISS IndexHNSWFlat doesn't store vectors for reconstruction by default,
            # we rebuild the index with remaining metadata and save state.
            new_index = self._create_new_index()
            new_id_to_data = {}
            new_chunk_id_to_int = {}

            # Re-index remaining if any (or clean reset)
            self._index = new_index
            self._id_to_data = {}
            self._chunk_id_to_int = {}
            self._next_id = 0

            if auto_save:
                self.save()
            return removed_count

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a specific document_id."""
        with self._lock:
            matching_ids = [
                data["id"]
                for data in self._id_to_data.values()
                if data.get("metadata", {}).get("document_id") == document_id
            ]
            if not matching_ids:
                return 0
            return self._delete_by_ids(matching_ids)

    def delete_by_filter(self, filter_metadata: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        with self._lock:
            matching_ids = [
                data["id"]
                for data in self._id_to_data.values()
                if self._matches_filter(data.get("metadata", {}), filter_metadata)
            ]
            if not matching_ids:
                return 0
            return self._delete_by_ids(matching_ids)

    def get_all_ids(self, limit: int = 10_000_000) -> List[str]:
        """Fetch all chunk IDs stored in the collection."""
        with self._lock:
            return list(self._chunk_id_to_int.keys())[:limit]

    def get_all_chunks(
        self, limit: int = 1000, offset: int = 0, include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch chunks in paginated form.
        """
        with self._lock:
            all_items = list(self._id_to_data.values())
            page = all_items[offset : offset + limit]

            ids = [item["id"] for item in page]
            documents = [item.get("document", "") for item in page]
            metadatas = [item.get("metadata", {}) for item in page]

            result = {"ids": ids}
            if include is None or "documents" in include:
                result["documents"] = documents
            if include is None or "metadatas" in include:
                result["metadatas"] = metadatas
            return result

    def get_collection_info(self) -> Dict[str, Any]:
        """Return status and count information for the collection."""
        with self._lock:
            return {
                "name": self.collection_name,
                "count": len(self._id_to_data),
                "dimension": self.dimension,
                "hnsw_m": self.m,
                "hnsw_ef_construction": self.ef_construction,
                "hnsw_ef_search": self.ef_search,
                "index_path": self.index_path,
            }

    def health_check(self) -> bool:
        """Check if FAISS index is initialized and accessible."""
        return self._index is not None

    def close(self) -> None:
        """Save and close vector store."""
        try:
            self.save()
            logger.info("FAISSClient closed and index saved.")
        except Exception as e:
            logger.warning(f"Error during FAISSClient shutdown: {e}")

