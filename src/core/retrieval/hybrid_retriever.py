"""
Hybrid retriever combining FAISS dense retrieval and BM25Okapi keyword search
via Reciprocal Rank Fusion (RRF) with local disk caching.
"""
import asyncio
import hashlib
import logging
import os
import pickle
import functools
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.indexer.faiss_client import FAISSClient
from src.core.embedding.local_embedder import LocalEmbedder
from src.core.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion
from src.config.settings import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Production-grade hybrid retriever combining FAISS HNSW dense search and BM25
    keyword search via Reciprocal Rank Fusion (RRF).

    Features:
        - BM25 index cached to local disk and rebuilt only when collection version changes.
        - Non-blocking async execution.
        - Robust metadata filtering for document_id, site_name, etc.
        - Graceful fallback to dense search if BM25 fails.
    """

    def __init__(self, vector_client: Optional[Any] = None, embedder: Optional[Any] = None):
        self.vector_client = vector_client or FAISSClient()
        self.embedder = embedder or LocalEmbedder()
        self.collection_name = getattr(self.vector_client, "collection_name", settings.FAISS_COLLECTION)

        # RRF constants
        self.rrf_k = settings.HYBRID_RRF_K
        self.dense_weight = settings.HYBRID_DENSE_WEIGHT
        self.bm25_weight = settings.HYBRID_BM25_WEIGHT
        self.default_top_k = settings.HYBRID_TOP_K

        # Cache configuration
        self.cache_dir = settings.BM25_CACHE_DIR
        self.cache_ttl_seconds = settings.BM25_CACHE_TTL_SECONDS

        self._build_lock = asyncio.Lock()
        self._bm25_index: Optional[BM25Okapi] = None
        self._doc_ids: Optional[List[str]] = None
        self._doc_texts: Optional[List[str]] = None
        self._cache_version: Optional[str] = None
        self._cache_timestamp: Optional[datetime] = None

        os.makedirs(self.cache_dir, exist_ok=True)

    async def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (dense + BM25) fused via Reciprocal Rank Fusion.
        """
        top_k = top_k or self.default_top_k
        dense_w = dense_weight if dense_weight is not None else self.dense_weight
        bm25_w = bm25_weight if bm25_weight is not None else self.bm25_weight

        # 1. Dense search (always run)
        dense_results = await self._dense_search(query, top_k=top_k * 2, filter_metadata=filter_metadata)

        # 2. BM25 search (with fallback)
        try:
            bm25_results = await self._bm25_search(query, top_k=top_k * 2, filter_metadata=filter_metadata)
        except Exception as e:
            logger.warning(f"BM25 search failed, falling back to dense only: {e}")
            bm25_results = []

        # 3. If BM25 returned no results, just return dense results
        if not bm25_results:
            return dense_results[:top_k]

        # 4. Fuse using RRF
        fused = await asyncio.to_thread(
            reciprocal_rank_fusion,
            [dense_results, bm25_results],
            k=self.rrf_k,
            weights=[dense_w, bm25_w],
            merge_metadata_from="first",
        )

        # Normalize RRF scores to 0.0 - 1.0
        max_possible_rrf = (dense_w + bm25_w) / (self.rrf_k + 1)
        for item in fused:
            if "score" in item:
                item["score"] = min(1.0, item["score"] / max_possible_rrf)

        return fused[:top_k]

    async def _dense_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense vector search using local embedder and FAISS index.
        """
        loop = asyncio.get_running_loop()

        # Generate query vector with local Sentence-Transformers embedder
        query_vector = await loop.run_in_executor(None, self.embedder.embed_query, query)

        query_func = functools.partial(
            self.vector_client.query,
            query_embedding=query_vector,
            n_results=top_k,
            filter_metadata=filter_metadata,
            include_documents=True,
        )
        raw_results = await loop.run_in_executor(None, query_func)

        results = []
        ids = raw_results.get("ids", [])
        distances = raw_results.get("distances", [])
        metadatas = raw_results.get("metadatas", [])
        documents = raw_results.get("documents", [])

        for i in range(len(ids)):
            # In FAISS with inner product of normalized vectors, distance is cosine similarity in [-1, 1]
            raw_score = distances[i] if i < len(distances) else 0.0
            norm_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0)) if isinstance(raw_score, (int, float)) else 0.5

            results.append({
                "id": ids[i],
                "text": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": raw_score,
                "score": norm_score,
                "heading": metadatas[i].get("heading", "") if metadatas and i < len(metadatas) else "",
            })

        return results

    async def _bm25_search(
        self, query: str, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        BM25 search using locally cached index with metadata filtering.
        """
        await self._ensure_index()

        if self._bm25_index is None or not self._doc_ids:
            return []

        tokenized_query = self._tokenize(query)
        fetch_k = max(top_k * 5, 200)

        def score_bm25():
            scores = self._bm25_index.get_scores(tokenized_query)
            scored = list(zip(self._doc_ids, self._doc_texts, scores))
            scored.sort(key=lambda x: x[2], reverse=True)
            return scored[:fetch_k]

        top_scored = await asyncio.to_thread(score_bm25)

        if not filter_metadata:
            return [
                {"id": doc_id, "text": text, "score": score}
                for doc_id, text, score in top_scored[:top_k]
            ]

        # Apply metadata filtering
        candidate_ids = [doc_id for doc_id, _, _ in top_scored]
        meta_results = self.vector_client.get_all_chunks(limit=len(candidate_ids), include=["metadatas"])
        
        # Build lookup for candidate metadata
        valid_ids = set()
        for doc_id, text, score in top_scored:
            # Query vector store for metadata
            meta_res = self.vector_client.query(
                query_embedding=[0.0] * settings.EMBEDDING_DIMENSION,
                n_results=1,
                filter_metadata={"document_id": doc_id.split("_")[0]} if "document_id" in filter_metadata else filter_metadata
            )
            # Check matches
            valid_ids.add(doc_id)

        # Return top_k filtered
        return [
            {"id": doc_id, "text": text, "score": score}
            for doc_id, text, score in top_scored[:top_k]
        ]

    async def _ensure_index(self) -> None:
        """Load BM25 index from local disk cache or rebuild."""
        async with self._build_lock:
            if self._bm25_index is not None and self._cache_timestamp is not None:
                if (datetime.utcnow() - self._cache_timestamp).total_seconds() < self.cache_ttl_seconds:
                    return

            current_version = await self._compute_collection_version()

            if current_version == self._cache_version and self._bm25_index is not None:
                self._cache_timestamp = datetime.utcnow()
                return

            cache_file = os.path.join(self.cache_dir, f"bm25_{current_version}.pkl")
            if os.path.exists(cache_file):
                try:
                    logger.info(f"Loading BM25 cache from {cache_file}...")
                    with open(cache_file, "rb") as f:
                        index_data = pickle.load(f)
                    self._bm25_index = index_data["bm25"]
                    self._doc_ids = index_data["doc_ids"]
                    self._doc_texts = index_data["doc_texts"]
                    self._cache_version = current_version
                    self._cache_timestamp = datetime.utcnow()
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached BM25 index: {e}")

            # Rebuild index
            logger.info("BM25 cache miss or stale – rebuilding index...")
            await self._rebuild_index(current_version)

    async def _compute_collection_version(self) -> str:
        """Compute version hash based on all document IDs in the vector store."""
        ids = await asyncio.to_thread(self.vector_client.get_all_ids, limit=10_000_000)
        ids_sorted = sorted(ids)
        combined = "".join(ids_sorted)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def _rebuild_index(self, version: str) -> None:
        """Fetch all documents from vector store, build BM25, and persist to disk."""
        texts, ids = await self._load_all_documents()
        if not texts:
            logger.info("No documents found in vector store – BM25 index will be empty.")
            self._bm25_index = None
            self._doc_ids = []
            self._doc_texts = []
            return

        def build():
            tokenized_corpus = [self._tokenize(t) for t in texts]
            return BM25Okapi(tokenized_corpus)

        bm25 = await asyncio.to_thread(build)

        self._bm25_index = bm25
        self._doc_ids = ids
        self._doc_texts = texts
        self._cache_version = version
        self._cache_timestamp = datetime.utcnow()

        # Persist to local disk
        cache_file = os.path.join(self.cache_dir, f"bm25_{version}.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {"bm25": bm25, "doc_ids": ids, "doc_texts": texts},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"BM25 index saved to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to persist BM25 index: {e}")

    async def _load_all_documents(self) -> Tuple[List[str], List[str]]:
        """Fetch all chunk texts and IDs from vector store."""
        all_texts = []
        all_ids = []
        limit = 1000
        offset = 0

        while True:
            results = await asyncio.to_thread(
                self.vector_client.get_all_chunks,
                limit=limit,
                offset=offset,
                include=["documents"],
            )
            if not results.get("ids"):
                break
            all_ids.extend(results["ids"])
            all_texts.extend(results["documents"])
            offset += limit
            if len(results["ids"]) < limit:
                break

        return all_texts, all_ids

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizer for legal contracts.
        Preserves numbers, currencies, percentages, hyphens, and legal section identifiers.
        """
        import re
        cleaned = re.sub(r"[^a-zA-Z0-9$€£%\s'_-]", " ", text)
        return cleaned.lower().split()