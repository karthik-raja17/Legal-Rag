import asyncio
import hashlib
import logging
import pickle
import functools
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.indexer.chroma_client import ChromaClient
from src.core.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion
from src.core.embedding.bge_client import BGEEmbedderClient
from src.settings import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Production‑grade hybrid retriever combining dense (ChromaDB) and BM25
    keyword search via Reciprocal Rank Fusion (RRF).

    - BM25 index is cached on GCS (or local disk) and rebuilt only when the
      collection version changes (based on a checksum of all chunk IDs).
    - All I/O and CPU‑bound work are off‑loaded to threads / async.
    - Graceful fallback to dense search if BM25 fails.
    """

    def __init__(self, chroma_client: ChromaClient):
        """
        Initialize hybrid retriever with Chroma client and BGE embedder.
        """
        self.chroma_client = chroma_client
        self.bge_embedder = BGEEmbedderClient()
        self.collection_name = settings.CHROMA_COLLECTION

        # RRF constants
        self.rrf_k = settings.HYBRID_RRF_K
        self.dense_weight = settings.HYBRID_DENSE_WEIGHT
        self.bm25_weight = settings.HYBRID_BM25_WEIGHT
        self.default_top_k = settings.HYBRID_TOP_K

        # Cache configuration
        self.cache_bucket = settings.GCS_BUCKET_NAME
        self.cache_blob_prefix = "bm25_cache/"
        self.cache_ttl_seconds = getattr(settings, "BM25_CACHE_TTL_SECONDS", 86400)

        self._build_lock = asyncio.Lock()
        self._bm25_index: Optional[BM25Okapi] = None
        self._doc_ids: Optional[List[str]] = None
        self._doc_texts: Optional[List[str]] = None
        self._cache_version: Optional[str] = None
        self._cache_timestamp: Optional[datetime] = None
        self._storage_client = None

    def _get_storage_client(self):
        if self._storage_client is None:
            self._storage_client = storage.Client(project=settings.GCP_PROJECT_ID)
        return self._storage_client

    # -------------------------------------------------------------------------
    # Public async methods
    # -------------------------------------------------------------------------

    async def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with optional per‑call overrides.
        Returns top‑k results with fused scores.
        """
        top_k = top_k or self.default_top_k
        dense_w = dense_weight if dense_weight is not None else self.dense_weight
        bm25_w = bm25_weight if bm25_weight is not None else self.bm25_weight

        # 1. Dense search (always run; this is the primary, fast path)
        dense_results = await self._dense_search(query, top_k=top_k * 2, filter_metadata=filter_metadata)

        # 2. BM25 search (with fallback)
        try:
            bm25_results = await self._bm25_search(query, top_k=top_k * 2, filter_metadata=filter_metadata)
        except Exception as e:
            logger.warning(f"BM25 search failed, falling back to dense only: {e}")
            bm25_results = []

        # 3. If BM25 returned no results, just return dense results (already sorted)
        if not bm25_results:
            return dense_results[:top_k]

        # 4. Fuse using the external RRF function
        fused = await asyncio.to_thread(
            reciprocal_rank_fusion,
            [dense_results, bm25_results],          # list of result lists
            k=self.rrf_k,
            weights=[dense_w, bm25_w],
            merge_metadata_from="first",
        )
        # Normalize RRF scores to 0.0 - 1.0 (relative to theoretical maximum: (dense_w + bm25_w) / (k + 1))
        max_possible_rrf = (dense_w + bm25_w) / (self.rrf_k + 1)
        for item in fused:
            if "score" in item:
                item["score"] = min(1.0, item["score"] / max_possible_rrf)
                
        return fused[:top_k]

    # -------------------------------------------------------------------------
    # Dense search (non‑blocking via thread)
    # -------------------------------------------------------------------------

    async def _dense_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        loop = asyncio.get_running_loop()

        # Generate query embedding using BGE embedder (with instruction prefix)
        # The BGE client handles the instruction internally.
        query_vector = await loop.run_in_executor(
            None,
            self.bge_embedder.embed_query,
            query
        )

        # Use provided filter_metadata directly (can be dict with document_id, site_name, etc.)
        query_func = functools.partial(
            self.chroma_client.query,
            query_embedding=query_vector,
            n_results=top_k,
            filter_metadata=filter_metadata,
            include_documents=True,
        )
        raw_results = await loop.run_in_executor(None, query_func)

        # Transform ChromaDB response into a list of dicts
        results = []
        ids = raw_results.get("ids", [])
        distances = raw_results.get("distances", [])
        metadatas = raw_results.get("metadatas", [])
        documents = raw_results.get("documents", [])

        for i in range(len(ids)):
            results.append({
                "id": ids[i],
                "text": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else 1.0,
            })

        # Normalise distances to similarity scores (0..1)
        if results:
            max_dist = max(r["distance"] for r in results)
            for r in results:
                r["score"] = 1.0 - (r["distance"] / (max_dist + 1e-9))

        return results

    # -------------------------------------------------------------------------
    # BM25 search with caching / lazy rebuild & metadata filtering
    # -------------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _bm25_search(self, query: str, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        BM25 search using cached index.
        Supports metadata filtering by fetching top candidates globally,
        then filtering them post-hoc.
        """
        # Ensure the BM25 index is loaded and up‑to‑date
        await self._ensure_index()

        if self._bm25_index is None or not self._doc_ids:
            return []  # No index, return empty

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # --- Step 1: Fetch a larger pool globally ---
        # We fetch more than top_k so that after filtering we still have enough.
        fetch_k = max(top_k * 5, 200)

        def score_bm25():
            scores = self._bm25_index.get_scores(tokenized_query)
            # Pair with ids and texts
            scored = list(zip(self._doc_ids, self._doc_texts, scores))
            scored.sort(key=lambda x: x[2], reverse=True)
            return scored[:fetch_k]

        top_scored = await asyncio.to_thread(score_bm25)

        # --- Step 2: If no filter, just return top_k directly ---
        if not filter_metadata:
            return [
                {"id": doc_id, "text": text, "score": score}
                for doc_id, text, score in top_scored[:top_k]
            ]

        # --- Step 3: Apply post-hoc metadata filtering ---
        candidate_ids = [doc_id for doc_id, _, _ in top_scored]
        collection = self.chroma_client.client.get_collection(self.collection_name)
        meta_results = collection.get(ids=candidate_ids, include=["metadatas"])

        # Build a set of IDs that match the filter
        valid_ids = set()
        for i, doc_id in enumerate(meta_results.get("ids", [])):
            meta = meta_results["metadatas"][i] if i < len(meta_results["metadatas"]) else {}
            matches = True
            for key, value in filter_metadata.items():
                if isinstance(value, dict) and "$in" in value:
                    # Handle list filters (e.g., document_id: {"$in": [...]})
                    if meta.get(key) not in value["$in"]:
                        matches = False
                        break
                else:
                    # Handle exact match (e.g., site_name: "Lentilly")
                    if meta.get(key) != value:
                        matches = False
                        break
            if matches:
                valid_ids.add(doc_id)

        # Filter the scored list to only valid IDs
        filtered_results = [item for item in top_scored if item[0] in valid_ids]

        # Return top_k from the filtered list (or all if fewer)
        final_results = filtered_results[:top_k]
        return [
            {"id": doc_id, "text": text, "score": score}
            for doc_id, text, score in final_results
        ]

    async def _ensure_index(self):
        """Load BM25 index from GCS cache or rebuild if outdated/missing."""
        async with self._build_lock:
            # Quick check: if index exists and cache is fresh, return
            if self._bm25_index is not None and self._cache_timestamp is not None:
                if (datetime.utcnow() - self._cache_timestamp).total_seconds() < self.cache_ttl_seconds:
                    return

            # 1. Compute current version (checksum of all chunk IDs)
            current_version = await self._compute_collection_version()

            # 2. Try to load from GCS
            if current_version == self._cache_version:
                # Same version, but TTL might have expired – we can still use it
                if self._bm25_index is not None:
                    # Refresh timestamp
                    self._cache_timestamp = datetime.utcnow()
                    return

            # 3. Load from GCS if blob exists for this version
            blob_name = f"{self.cache_blob_prefix}{current_version}.pkl"
            bucket = self._get_storage_client().bucket(self.cache_bucket)
            blob = bucket.blob(blob_name)
            if blob.exists():
                logger.info(f"Loading BM25 cache from gs://{self.cache_bucket}/{blob_name}")
                data = await asyncio.to_thread(blob.download_as_bytes)
                index_data = pickle.loads(data)
                self._bm25_index = index_data["bm25"]
                self._doc_ids = index_data["doc_ids"]
                self._doc_texts = index_data["doc_texts"]
                self._cache_version = current_version
                self._cache_timestamp = datetime.utcnow()
                return

            # 4. Rebuild from scratch
            logger.info("BM25 cache miss or stale – rebuilding index...")
            await self._rebuild_index(current_version)

    async def _compute_collection_version(self) -> str:
        """Compute a version hash based on all document IDs in the collection."""
        ids = await asyncio.to_thread(
            self.chroma_client.get_all_ids,
            limit=10_000_000,
        )
        ids_sorted = sorted(ids)
        combined = "".join(ids_sorted)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def _rebuild_index(self, version: str):
        """Fetch all documents from ChromaDB, build BM25, and persist to GCS."""
        texts, ids = await self._load_all_documents()
        if not texts:
            logger.warning("No documents found in ChromaDB – BM25 index will be empty.")
            self._bm25_index = None
            self._doc_ids = []
            self._doc_texts = []
            return

        # Tokenize and build BM25 (CPU‑intensive)
        def build():
            tokenized_corpus = [self._tokenize(t) for t in texts]
            return BM25Okapi(tokenized_corpus)

        bm25 = await asyncio.to_thread(build)

        # Store in memory
        self._bm25_index = bm25
        self._doc_ids = ids
        self._doc_texts = texts
        self._cache_version = version
        self._cache_timestamp = datetime.utcnow()

        # Persist to GCS asynchronously (fire‑and‑forget to avoid blocking)
        asyncio.create_task(self._persist_index(version, bm25, ids, texts))

    async def _persist_index(self, version: str, bm25: BM25Okapi, ids: List[str], texts: List[str]):
        """Save the BM25 index to GCS."""
        try:
            data = pickle.dumps({
                "bm25": bm25,
                "doc_ids": ids,
                "doc_texts": texts,
            })
            blob_name = f"{self.cache_blob_prefix}{version}.pkl"
            bucket = self._get_storage_client().bucket(self.cache_bucket)
            blob = bucket.blob(blob_name)
            await asyncio.to_thread(
                blob.upload_from_string,
                data,
                content_type="application/octet-stream",
            )
            logger.info(f"BM25 index persisted to gs://{self.cache_bucket}/{blob_name}")
        except Exception as e:
            logger.error(f"Failed to persist BM25 index: {e}")

    async def _load_all_documents(self) -> Tuple[List[str], List[str]]:
        """Fetch all chunk texts and IDs from ChromaDB via pagination."""
        all_texts = []
        all_ids = []
        limit = 1000
        offset = 0

        while True:
            results = await asyncio.to_thread(
                self.chroma_client.get_all_chunks,
                limit=limit,
                offset=offset,
                include=["documents"],   # only need documents and ids
            )
            if not results.get("ids"):
                break
            all_ids.extend(results["ids"])
            all_texts.extend(results["documents"])
            offset += limit
            if len(results["ids"]) < limit:
                break

        return all_texts, all_ids

    # -------------------------------------------------------------------------
    # Tokenisation (French‑aware) — NOW KEEPS NUMBERS & CURRENCY
    # -------------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """
        Robust tokenizer for French legal text.
        - Lowercases
        - Keeps letters (with accents), digits, €, $, %, hyphens, apostrophes.
        - Removes generic punctuation (commas, semicolons, brackets, etc.)
        """
        import re
        # FIXED: Keep 0-9, €, $, % along with letters and hyphens/apostrophes
        cleaned = re.sub(r"[^a-zA-Zàâäéèêëîïôöùûüÿç0-9€$%\s'-]", " ", text)
        # Lowercase and split
        tokens = cleaned.lower().split()
        # Optional: remove stopwords (if you have a list)
        # from src.core.stopwords import FRENCH_STOPWORDS
        # tokens = [t for t in tokens if t not in FRENCH_STOPWORDS]
        return tokens