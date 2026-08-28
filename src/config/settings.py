"""
Settings and configuration for the Local Legal RAG Engine (English & CUAD).
100% Local: FAISS HNSW, Qwen3-Embedding-0.6B (MRL 512d), Ollama, and local storage.
"""
from typing import Optional, Dict

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Local application settings loaded from .env or system environment.
    """

    # ==================== VECTOR STORE (FAISS HNSW) ====================
    VECTOR_STORE_TYPE: str = Field(
        default="faiss",
        description="Vector store engine: 'faiss'"
    )
    FAISS_INDEX_DIR: str = Field(
        default="./data/faiss_index",
        description="Directory to persist FAISS index and chunk metadata"
    )
    FAISS_COLLECTION: str = Field(
        default="legal_contracts",
        description="FAISS collection / namespace identifier"
    )
    HNSW_M: int = Field(
        default=24,
        description="HNSW max connections per node (M=24)"
    )
    HNSW_EF_CONSTRUCTION: int = Field(
        default=100,
        description="HNSW construction-time search width (efConstruction=100)"
    )
    HNSW_EF_SEARCH: int = Field(
        default=100,
        description="HNSW query-time search depth (efSearch=100)"
    )

    # ==================== EMBEDDINGS (QWEN3-EMBEDDING 512d MRL) ====================
    EMBEDDING_PROVIDER: str = Field(
        default="local",
        description="Embedding provider: 'local' (SentenceTransformers)"
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Embedding model name"
    )
    EMBEDDING_DIMENSION: int = Field(
        default=512,
        description="Embedding vector dimension with MRL truncation (512)"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for generating embeddings"
    )
    EMBEDDING_DEVICE: Optional[str] = Field(
        default=None,
        description="Device for embeddings: 'cuda', 'cpu', or auto-detect"
    )

    # ==================== LOCAL LLM (OLLAMA) ====================
    LLM_PROVIDER: str = Field(
        default="ollama",
        description="LLM provider: 'ollama'"
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    OLLAMA_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Ollama model name (e.g., 'qwen2.5:7b', 'llama3.2', 'mistral')"
    )
    OLLAMA_TEMPERATURE: float = Field(
        default=0.1,
        description="Default temperature for Ollama generation"
    )
    OLLAMA_TIMEOUT: float = Field(
        default=120.0,
        description="Timeout in seconds for Ollama requests"
    )

    # ==================== LOCAL STORAGE & CACHE ====================
    LOCAL_STORAGE_DIR: str = Field(
        default="./data/storage",
        description="Local directory for storing uploaded PDFs, parsed documents, and status"
    )
    BM25_CACHE_DIR: str = Field(
        default="./data/bm25_cache",
        description="Directory for caching BM25 index on local disk"
    )
    PARSER_CACHE_DIR: Optional[str] = Field(
        default="./data/parser_cache",
        description="Base directory for caching parser outputs"
    )

    # ==================== CHUNKING ====================
    MAX_CHUNK_TOKENS: int = Field(
        default=512,
        description="Max tokens per chunk before splitting",
        gt=0
    )

    # ==================== HYBRID RETRIEVAL ====================
    HYBRID_DENSE_WEIGHT: float = Field(
        default=0.6,
        description="Weight for dense retrieval in RRF fusion"
    )
    HYBRID_BM25_WEIGHT: float = Field(
        default=0.4,
        description="Weight for BM25 retrieval in RRF fusion"
    )
    HYBRID_RRF_K: int = Field(
        default=60,
        description="RRF smoothing constant (typically 60)"
    )
    HYBRID_TOP_K: int = Field(
        default=5,
        description="Default number of final results to return from hybrid search"
    )
    BM25_CACHE_TTL_SECONDS: int = Field(
        default=86400,
        description="TTL for BM25 cache in seconds (default 24h)"
    )

    # ==================== RERANKER ====================
    RERANKER_TYPE: str = Field(
        default="none",
        description="Reranker type: 'none' or 'local_cross_encoder'"
    )

    # ==================== QUERY EXPANSION ====================
    ENABLE_QUERY_EXPANSION: bool = Field(
        default=False,
        description="Enable query expansion (generate multiple variants)"
    )
    QUERY_EXPANSION_NUM_VARIANTS: int = Field(
        default=3,
        description="Number of query variants to generate"
    )
    QUERY_EXPANSION_TEMPERATURE: float = Field(
        default=0.3,
        description="Temperature for query expansion"
    )

    # ==================== QUERY ANALYZER ====================
    QUERY_ANALYZER_ENABLED: bool = Field(
        default=True,
        description="Enable automatic query complexity analysis"
    )
    QUERY_ANALYZER_TOP_K_LOW: int = Field(
        default=5,
        description="top_k for low-complexity queries"
    )
    QUERY_ANALYZER_TOP_K_MEDIUM: int = Field(
        default=10,
        description="top_k for medium-complexity queries"
    )
    QUERY_ANALYZER_TOP_K_HIGH: int = Field(
        default=15,
        description="top_k for high-complexity queries"
    )
    QUERY_ANALYZER_EXPAND_MAP: Dict[str, bool] = Field(
        default={"low": False, "medium": False, "high": True},
        description="Whether to enable expansion per complexity level"
    )
    QUERY_ANALYZER_RERANK_MAP: Dict[str, bool] = Field(
        default={"low": False, "medium": False, "high": False},
        description="Whether to enable reranking per complexity level"
    )
    QUERY_ANALYZER_MODE: str = Field(
        default="heuristic",
        description="Query analysis mode: 'heuristic' or 'llm'"
    )
    QUERY_ANALYZER_LLM_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Model to use for LLM-based analysis"
    )

    # ==================== QUERY REWRITING ====================
    ENABLE_QUERY_REWRITING: bool = Field(
        default=True,
        description="Enable query rewriting (formal legal reformulation)"
    )
    REWRITER_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Model to use for query rewriting"
    )
    REWRITER_TEMPERATURE: float = Field(
        default=0.1,
        description="Temperature for rewriting"
    )
    REWRITER_CACHE_TTL_SECONDS: int = Field(
        default=3600,
        description="TTL for rewrite cache in seconds (1 hour)"
    )

    # ==================== OPTIONAL LOCAL DEDOC ====================
    DEDOC_SERVICE_URL: Optional[str] = Field(
        default="",
        description="URL of optional local Dedoc service (e.g. http://localhost:1231)"
    )

    # ==================== CONFIGURATION ====================
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Singleton instance
settings = Settings()