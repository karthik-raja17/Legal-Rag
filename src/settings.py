"""
Settings and configuration for the Legal RAG system.
Uses pydantic-settings with environment variable overrides.
"""
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables (`.env` file or system env).

    All fields can be overridden by setting environment variables with the same name
    (e.g., `GCP_PROJECT_ID=my-project`).
    """

    # ==================== GCP PROJECT ====================
    GCP_PROJECT_ID: str = Field(
        default="your-gcp-project-id",
        description="Google Cloud Project ID"
    )

    # ==================== GCP LOCATIONS ====================
    GCP_LOCATION: str = Field(
        default="europe-west9",
        description="Default GCP region for Vertex AI, Compute, and other services"
    )

    DEDOC_SERVICE_URL: Optional[str] = Field(
        default="",
        description="URL of the Dedoc Cloud Run service"
    )

    # Document AI processor location (must be 'eu' or 'us' for multi-region processors)
    DOCUMENT_AI_LOCATION: str = Field(
        default="eu",
        description="GCP location of the Document AI processor (typically 'eu' or 'us')"
    )

    DOCUMENT_AI_PROCESSOR_ID: str = Field(
        default="your-document-ai-processor-id",
        description="Processor ID for Document AI OCR"
    )

    # ==================== DRIVE & STORAGE ====================
    DRIVE_EXCEL_FILE_ID: Optional[str] = Field(
        default=None,
        description="Google Drive file ID of the master Excel file (optional until Drive sync is wired up)"
    )

    DRIVE_PDF_FOLDER_ID: Optional[str] = Field(
        default=None,
        description="Google Drive folder ID for storing/reading PDF contracts"
    )
    GCS_BUCKET_NAME: str = Field(
        default="your-gcs-bucket-name",
        description="GCS bucket name for storing PDFs and parsed results"
    )

    # ==================== FIRESTORE & CHROMA ====================
    FIRESTORE_COLLECTION: str = Field(
        default="contract_state",
        description="Firestore collection name for document state tracking"
    )
    CHROMA_HOST: str = Field(
        default="localhost",
        description="Chroma vector database host"
    )
    CHROMA_PORT: int = Field(
        default=8000,
        description="Chroma vector database port"
    )
    CHROMA_COLLECTION: str = Field(
        default="legal_contracts",
        description="Chroma collection name for document embeddings"
    )

    # ==================== VERTEX AI ====================
    VERTEX_AI_EMBEDDING_MODEL: str = Field(
        default="text-multilingual-embedding-002",
        description="Vertex AI embedding model name"
    )
    VERTEX_AI_LLM_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Vertex AI LLM model name for generation"
    )

    # ==================== CHUNKING ====================
    MAX_CHUNK_TOKENS: int = Field(
        default=512,  # headroom below Vertex's hard 2048 cap
        description="Max tokens per chunk before falling back to paragraph-level splitting",
        gt=0
    )

    # ==================== PUB/SUB ====================
    PUBSUB_TOPIC_ID: str = Field(
        default="sync-requests",
        description="Pub/Sub topic for sync requests"
    )
    PUBSUB_SUBSCRIPTION_ID: str = Field(
        default="sync-subscription",
        description="Pub/Sub subscription for sync requests"
    )

    # ==================== PARSING CACHE ====================
    PARSER_CACHE_DIR: Optional[str] = Field(
        default="./parser_cache",
        description="Base directory for caching parser outputs across layers (set to None to disable)"
    )

    HNSW_M: int = Field(
        default=128,
        description="HNSW max neighbors per node (higher = better recall, more memory)"
    )
    HNSW_EF_CONSTRUCTION: int = Field(
        default=1000,
        description="HNSW construction-time search width (higher = better index quality, slower build)"
    )

    HNSW_EF_SEARCH: int = Field(
        default=2000,
        description="HNSW query-time search depth (higher = better recall, slower query)"
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
        description="TTL for BM25 cache on GCS (in seconds, default 24h)"
    )

    # ==================== VERTEX AI RERANKER ====================
    VERTEX_RERANKER_ENABLED: bool = Field(
        default=True,
        description="Enable Vertex AI reranking (cross-encoder) for /query"
    )
    VERTEX_RERANKER_LOCATION: str = Field(
        default="global",
        description="Location for Vertex AI Ranking API ('global' or 'us')"
    )
    VERTEX_RERANKER_MODEL: str = Field(
        default="semantic-ranker-512-004",
        description="Ranking model name (semantic-ranker-512-004 or semantic-ranker-512-003)"
    )
    VERTEX_RERANKER_TOP_N: int = Field(
        default=5,
        description="Number of chunks to return after reranking"
    )
    VERTEX_RERANKER_CANDIDATE_K: int = Field(
        default=75,
        description="Number of candidates to retrieve before reranking"
    )

    # ==================== QUERY EXPANSION ====================
    ENABLE_QUERY_EXPANSION: bool = Field(
        default=True,
        description="Enable Gemini-based query expansion (generate multiple variants)"
    )
    QUERY_EXPANSION_NUM_VARIANTS: int = Field(
        default=3,
        description="Number of query variants to generate (including original)"
    )
    QUERY_EXPANSION_TEMPERATURE: float = Field(
        default=0.3,
        description="Temperature for Gemini query expansion (lower = more focused)"
    )
    QUERY_EXPANSION_MAX_VARIANTS: int = Field(
        default=3,
        description="Maximum number of variants to generate"
    )

    # ==================== QUERY ANALYZER ====================
    QUERY_ANALYZER_ENABLED: bool = Field(
        default=True,
        description="Enable automatic query complexity analysis and adaptive parameters"
    )
    QUERY_ANALYZER_TOP_K_LOW: int = Field(
        default=7,
        description="top_k for low-complexity queries"
    )
    QUERY_ANALYZER_TOP_K_MEDIUM: int = Field(
        default=13,
        description="top_k for medium-complexity queries"
    )
    QUERY_ANALYZER_TOP_K_HIGH: int = Field(
        default=19,
        description="top_k for high-complexity queries"
    )
    QUERY_ANALYZER_EXPAND_MAP: dict = Field(
        default={"low": False, "medium": True, "high": True},
        description="Whether to enable expansion per complexity level"
    )
    QUERY_ANALYZER_RERANK_MAP: dict = Field(
        default={"low": True, "medium": True, "high": True},
        description="Whether to enable reranking per complexity level (default: always true)"
    )
    QUERY_ANALYZER_MODE: str = Field(
        default="heuristic",
        description="Query analysis mode: 'heuristic' or 'llm'"
    )
    QUERY_ANALYZER_LLM_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Model to use for LLM-based analysis"
    )

    # ==================== QUERY REWRITING ====================
    ENABLE_QUERY_REWRITING: bool = Field(
        default=True,
        description="Enable Gemini-based query rewriting (formal legal reformulation)"
    )
    REWRITER_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Model to use for query rewriting"
    )
    REWRITER_TEMPERATURE: float = Field(
        default=0.2,
        description="Temperature for rewriting (low = deterministic)"
    )
    REWRITER_CACHE_TTL_SECONDS: int = Field(
        default=3600,
        description="TTL for rewrite cache in seconds (1 hour)"
    )

    # ==================== VALIDATION ====================
    @field_validator("DOCUMENT_AI_LOCATION")
    def validate_document_ai_location(cls, v: str) -> str:
        """Document AI processor location must be 'eu' or 'us' for multi-region."""
        if v not in ("eu", "us"):
            raise ValueError("DOCUMENT_AI_LOCATION must be 'eu' or 'us' for multi-region processors")
        return v

    @field_validator("GCP_PROJECT_ID")
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("This field cannot be empty")
        return v.strip()

    # ==================== COMPUTED PROPERTIES ====================
    @property
    def document_ai_processor_name(self) -> str:
        """Full resource name of the Document AI processor."""
        return (f"projects/{self.GCP_PROJECT_ID}/locations/{self.DOCUMENT_AI_LOCATION}"
                f"/processors/{self.DOCUMENT_AI_PROCESSOR_ID}")

    # ==================== CONFIGURATION ====================
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # Allow extra fields for future expansion
        extra = "ignore"


# Singleton instance
settings = Settings()