"""
Indexer service: Chunks parsed documents, generates embeddings,
and indexes them into the local FAISS HNSW vector store.
"""
import logging
from typing import List, Dict, Any, Optional

from src.core.parser.chunker import DocumentChunker
from src.core.indexer.faiss_client import FAISSClient
from src.core.embedding.local_embedder import LocalEmbedder
from src.core.parser.pdf_parser import ParsedDocument
from src.config.settings import settings

logger = logging.getLogger(__name__)


class Indexer:
    """
    Indexer for legal documents using local Sentence-Transformers + FAISS HNSW.
    """

    def __init__(self, vector_client: Optional[Any] = None, embedder: Optional[Any] = None):
        self.chunker = DocumentChunker()
        self.vector_store = vector_client or FAISSClient()
        self.embedder = embedder or LocalEmbedder()
        self.embedding_batch_size = settings.EMBEDDING_BATCH_SIZE

        logger.info(
            f"Indexer initialized with {self.vector_store.__class__.__name__} "
            f"and {self.embedder.__class__.__name__}"
        )

    def _prepare_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare text for embedding by prepending heading context if available.
        """
        heading = chunk.get("heading", "")
        text = chunk.get("text", "")
        if heading:
            return f"{heading} | {text}"
        return text

    def index_document(
        self, parsed_doc: ParsedDocument, document_id: str, site_name: Optional[str] = None
    ) -> int:
        """
        Chunk the document, generate local embeddings, and index into FAISS.

        Args:
            parsed_doc: ParsedDocument object from the parser.
            document_id: Unique identifier for the document.
            site_name: Optional site or project label.

        Returns:
            Number of chunks indexed.
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

        logger.info(f"Generating embeddings for {len(chunks)} chunks (document {document_id})")

        # 2. Prepare texts with structural heading
        prepared_texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]

        # 3. Generate embeddings locally with Sentence-Transformers (batch-wise)
        embeddings = self.embedder.embed_documents(prepared_texts, batch_size=self.embedding_batch_size)

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        # 4. Prepare metadata and documents
        ids: List[str] = []
        doc_embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        site = site_name or parsed_doc.metadata.get("site_name", "unknown")

        for idx, chunk in enumerate(chunks):
            emb = chunk.get("embedding")
            if not emb:
                continue

            chunk_id = f"{document_id}_{chunk.get('section_id', 'sec')}_{idx}"
            ids.append(chunk_id)
            doc_embeddings.append(emb)

            metadata = {
                "section_id": chunk.get("section_id", "unknown"),
                "parent_section_id": chunk.get("parent_section_id"),
                "is_part": chunk.get("is_part", False),
                "part_number": chunk.get("part_number", 0),
                "total_parts": chunk.get("total_parts", 1),
                "heading": (chunk.get("heading", "") or "")[:200],
                "breadcrumb": chunk.get("breadcrumb", ""),
                "level": chunk.get("level", 0),
                "page": chunk.get("page", 1),
                "clause_type": chunk.get("clause_type", "general"),
                "document_id": document_id,
                "site_name": site,
            }
            metadatas.append(metadata)
            documents.append(self._prepare_text_for_embedding(chunk))

        if not ids:
            logger.warning(f"No valid chunks to index for document {document_id}")
            return 0

        # 5. Insert into vector store
        self.vector_store.add_documents(
            ids=ids,
            embeddings=doc_embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        logger.info(f"Successfully indexed {len(ids)} chunks for document {document_id}")
        return len(ids)

    def close(self) -> None:
        """Close any persistent connections."""
        if hasattr(self.vector_store, "close"):
            try:
                self.vector_store.close()
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")