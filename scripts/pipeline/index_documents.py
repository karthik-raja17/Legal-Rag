#!/usr/bin/env python3
"""
Step 2: Document Indexing Pipeline
Reads parsed contract JSONs from data/storage/parsed/, generates 512d MRL embeddings
with Qwen3-Embedding-0.6B for leaf chunks, and indexes vectors into FAISS HNSW (M=24, efC=100, efS=100).
Supports both lightweight Small-to-Big chunks and legacy AST structures.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import settings
from src.core.parser.pdf_parser import ParsedDocument
from src.core.parser.chunker import DocumentChunker
from src.core.embedding.local_embedder import LocalEmbedder
from src.core.indexer.faiss_client import FAISSClient
from src.core.indexer.indexer import Indexer
from src.core.storage.local_storage import LocalStorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline.index")


def index_lightweight_chunks(
    doc_id: str,
    doc_dict: Dict[str, Any],
    embedder: LocalEmbedder,
    faiss_client: FAISSClient
) -> int:
    """Index pre-chunked Small-to-Big leaf chunks into FAISS."""
    chunks = doc_dict.get("chunks", [])
    if not chunks:
        return 0

    ids = []
    texts_to_embed = []
    metadatas = []

    site_name = doc_dict.get("metadata", {}).get("site_name", "CUAD_Contracts")

    for c in chunks:
        leaf_id = c.get("leaf_id") or f"{doc_id}_{len(ids)}"
        leaf_text = c.get("leaf_text") or c.get("text", "")
        if not leaf_text.strip():
            continue

        ids.append(leaf_id)
        texts_to_embed.append(leaf_text)
        metadatas.append({
            "chunk_id": leaf_id,
            "document_id": doc_id,
            "parent_id": c.get("parent_id", ""),
            "breadcrumb": c.get("breadcrumb", ""),
            "site_name": site_name,
            "char_count": len(leaf_text)
        })

    if not ids:
        return 0

    embeddings = embedder.embed_documents(texts_to_embed, batch_size=16)

    faiss_client.add_documents(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts_to_embed
    )
    return len(ids)


def run_indexing(
    doc_ids: Optional[List[str]] = None,
    reset_index: bool = False,
    limit: Optional[int] = None
) -> int:
    """
    Index parsed documents from data/storage/parsed/ into the FAISS vector database.
    Returns total chunks in FAISS index.
    """
    storage = LocalStorageClient()
    faiss_client = FAISSClient()

    if reset_index:
        logger.warning("⚠️ Resetting FAISS collection before indexing...")
        faiss_client.reset_collection()

    embedder = LocalEmbedder()
    legacy_indexer = Indexer(vector_client=faiss_client, embedder=embedder)

    parsed_dir = Path("data/storage/parsed")
    if not parsed_dir.exists():
        logger.warning("No parsed documents directory found. Run ingest.py first.")
        return 0

    if doc_ids:
        target_files = [parsed_dir / f"{doc_id}.json" for doc_id in doc_ids if (parsed_dir / f"{doc_id}.json").exists()]
    else:
        target_files = sorted(list(parsed_dir.glob("*.json")))

    if limit and limit > 0:
        target_files = target_files[:limit]

    logger.info(f"🚀 Starting indexing for {len(target_files)} parsed contract(s)...")
    total_indexed_chunks = 0
    t_start = time.time()

    for idx, fpath in enumerate(target_files, start=1):
        doc_id = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                doc_dict = json.load(f)

            t0 = time.time()

            # Path A: Lightweight parser output
            if "chunks" in doc_dict:
                chunk_count = index_lightweight_chunks(doc_id, doc_dict, embedder, faiss_client)
            # Path B: Legacy AST parsed output
            else:
                parsed_doc = ParsedDocument(
                    document_id=doc_id,
                    metadata=doc_dict.get("metadata", {}),
                    structure=doc_dict.get("structure", {}),
                    elements=doc_dict.get("elements", {}),
                    raw_text=doc_dict.get("raw_text", ""),
                    processing_time=doc_dict.get("processing_time", 0.0),
                    errors=doc_dict.get("errors", []),
                    warnings=doc_dict.get("warnings", []),
                    ocr_used=doc_dict.get("ocr_used", False),
                )
                chunk_count = legacy_indexer.index_document(
                    parsed_doc,
                    document_id=doc_id,
                    site_name=doc_dict.get("metadata", {}).get("site_name", "CUAD_Contracts")
                )

            elapsed = time.time() - t0
            storage.update_status(doc_id, "indexed", f"Indexed {chunk_count} chunks into FAISS in {elapsed:.2f}s", chunk_count=chunk_count)
            total_indexed_chunks += chunk_count

            if idx % 10 == 0 or idx == len(target_files):
                logger.info(f"[{idx}/{len(target_files)}] Indexed {doc_id}: {chunk_count} chunks (Total so far: {total_indexed_chunks})")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ Failed to index {doc_id}: {e}", exc_info=True)
            storage.update_status(doc_id, "failed", f"Indexing error: {str(e)}")

    total_time = time.time() - t_start
    info = faiss_client.get_collection_info()
    logger.info(
        f"🎉 Indexing complete! FAISS Collection '{info['name']}' now has {info['count']} total chunks "
        f"across {len(target_files)} contracts in {total_time:.2f}s."
    )
    return info["count"]


def main():
    arg_parser = argparse.ArgumentParser(description="Index parsed contracts into the FAISS HNSW vector database.")
    arg_parser.add_argument("--doc-ids", nargs="+", default=None, help="Specific document IDs to index.")
    arg_parser.add_argument("--reset", action="store_true", help="Reset and clear existing FAISS index before indexing.")
    arg_parser.add_argument("--limit", type=int, default=None, help="Maximum number of parsed files to index.")
    args = arg_parser.parse_args()

    run_indexing(
        doc_ids=args.doc_ids,
        reset_index=args.reset,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
