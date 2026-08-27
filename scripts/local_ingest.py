"""
CLI Tool for Local Document Ingestion into FAISS.

Usage:
  # Ingest a single PDF:
  python scripts/local_ingest.py --pdf data/sample_contract.pdf --doc-id bail_001 --site Site_Lyon

  # Ingest a folder of PDFs:
  python scripts/local_ingest.py --dir data/contracts/
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.parser.pdf_parser import PDFParser
from src.core.indexer.faiss_client import FAISSClient
from src.core.embedding.local_embedder import LocalEmbedder
from src.core.indexer.indexer import Indexer
from src.core.storage.local_storage import LocalStorageClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("local_ingest")


def ingest_file(
    file_path: str,
    doc_id: str,
    site_name: str,
    parser: PDFParser,
    indexer: Indexer,
    storage: LocalStorageClient,
) -> int:
    logger.info(f"Ingesting {file_path} (doc_id={doc_id}, site={site_name})...")
    with open(file_path, "rb") as f:
        content = f.read()

    # Save raw PDF to local storage
    storage.save_pdf(doc_id, content)

    # Parse document structure
    parsed_doc = parser.parse(content, doc_id)
    storage.save_parsed_json(doc_id, parsed_doc.to_dict())

    # Index into FAISS
    chunk_count = indexer.index_document(parsed_doc, doc_id, site_name=site_name)
    storage.update_status(doc_id, "indexed", "Indexed into FAISS", chunk_count=chunk_count)
    logger.info(f"✅ Ingested {doc_id}: {chunk_count} chunks indexed.")
    return chunk_count


def main():
    parser = argparse.ArgumentParser(description="Ingest local PDF contracts into FAISS index.")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF contract.")
    parser.add_argument("--dir", type=str, help="Path to directory containing PDF contracts.")
    parser.add_argument("--doc-id", type=str, default=None, help="Optional Document ID.")
    parser.add_argument("--site", type=str, default="local_site", help="Site / project name.")
    args = parser.parse_args()

    if not args.pdf and not args.dir:
        parser.error("Must specify either --pdf or --dir")

    faiss_client = FAISSClient()
    embedder = LocalEmbedder()
    indexer = Indexer(vector_client=faiss_client, embedder=embedder)
    doc_parser = PDFParser(use_ocr=False, extract_tables=True, semantic_enrichment=True)
    storage = LocalStorageClient()

    total_chunks = 0
    if args.pdf:
        if not os.path.exists(args.pdf):
            logger.error(f"File not found: {args.pdf}")
            sys.exit(1)
        doc_id = args.doc_id or os.path.splitext(os.path.basename(args.pdf))[0]
        total_chunks += ingest_file(args.pdf, doc_id, args.site, doc_parser, indexer, storage)
    elif args.dir:
        if not os.path.isdir(args.dir):
            logger.error(f"Directory not found: {args.dir}")
            sys.exit(1)
        pdf_files = list(Path(args.dir).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {args.dir}")
        for pdf_path in pdf_files:
            doc_id = os.path.splitext(pdf_path.name)[0]
            total_chunks += ingest_file(str(pdf_path), doc_id, args.site, doc_parser, indexer, storage)

    info = faiss_client.get_collection_info()
    logger.info(f"🎉 Ingestion complete! Total collection count: {info['count']} chunks in {info['index_path']}")


if __name__ == "__main__":
    main()

