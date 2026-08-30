#!/usr/bin/env python3
"""
Step 1: Document Ingestion Pipeline
High-speed legal document ingestion (<0.03s/doc) using PyMuPDF + Stateful Regex Parser + LocalDocStore.
Preserves clause hierarchies (breadcrumbs: ARTICLE > SECTION > (b)) and Small-to-Big parent texts.
Supports:
  - All ~510 CUAD benchmark contracts (--all-cuad or --cuad)
  - PDF folders (Part_I, Part_II, Part_III)
  - Custom PDFs and directories
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import settings
from src.core.lightweight_parser import parse_and_chunk_contract, parse_and_chunk_text
from src.core.docstore import LocalDocStore
from src.core.parser.pdf_parser import PDFParser, ParsedDocument
from src.core.storage.local_storage import LocalStorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline.ingest")


def parse_and_store_lightweight(
    pdf_path: str,
    doc_id: str,
    category: str,
    storage: LocalStorageClient,
    docstore: LocalDocStore,
    force: bool = False
) -> Optional[Dict[str, Any]]:
    """
    High-speed parse and chunk (< 0.03s) with stateful regex breadcrumbs and Small-to-Big DocStore.
    """
    if not force:
        existing = storage.get_parsed_json(doc_id)
        if existing and "chunks" in existing:
            return existing

    try:
        t0 = time.time()
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        storage.save_pdf(doc_id, pdf_bytes)
        storage.update_status(doc_id, "parsing", f"Parsing PDF ({len(pdf_bytes)} bytes)...")

        chunks, full_text = parse_and_chunk_contract(str(pdf_path), doc_id=doc_id)

        parent_dict = {}
        for chunk in chunks:
            p_id = chunk.get("parent_id")
            p_text = chunk.get("parent_text")
            if p_id and p_text:
                parent_dict[p_id] = p_text

        if parent_dict:
            docstore.set_batch(parent_dict)

        elapsed = time.time() - t0

        doc_dict = {
            "document_id": doc_id,
            "raw_text": full_text,
            "chunks": chunks,
            "metadata": {
                "category": category,
                "site_name": category,
                "source_path": str(pdf_path),
                "num_chunks": len(chunks),
                "num_parents": len(parent_dict),
                "processing_time": elapsed,
                "parser": "lightweight_regex_ast"
            },
            "processing_time": elapsed
        }

        storage.save_parsed_json(doc_id, doc_dict)
        storage.update_status(doc_id, "parsed", f"Parsed in {elapsed:.4f}s ({len(chunks)} chunks)")
        return doc_dict

    except Exception as e:
        logger.error(f"❌ Failed to parse {pdf_path}: {e}", exc_info=True)
        storage.update_status(doc_id, "failed", f"Ingestion error: {str(e)}")
        return None


def parse_and_store_text_contract(
    doc_id: str,
    full_text: str,
    category: str,
    storage: LocalStorageClient,
    docstore: LocalDocStore,
    force: bool = False
) -> Optional[Dict[str, Any]]:
    """Parse and store a contract directly from full text in CUAD dataset (< 0.005s)."""
    if not force:
        existing = storage.get_parsed_json(doc_id)
        if existing and "chunks" in existing:
            return existing

    try:
        t0 = time.time()
        chunks, full_text = parse_and_chunk_text(full_text=full_text, doc_id=doc_id)

        parent_dict = {}
        for chunk in chunks:
            p_id = chunk.get("parent_id")
            p_text = chunk.get("parent_text")
            if p_id and p_text:
                parent_dict[p_id] = p_text

        if parent_dict:
            docstore.set_batch(parent_dict)

        elapsed = time.time() - t0

        doc_dict = {
            "document_id": doc_id,
            "raw_text": full_text,
            "chunks": chunks,
            "metadata": {
                "category": category,
                "site_name": category,
                "source_path": f"CUAD_v1:{doc_id}",
                "num_chunks": len(chunks),
                "num_parents": len(parent_dict),
                "processing_time": elapsed,
                "parser": "lightweight_regex_ast"
            },
            "processing_time": elapsed
        }

        storage.save_parsed_json(doc_id, doc_dict)
        storage.update_status(doc_id, "parsed", f"Parsed in {elapsed:.4f}s ({len(chunks)} chunks)")
        return doc_dict

    except Exception as e:
        logger.error(f"❌ Failed to parse text contract {doc_id}: {e}", exc_info=True)
        storage.update_status(doc_id, "failed", f"Ingestion error: {str(e)}")
        return None


def collect_cuad_pdf_files(
    cuad_base_dir: Path = Path("data/cuad/pdfs"),
    parts: Optional[List[str]] = None
) -> List[Tuple[Path, str]]:
    """Collect all PDF files from data/cuad/pdfs/ across Part_I, Part_II, and Part_III."""
    target_parts = parts or ["Part_I", "Part_II", "Part_III"]
    collected: List[Tuple[Path, str]] = []

    for part in target_parts:
        part_dir = cuad_base_dir / part
        if not part_dir.exists():
            continue

        for pdf_path in sorted(list(part_dir.rglob("*.pdf"))):
            category = pdf_path.parent.name
            if category == part:
                category = "General"
            collected.append((pdf_path, f"{part}/{category}"))

    return collected


def run_ingest(
    pdf_path: Optional[str] = None,
    dir_path: Optional[str] = None,
    cuad: bool = False,
    all_cuad_510: bool = True,
    parts: Optional[List[str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
    use_heavy_parser: bool = False,
) -> List[str]:
    """
    Main ingestion function.
    By default, ingests all ~510 CUAD contracts (combining raw PDFs and complete CUAD dataset).
    """
    storage = LocalStorageClient()
    docstore = LocalDocStore()

    successful_ids = []
    t_start = time.time()

    # 1. Collect PDFs on disk
    pdf_items: List[Tuple[Path, str]] = []
    if pdf_path:
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        pdf_items.append((p, "Custom"))
    elif dir_path:
        d = Path(dir_path)
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        for p in sorted(list(d.rglob("*.pdf"))):
            pdf_items.append((p, p.parent.name))
    else:
        # Collect all CUAD PDFs
        pdf_items = collect_cuad_pdf_files(
            cuad_base_dir=Path("data/cuad/pdfs"),
            parts=parts or ["Part_I", "Part_II", "Part_III"]
        )

    # 2. Ingest raw PDFs first
    logger.info(f"🚀 Ingesting {len(pdf_items)} PDF files from disk (Lightweight Parser)...")
    for idx, (fpath, category) in enumerate(pdf_items, start=1):
        doc_id = fpath.stem
        res = parse_and_store_lightweight(str(fpath), doc_id, category, storage, docstore, force=force)
        if res:
            successful_ids.append(doc_id)

    # 3. If all_cuad_510 is enabled, also ingest any remaining CUAD contracts from CUAD_v1.json
    if (all_cuad_510 or cuad) and os.path.exists("data/cuad/annotations/CUAD_v1.json"):
        with open("data/cuad/annotations/CUAD_v1.json", "r", encoding="utf-8") as f:
            cuad_data = json.load(f)

        already_ingested = set(successful_ids)
        cuad_docs = cuad_data.get("data", [])
        logger.info(f"📑 Checking full CUAD dataset ({len(cuad_docs)} total contracts)...")

        for doc in cuad_docs:
            title = doc.get("title", "")
            if not title:
                continue

            # Check if already covered by an ingested PDF stem
            if title in already_ingested:
                continue

            # Ingest text directly
            paragraphs = doc.get("paragraphs", [])
            if paragraphs:
                full_text = paragraphs[0].get("context", "")
                if full_text:
                    res = parse_and_store_text_contract(
                        doc_id=title,
                        full_text=full_text,
                        category="CUAD_Dataset",
                        storage=storage,
                        docstore=docstore,
                        force=force
                    )
                    if res:
                        successful_ids.append(title)

    if limit and limit > 0:
        successful_ids = successful_ids[:limit]

    total_time = time.time() - t_start
    avg_time = (total_time / max(len(successful_ids), 1))
    logger.info(
        f"🎉 Ingestion complete: {len(successful_ids)} contracts successfully parsed & stored in {total_time:.2f}s "
        f"(avg: {avg_time:.4f}s/doc)."
    )
    return successful_ids


def main():
    arg_parser = argparse.ArgumentParser(
        description="Ingest and parse contract PDFs from CUAD (all ~510 contracts) or custom folders into structured JSON."
    )
    arg_parser.add_argument("--pdf", type=str, default=None, help="Path to a single PDF contract.")
    arg_parser.add_argument("--dir", type=str, default=None, help="Directory containing PDF contracts.")
    arg_parser.add_argument("--cuad", action="store_true", help="Ingest all ~510 CUAD contracts.")
    arg_parser.add_argument(
        "--parts",
        nargs="+",
        default=None,
        choices=["Part_I", "Part_II", "Part_III"],
        help="Specify which CUAD parts to ingest."
    )
    arg_parser.add_argument("--limit", type=int, default=None, help="Maximum number of PDFs to ingest.")
    arg_parser.add_argument("--force", action="store_true", help="Force re-parsing even if already in cache/storage.")
    args = arg_parser.parse_args()

    run_ingest(
        pdf_path=args.pdf,
        dir_path=args.dir,
        cuad=args.cuad,
        all_cuad_510=True,
        parts=args.parts,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
