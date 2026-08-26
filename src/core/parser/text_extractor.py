"""
Layer 1: Text & OCR Extraction
Handles both digital PDFs and scanned documents with OCR fallback.
Supports synchronous Document AI for documents <= 40 MB and asynchronous batch Document AI for large documents > 40 MB.
Includes local thread-safe/process-safe caching via FileLock.
"""
import io
import json
import logging
import hashlib
import os
import re
from typing import Optional, Dict, Any, List
from filelock import FileLock

import fitz  # PyMuPDF
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.api_core.exceptions import GoogleAPICallError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Document AI synchronous limit is 40 MB
DOCAI_SYNC_MAX_BYTES = 40 * 1024 * 1024


class TextExtractor:
    """
    Extracts text from PDF bytes with per-page granularity and optional OCR cache.

    Strategy:
    1. Compute SHA-256 hash of PDF bytes.
    2. Check local FileLock cache.
    3. Attempt digital extraction via PyMuPDF.
    4. If text is insufficient:
       - If payload <= 40 MB: call Synchronous Document AI.
       - If payload > 40 MB: call Asynchronous Document AI Batch API via GCS staging.
    5. Cache and return normalized result.
    """

    def __init__(
        self,
        use_ocr: bool = True,
        ocr_threshold_chars_per_page: int = 50,
        always_use_ocr: bool = False,
        cache_dir: Optional[str] = None,
        async_timeout_seconds: int = 1800,
    ):
        self.use_ocr = use_ocr
        self.ocr_threshold = ocr_threshold_chars_per_page
        self.always_use_ocr = always_use_ocr
        self.cache_dir = cache_dir
        self.async_timeout_seconds = async_timeout_seconds

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "ocr_cache.json")
            self.cache_lock_file = self.cache_file + ".lock"
            self._ensure_cache_file()

        if self.use_ocr or self.always_use_ocr:
            location = settings.DOCUMENT_AI_LOCATION
            endpoint = f"{location}-documentai.googleapis.com" if location != "global" else "documentai.googleapis.com"
            client_options = {"api_endpoint": endpoint}
            self.documentai_client = documentai.DocumentProcessorServiceClient(
                client_options=client_options
            )
            self.processor_name = (
                f"projects/{settings.GCP_PROJECT_ID}/locations/{settings.DOCUMENT_AI_LOCATION}"
                f"/processors/{settings.DOCUMENT_AI_PROCESSOR_ID}"
            )
            self.storage_client = storage.Client(project=settings.GCP_PROJECT_ID)

    def _ensure_cache_file(self) -> None:
        """Create an empty cache file if it does not exist."""
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _get_pdf_hash(self, pdf_content: bytes) -> str:
        """Compute SHA-256 hash of the PDF bytes."""
        return hashlib.sha256(pdf_content).hexdigest()

    def _load_from_cache(self, pdf_hash: str) -> Optional[Dict[str, Any]]:
        """Load cached result for a given PDF hash, or None if not found."""
        if not self.cache_dir:
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if pdf_hash in cache:
                logger.info(f"Cache hit for PDF hash {pdf_hash[:8]}...")
                return cache[pdf_hash]
            logger.info(f"Cache miss for PDF hash {pdf_hash[:8]}...")
            return None
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache: {e}")
            return None

    def _save_to_cache(self, pdf_hash: str, result: Dict[str, Any]) -> None:
        """Save the extraction result to the cache under file lock protection."""
        if not self.cache_dir:
            return
        try:
            with FileLock(self.cache_lock_file, timeout=10):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                cache[pdf_hash] = result
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            logger.info(f"Cached result for PDF hash {pdf_hash[:8]}...")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to write cache: {e}")
        except TimeoutError:
            logger.warning(f"Cache lock timeout for hash {pdf_hash[:8]} - skipping cache write")

    def extract(self, pdf_content: bytes, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from PDF bytes with caching.

        Args:
            pdf_content: Raw PDF bytes.
            document_id: Optional document identifier for tracking and staging paths.

        Returns:
            Dict containing:
            - pages: List[Dict] with 'page_num', 'text', 'confidence'
            - text: Full text combined
            - ocr_used: bool
            - error: Optional error string
            - total_pages: int
        """
        pdf_hash = self._get_pdf_hash(pdf_content)
        cached_result = self._load_from_cache(pdf_hash)
        if cached_result is not None:
            return cached_result

        doc_ref = document_id or pdf_hash[:16]
        result = self._extract_without_cache(pdf_content, doc_ref)

        if result.get("error") is None:
            self._save_to_cache(pdf_hash, result)

        return result

    def _extract_without_cache(self, pdf_content: bytes, document_id: str) -> Dict[str, Any]:
        result = {
            "pages": [],
            "text": "",
            "ocr_used": False,
            "error": None,
            "total_pages": 0,
        }

        try:
            # Force OCR mode check
            if self.always_use_ocr and self.use_ocr:
                temp_pages = self._extract_with_pymupdf_per_page(pdf_content)
                page_count = len(temp_pages) if temp_pages else 0
                return self._route_documentai(pdf_content, document_id, page_count)

            # 1. Attempt PyMuPDF text extraction
            pymu_pages = self._extract_with_pymupdf_per_page(pdf_content)
            if pymu_pages is None:
                if self.use_ocr:
                    return self._route_documentai(pdf_content, document_id)
                result["error"] = "PyMuPDF failed and OCR is disabled"
                return result

            total_pages = len(pymu_pages)
            result["total_pages"] = total_pages
            total_chars = sum(len(page["text"]) for page in pymu_pages)
            avg_chars = total_chars / total_pages if total_pages > 0 else 0

            # Low-text threshold check for scanned PDF detection
            use_ocr_fallback = (
                self.use_ocr
                and avg_chars < self.ocr_threshold
                and total_chars < 1000
            )

            if not use_ocr_fallback:
                result["pages"] = pymu_pages
                result["text"] = "\n\n".join(p["text"] for p in pymu_pages)
                result["ocr_used"] = False
                logger.info(f"PyMuPDF extracted {total_chars} chars across {total_pages} pages (avg {avg_chars:.1f}/page)")
                return result

            logger.info(f"Low text detected ({avg_chars:.1f} chars/page) - routing to Document AI")
            return self._route_documentai(pdf_content, document_id, page_count=total_pages)

        except Exception as e:
            logger.error(f"Text extraction failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result

    def _extract_with_pymupdf_per_page(self, pdf_content: bytes) -> Optional[List[Dict[str, Any]]]:
        """Extract text per page using PyMuPDF."""
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf")
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pages.append({
                    "page_num": page_num + 1,
                    "text": page.get_text().strip(),
                    "confidence": 1.0,
                })
            doc.close()
            return pages
        except Exception as e:
            logger.warning(f"PyMuPDF per-page extraction failed: {e}")
            return None

    def _route_documentai(self, pdf_content: bytes, document_id: str, page_count: int=0) -> Dict[str, Any]:
        """Route to synchronous or asynchronous Document AI API based on payload size."""
        file_size = len(pdf_content)
        if file_size > DOCAI_SYNC_MAX_BYTES or page_count > 30:
            logger.info(f"Payload size ({file_size / (1024 * 1024):.2f} MB) exceeds 40 MB - using Async Batch Document AI")
            return self._extract_with_documentai_async(pdf_content, document_id)
        
        try:
            return self._extract_with_documentai_sync(pdf_content)
        except GoogleAPICallError as e:
            # If synchronous API fails due to size or page count limits, fallback to async
            if "exceeds" in str(e).lower() or "too large" in str(e).lower() or e.code == 413:
                logger.warning("Synchronous Document AI rejected payload - falling back to Async Document AI")
                return self._extract_with_documentai_async(pdf_content, document_id)
            return {"pages": [], "text": "", "ocr_used": True, "error": f"Document AI OCR failed: {e}", "total_pages": 0}

    # -------------------------------------------------------------------------
    # Synchronous Document AI (<= 40 MB)
    # -------------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _extract_with_documentai_sync(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process document using synchronous Document AI API."""
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf",
        )
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document,
        )

        response = self.documentai_client.process_document(request=request)
        document = response.document

        pages_data = []
        for page in document.pages:
            page_text = page.text if hasattr(page, "text") and page.text else ""
            
            # If page.text is empty, extract text via layout text anchors
            if not page_text and hasattr(page, "layout") and page.layout.text_anchor:
                segments = page.layout.text_anchor.text_segments
                page_text = "".join(document.text[s.start_index:s.end_index] for s in segments if s.end_index)

            confidence = getattr(page.layout, "confidence", None) if hasattr(page, "layout") else None
            pages_data.append({
                "page_num": page.page_number,
                "text": page_text.strip(),
                "confidence": confidence,
            })

        if not pages_data and document.text:
            pages_data.append({
                "page_num": 1,
                "text": document.text.strip(),
                "confidence": None,
            })

        full_text = "\n\n".join(p["text"] for p in pages_data)
        logger.info(f"Document AI Sync extracted {len(full_text)} chars across {len(pages_data)} pages")

        return {
            "pages": pages_data,
            "text": full_text,
            "ocr_used": True,
            "error": None,
            "total_pages": len(pages_data),
        }

    # -------------------------------------------------------------------------
    # Asynchronous Batch Document AI (> 40 MB)
    # -------------------------------------------------------------------------
    def _extract_with_documentai_async(self, pdf_content: bytes, document_id: str) -> Dict[str, Any]:
        """
        Process large documents asynchronously via Document AI Batch API.
        Stages input PDF in GCS, executes batch operation, merges output JSON shards,
        and cleans up temporary GCS blobs.
        """
        sanitized_doc_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", document_id)
        staging_prefix = f"docai_staging/{sanitized_doc_id}"
        input_blob_name = f"{staging_prefix}/input.pdf"
        output_prefix = f"{staging_prefix}/output/"
        
        gcs_bucket_name = settings.GCS_BUCKET_NAME
        input_uri = f"gs://{gcs_bucket_name}/{input_blob_name}"
        output_uri = f"gs://{gcs_bucket_name}/{output_prefix}"

        bucket = self.storage_client.bucket(gcs_bucket_name)

        try:
            # 1. Upload raw PDF to GCS staging
            logger.info(f"Uploading {len(pdf_content) / (1024 * 1024):.2f} MB to {input_uri}")
            input_blob = bucket.blob(input_blob_name)
            input_blob.upload_from_string(pdf_content, content_type="application/pdf")

            # 2. Build BatchProcessRequest
            gcs_document = documentai.GcsDocument(
                gcs_uri=input_uri,
                mime_type="application/pdf",
            )
            input_config = documentai.BatchDocumentsInputConfig(
                gcs_documents=documentai.GcsDocuments(documents=[gcs_document])
            )
            output_config = documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=output_uri
                )
            )
            request = documentai.BatchProcessRequest(
                name=self.processor_name,
                input_documents=input_config,
                document_output_config=output_config,
            )

            # 3. Trigger Async Operation
            logger.info(f"Triggering Document AI Batch operation for {document_id}")
            operation = self.documentai_client.batch_process_documents(request=request)
            logger.info(f"Document AI Batch LRO started: {operation.operation.name}")
            
            # Block until completed with configured timeout
            operation.result(timeout=self.async_timeout_seconds)
            logger.info(f"Document AI Batch LRO finished successfully for {document_id}")

            # 4. Read and aggregate JSON output shards
            pages_data = []
            blobs = list(self.storage_client.list_blobs(gcs_bucket_name, prefix=output_prefix))
            json_blobs = [b for b in blobs if b.name.endswith(".json")]

            if not json_blobs:
                raise RuntimeError(f"No JSON output shards found under {output_uri}")

            # Sort shards by name to maintain document page continuity
            for blob in sorted(json_blobs, key=lambda b: b.name):
                content_str = blob.download_as_text()
                doc_dict = json.loads(content_str)
                shard_doc = documentai.Document.from_json(json.dumps(doc_dict))

                shard_text = shard_doc.text or ""
                for page in shard_doc.pages:
                    page_text = ""
                    if hasattr(page, "layout") and page.layout.text_anchor:
                        for segment in page.layout.text_anchor.text_segments:
                            start = segment.start_index or 0
                            end = segment.end_index or len(shard_text)
                            page_text += shard_text[start:end]
                    elif hasattr(page, "text") and page.text:
                        page_text = page.text

                    confidence = getattr(page.layout, "confidence", None) if hasattr(page, "layout") else None
                    pages_data.append({
                        "page_num": page.page_number,
                        "text": page_text.strip(),
                        "confidence": confidence,
                    })

            # Sort all pages by page number across shards
            pages_data.sort(key=lambda p: p["page_num"])
            full_text = "\n\n".join(p["text"] for p in pages_data)

            logger.info(
                f"Document AI Async completed: {len(full_text)} chars extracted "
                f"across {len(pages_data)} pages from {len(json_blobs)} shards"
            )

            return {
                "pages": pages_data,
                "text": full_text,
                "ocr_used": True,
                "error": None,
                "total_pages": len(pages_data),
            }

        except Exception as e:
            logger.error(f"Document AI Async Batch failed for {document_id}: {e}", exc_info=True)
            return {
                "pages": [],
                "text": "",
                "ocr_used": True,
                "error": f"Async Document AI Batch error: {str(e)}",
                "total_pages": 0,
            }

        finally:
            # 5. Clean up temporary staging blobs on GCS
            self._cleanup_staging_blobs(bucket, staging_prefix)

    def _cleanup_staging_blobs(self, bucket: storage.Bucket, staging_prefix: str) -> None:
        """Delete temporary staging files created for the batch OCR request."""
        try:
            blobs_to_delete = list(bucket.list_blobs(prefix=staging_prefix))
            if blobs_to_delete:
                bucket.delete_blobs(blobs_to_delete)
                logger.info(f"Cleaned up {len(blobs_to_delete)} staging blobs under {staging_prefix}")
        except Exception as e:
            logger.warning(f"Failed to clean up staging blobs under {staging_prefix}: {e}")