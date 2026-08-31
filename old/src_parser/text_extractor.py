"""
Layer 1: Local Text Extraction using PyMuPDF.
Extracts clean per-page text from PDF files with local caching and optional local OCR fallback.
"""
import io
import json
import logging
import hashlib
import os
from typing import Optional, Dict, Any, List
from filelock import FileLock

import fitz  # PyMuPDF
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Local PDF Text Extractor using PyMuPDF.
    Extracts text per-page with character counts, page numbers, and optional disk caching.
    """

    def __init__(
        self,
        use_ocr: bool = False,
        ocr_threshold_chars_per_page: int = 50,
        always_use_ocr: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        self.use_ocr = use_ocr
        self.ocr_threshold = ocr_threshold_chars_per_page
        self.always_use_ocr = always_use_ocr
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "ocr_cache.json")
            self.cache_lock_file = self.cache_file + ".lock"
            self._ensure_cache_file()

    def _ensure_cache_file(self) -> None:
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _get_pdf_hash(self, pdf_content: bytes) -> str:
        return hashlib.sha256(pdf_content).hexdigest()

    def _load_from_cache(self, pdf_hash: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            return cache.get(pdf_hash)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_to_cache(self, pdf_hash: str, result: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        try:
            with FileLock(self.cache_lock_file, timeout=10):
                cache = {}
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        cache = json.load(f)
                cache[pdf_hash] = result
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write extraction cache: {e}")

    def extract(self, pdf_content: bytes, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from PDF bytes.
        """
        pdf_hash = self._get_pdf_hash(pdf_content)
        cached_result = self._load_from_cache(pdf_hash)
        if cached_result is not None:
            return cached_result

        result = self._extract_with_pymupdf(pdf_content)
        if result.get("error") is None:
            self._save_to_cache(pdf_hash, result)

        return result

    def _extract_with_pymupdf(self, pdf_content: bytes) -> Dict[str, Any]:
        result = {
            "pages": [],
            "text": "",
            "ocr_used": False,
            "error": None,
            "total_pages": 0,
        }

        try:
            doc = fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf")
            pages = []
            total_chars = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text().strip()

                # If page is empty and local OCR is available, attempt pytesseract
                if not page_text and self.use_ocr:
                    page_text = self._ocr_page_fallback(page)

                pages.append({
                    "page_num": page_num + 1,
                    "text": page_text,
                    "confidence": 1.0 if page_text else 0.0,
                })
                total_chars += len(page_text)

            total_pages = len(pages)
            result["pages"] = pages
            result["total_pages"] = total_pages
            result["text"] = "\n\n".join(p["text"] for p in pages if p["text"])
            doc.close()

            avg_chars = total_chars / total_pages if total_pages > 0 else 0
            logger.info(f"PyMuPDF extracted {total_chars} chars across {total_pages} pages (avg {avg_chars:.1f}/page)")
            return result

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result

    def _ocr_page_fallback(self, page: fitz.Page) -> str:
        """Optional local fallback using pytesseract for scanned pages."""
        try:
            import pytesseract
            from PIL import Image
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="fra")
            return text.strip()
        except Exception:
            return ""