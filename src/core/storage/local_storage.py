"""
Local storage and document status manager for personal / offline execution.
Manages PDFs, parsed JSON results, and indexing status on the local filesystem.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from src.config.settings import settings

logger = logging.getLogger(__name__)


class LocalStorageClient:
    """
    Manages local filesystem storage for PDF documents, parsed structures, and document state.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or settings.LOCAL_STORAGE_DIR
        self.pdf_dir = os.path.join(self.base_dir, "pdfs")
        self.parsed_dir = os.path.join(self.base_dir, "parsed")
        self.status_file = os.path.join(self.base_dir, "status.json")
        self._lock = threading.RLock()

        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.parsed_dir, exist_ok=True)
        self._init_status_file()

    def _init_status_file(self) -> None:
        with self._lock:
            if not os.path.exists(self.status_file):
                with open(self.status_file, "w", encoding="utf-8") as f:
                    json.dump({}, f, indent=2)

    def save_pdf(self, document_id: str, content: bytes) -> str:
        """Save raw PDF bytes to local storage."""
        file_path = os.path.join(self.pdf_dir, f"{document_id}.pdf")
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved PDF to {file_path} ({len(content)} bytes)")
        return file_path

    def get_pdf(self, document_id: str) -> Optional[bytes]:
        """Read PDF bytes from local storage."""
        file_path = os.path.join(self.pdf_dir, f"{document_id}.pdf")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return f.read()
        return None

    def save_parsed_json(self, document_id: str, data: dict) -> str:
        """Save parsed document structure JSON to local storage."""
        file_path = os.path.join(self.parsed_dir, f"{document_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved parsed JSON to {file_path}")
        return file_path

    def get_parsed_json(self, document_id: str) -> Optional[dict]:
        """Load parsed document structure JSON from local storage."""
        file_path = os.path.join(self.parsed_dir, f"{document_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def update_status(
        self,
        document_id: str,
        status: str,
        message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update processing and indexing status for a document."""
        with self._lock:
            statuses = self.get_all_statuses()
            doc_status = statuses.get(document_id, {})

            doc_status["document_id"] = document_id
            doc_status["status"] = status
            doc_status["updated_at"] = datetime.now(timezone.utc).isoformat()
            if message is not None:
                doc_status["message"] = message
            if chunk_count is not None:
                doc_status["chunk_count"] = chunk_count

            if status == "indexing":
                doc_status["indexing_started_at"] = datetime.now(timezone.utc).isoformat()
            elif status == "indexed":
                doc_status["indexed_at"] = datetime.now(timezone.utc).isoformat()
            elif status == "failed":
                doc_status["failed_at"] = datetime.now(timezone.utc).isoformat()

            for k, v in kwargs.items():
                doc_status[k] = v

            statuses[document_id] = doc_status

            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(statuses, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated status for {document_id}: {status}")
            return doc_status

    def get_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a document."""
        with self._lock:
            statuses = self.get_all_statuses()
            return statuses.get(document_id)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get all document statuses."""
        with self._lock:
            if not os.path.exists(self.status_file):
                return {}
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading status file: {e}")
                return {}

