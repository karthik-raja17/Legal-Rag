import logging
from datetime import datetime
from typing import Optional, Dict, Any

from google.api_core.exceptions import GoogleAPICallError, NotFound, PermissionDenied
from google.cloud import firestore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.settings import settings

logger = logging.getLogger(__name__)


class FirestoreClient:
    def __init__(self):
        self.db = firestore.Client(project=settings.GCP_PROJECT_ID)
        self.collection = settings.FIRESTORE_COLLECTION

    # ==================== EXCEL STATE ====================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def get_excel_state(self) -> Dict[str, Any]:
        """Get the stored state for the Excel file (modified_time, etc.)"""
        doc_ref = self.db.collection(self.collection).document("excel_tracker")
        try:
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return {}
        except PermissionDenied:
            logger.error("Permission denied accessing Firestore")
            raise
        except GoogleAPICallError as e:
            logger.error(f"Firestore error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def update_excel_state(self, modified_time: str, version: str, status: str = "synced") -> None:
        """Update the Excel tracker state."""
        doc_ref = self.db.collection(self.collection).document("excel_tracker")
        try:
            doc_ref.set({
                "modified_time": modified_time,
                "version": version,
                "status": status,
                "last_sync": datetime.utcnow().isoformat()
            }, merge=True)
            logger.info(f"Updated Excel state: modified_time={modified_time}, version={version}")
        except PermissionDenied:
            logger.error("Permission denied updating Excel state")
            raise
        except GoogleAPICallError as e:
            logger.error(f"Firestore error: {e}")
            raise

    # ==================== DOCUMENT STATE ====================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def get_document_state(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get the state of a specific contract document."""
        doc_ref = self.db.collection(self.collection).document(doc_id)
        try:
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        except NotFound:
            return None
        except PermissionDenied:
            logger.error(f"Permission denied accessing document {doc_id}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def is_document_processed(self, doc_id: str, drive_modified_time: str) -> bool:
        """
        Quickly check if a document is already processed and up-to-date.
        Returns True if the document exists and the stored modified_time matches.
        """
        state = self.get_document_state(doc_id)
        if not state:
            return False
        return state.get("drive_modified_time") == drive_modified_time

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def update_document_state(self, doc_id: str, data: Dict[str, Any]) -> None:
        """Update the state of a contract document."""
        doc_ref = self.db.collection(self.collection).document(doc_id)
        try:
            # Add timestamp automatically
            data["last_updated"] = datetime.utcnow().isoformat()
            doc_ref.set(data, merge=True)
            logger.debug(f"Updated document state for {doc_id}")
        except PermissionDenied:
            logger.error(f"Permission denied updating document {doc_id}")
            raise
        except GoogleAPICallError as e:
            logger.error(f"Firestore error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def delete_document_state(self, doc_id: str) -> bool:
        """Remove state for a deleted contract. Returns True if deletion succeeded."""
        doc_ref = self.db.collection(self.collection).document(doc_id)
        try:
            doc_ref.delete()
            logger.info(f"Deleted document state for {doc_id}")
            return True
        except NotFound:
            logger.warning(f"Document state {doc_id} not found – nothing to delete")
            return False
        except PermissionDenied:
            logger.error(f"Permission denied deleting document {doc_id}")
            raise
        except GoogleAPICallError as e:
            logger.error(f"Firestore error: {e}")
            raise