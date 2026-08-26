import io
import logging
from typing import Optional

from google.api_core.exceptions import GoogleAPICallError, NotFound, Forbidden
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.settings import settings

logger = logging.getLogger(__name__)


class GCSClient:
    def __init__(self):
        self.client = storage.Client(project=settings.GCP_PROJECT_ID)
        self.bucket_name = settings.GCS_BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)

    def _get_blob_path(self, file_id: str) -> str:
        return f"pdfs/{file_id}.pdf"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def upload_pdf(self, file_id: str, content: bytes, drive_modified_time: Optional[str] = None) -> str:
        """Upload PDF content to GCS with custom metadata."""
        blob_name = self._get_blob_path(file_id)
        blob = self.bucket.blob(blob_name)

        # Store modified_time for cache validation
        metadata = {
            "source": "google_drive",
            "uploaded_by": "legal-rag-engine",
            "file_id": file_id,
        }
        if drive_modified_time:
            metadata["drive_modified_time"] = drive_modified_time
        blob.metadata = metadata

        blob.upload_from_string(content, content_type="application/pdf")
        logger.info(f"Uploaded PDF {file_id} to gs://{self.bucket_name}/{blob_name}")
        return blob_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def download_pdf(self, file_id: str) -> Optional[bytes]:
        """Download PDF content from GCS. Returns None if not found."""
        blob_name = self._get_blob_path(file_id)
        blob = self.bucket.blob(blob_name)

        try:
            return blob.download_as_bytes()
        except NotFound:
            logger.warning(f"PDF {file_id} not found in GCS")
            return None
        except Forbidden:
            logger.error(f"Permission denied accessing PDF {file_id} in GCS")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def get_metadata(self, file_id: str) -> Optional[dict]:
        """Fetch custom metadata of a cached PDF (e.g., drive_modified_time)."""
        blob_name = self._get_blob_path(file_id)
        blob = self.bucket.blob(blob_name)

        try:
            blob.reload()  # Fetch metadata from GCS
            return blob.metadata
        except NotFound:
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(GoogleAPICallError),
        reraise=True
    )
    def delete_pdf(self, file_id: str) -> bool:
        """Delete PDF from GCS. Returns True if deletion succeeded."""
        blob_name = self._get_blob_path(file_id)
        blob = self.bucket.blob(blob_name)

        try:
            blob.delete()
            logger.info(f"Deleted PDF {file_id} from GCS")
            return True
        except NotFound:
            logger.warning(f"PDF {file_id} not found in GCS – nothing to delete")
            return False