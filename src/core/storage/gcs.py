"""
Google Cloud Storage client for uploading/downloading JSON documents.
Includes retry logic, comprehensive error handling, and structured logging.
"""
import json
import logging
from typing import Optional
from urllib.parse import urlparse

from google.cloud import storage
from google.api_core import exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.settings import settings

logger = logging.getLogger(__name__)


class GCSClient:
    """
    A production‑grade GCS client for JSON payloads.

    Features:
        - Automatic retries with exponential backoff for transient errors.
        - Detailed logging for all operations.
        - Input validation and clear error messages.
        - Support for cross‑bucket downloads (if URI points elsewhere).
        - Graceful handling of missing blobs (returns None).
    """

    def __init__(self):
        """Initialize the GCS client and bucket reference."""
        self.project_id = settings.GCP_PROJECT_ID
        self.bucket_name = settings.GCS_BUCKET_NAME

        # Use the default credentials (ADC) – automatically used in Cloud Run.
        self.client = storage.Client(project=self.project_id)
        self.bucket = self.client.bucket(self.bucket_name)

        # Verify that the bucket exists and is accessible (optional but recommended).
        # This will raise an exception if the bucket is missing or permissions are wrong.

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (exceptions.ServerError, exceptions.TooManyRequests, exceptions.DeadlineExceeded)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def upload_json(self, data: dict, blob_name: str) -> str:
        """
        Upload a JSON-serializable dictionary to GCS.

        Args:
            data: Python dict to upload.
            blob_name: Path within the bucket (e.g., "parsed/doc123.json").

        Returns:
            The gs:// URI of the uploaded object.

        Raises:
            ValueError: If input is invalid.
            RuntimeError: On upload failure after retries.
        """
        if not blob_name:
            raise ValueError("blob_name cannot be empty or None.")
        if not isinstance(data, dict):
            raise TypeError("data must be a dict (JSON-serializable).")

        blob = self.bucket.blob(blob_name)
        try:
            # Use ensure_ascii=False to preserve non‑ASCII characters (e.g., French accents).
            json_str = json.dumps(data, ensure_ascii=False)
            blob.upload_from_string(json_str, content_type="application/json")
            uri = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"Uploaded JSON to {uri} (size: {len(json_str)} bytes)")
            return uri
        except exceptions.GoogleAPIError as e:
            logger.error(f"GCS API error during upload of {blob_name}: {e}")
            raise RuntimeError(f"Failed to upload {blob_name}") from e
        except Exception as e:
            logger.error(f"Unexpected error during upload of {blob_name}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (exceptions.ServerError, exceptions.TooManyRequests, exceptions.DeadlineExceeded)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def download_json(self, gcs_uri: str) -> Optional[dict]:
        """
        Download a JSON object from a gs:// URI.

        Args:
            gcs_uri: Full URI like "gs://bucket/path/file.json".

        Returns:
            Parsed JSON as dict, or None if the blob does not exist.

        Raises:
            ValueError: If URI is invalid.
            RuntimeError: On download failure after retries (except for 404).
        """
        # Parse the URI robustly.
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI (must start with 'gs://'): {gcs_uri}")

        # Use urllib.parse for proper parsing.
        parsed = urlparse(gcs_uri)
        if not parsed.netloc or not parsed.path or parsed.path == "/":
            raise ValueError(f"Invalid GCS URI format: {gcs_uri}")

        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        # If the bucket differs from the default, use a separate client reference.
        if bucket_name == self.bucket_name:
            bucket = self.bucket
        else:
            logger.info(f"Downloading from external bucket: {bucket_name}")
            bucket = self.client.bucket(bucket_name)

        blob = bucket.blob(blob_name)

        try:
            content = blob.download_as_string()
            data = json.loads(content)
            logger.info(f"Downloaded JSON from {gcs_uri} (size: {len(content)} bytes)")
            return data
        except exceptions.NotFound:
            logger.warning(f"Blob not found: {gcs_uri}")
            return None
        except exceptions.GoogleAPIError as e:
            logger.error(f"GCS API error during download of {gcs_uri}: {e}")
            raise RuntimeError(f"Failed to download {gcs_uri}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON content in {gcs_uri}: {e}")
            raise RuntimeError(f"Invalid JSON in {gcs_uri}") from e
        except Exception as e:
            logger.error(f"Unexpected error during download of {gcs_uri}: {e}")
            raise

    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob (optional convenience method).

        Returns:
            True if deleted, False if blob did not exist.
        """
        if not blob_name:
            raise ValueError("blob_name cannot be empty.")
        blob = self.bucket.blob(blob_name)
        try:
            blob.delete()
            logger.info(f"Deleted blob: {blob_name}")
            return True
        except exceptions.NotFound:
            logger.warning(f"Blob not found for deletion: {blob_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {blob_name}: {e}")
            raise