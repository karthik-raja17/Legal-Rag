import io
import os
import logging
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from requests.exceptions import ConnectionError, Timeout

from src.config.settings import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class DriveFileNotFound(Exception):
    """Raised when a Drive file genuinely doesn't exist or isn't accessible. Not retried."""
    def __init__(self, file_id: str, status: int):
        self.file_id = file_id
        self.status = status
        super().__init__(f"Drive file {file_id} not accessible (HTTP {status})")


def _is_transient_http_error(exc: BaseException) -> bool:
    """
    Only retry errors that might resolve on their own: rate limits, server errors,
    network blips. 404 (not found) and 403 (permission denied) are permanent for a
    given file and must NOT be retried - retrying them just burns time for nothing.
    """
    if isinstance(exc, (ConnectionError, Timeout)):
        return True
    if isinstance(exc, HttpError):
        status = exc.resp.status if exc.resp is not None else None
        # 429 = rate limited, 5xx = transient server-side issue
        return status == 429 or (status is not None and 500 <= status < 600)
    return False


class DriveClient:
    def __init__(self, credentials_path: Optional[str] = None):
        self.creds = None
        if credentials_path and os.path.exists(credentials_path):
            from google.oauth2 import service_account
            self.creds = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=SCOPES
            )
        else:
            import google.auth
            self.creds, _ = google.auth.default(scopes=SCOPES)
        self.service = build("drive", "v3", credentials=self.creds)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_transient_http_error),
        reraise=True,
    )
    def download_file(self, file_id: str) -> bytes:
        """Download a file from Drive by its ID, return content as bytes."""
        try:
            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.debug(f"Download {status.progress() * 100:.2f}% complete")
            return fh.getvalue()
        except HttpError as e:
            status = e.resp.status if e.resp is not None else None
            if status in (404, 403):
                logger.warning(f"Drive file {file_id} not accessible (HTTP {status}), not retrying.")
                raise DriveFileNotFound(file_id, status) from e
            logger.error(f"Failed to download file {file_id}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_transient_http_error),
        reraise=True,
    )
    def get_file_metadata(self, file_id: str) -> dict:
        """
        Get file metadata (modifiedTime, name, etc.).
        Raises DriveFileNotFound immediately (no retries) on 404/403, since those
        are permanent for a given file_id and retrying wastes ~30-60s per row.
        """
        try:
            return self.service.files().get(
                fileId=file_id,
                fields="id,name,modifiedTime,mimeType,size",
                supportsAllDrives=True,
            ).execute()
        except HttpError as e:
            status = e.resp.status if e.resp is not None else None
            if status in (404, 403):
                logger.warning(f"Drive file {file_id} not accessible (HTTP {status}), not retrying.")
                raise DriveFileNotFound(file_id, status) from e
            logger.error(f"Failed to get metadata for {file_id}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_transient_http_error),
        reraise=True,
    )
    def list_pdfs_in_folder(self, folder_id: str) -> list:
        """List all PDF files in a given folder."""
        try:
            query = f"'{folder_id}' in parents and mimeType='application/pdf'"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, modifiedTime)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            return results.get("files", [])
        except HttpError as e:
            logger.error(f"Failed to list PDFs in folder {folder_id}: {e}")
            raise