"""
Indexer service for Legal RAG.
Listens to Pub/Sub push messages, indexes parsed documents into ChromaDB.
Includes lazy initialization, robust error handling, retries, and Firestore status updates.
"""
import base64
import json
import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response, HTTPException
from google.api_core import exceptions as gcp_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from google.cloud import firestore
from datetime import datetime, timezone

from src.core.indexer.indexer import Indexer
from src.core.storage.gcs import GCSClient
from src.core.parser.pdf_parser import ParsedDocument
from src.settings import settings

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Legal RAG Indexer",
    description="Indexes parsed legal documents into ChromaDB.",
    version="1.0.0"
)

# ----------------------------------------------------------------------------
# Lazy Initialization
# ----------------------------------------------------------------------------
_gcs_client: Optional[GCSClient] = None
_indexer: Optional[Indexer] = None
_firestore_client = None  # Placeholder for Firestore


def get_gcs_client() -> GCSClient:
    """Lazy‑initialize the GCS client."""
    global _gcs_client
    if _gcs_client is None:
        try:
            _gcs_client = GCSClient()
            logger.info("GCS client initialized.")
        except Exception as e:
            logger.error(f"GCS client initialization failed: {e}", exc_info=True)
            raise
    return _gcs_client


def get_indexer() -> Indexer:
    """Lazy‑initialize the Indexer (connects to ChromaDB and embedding model)."""
    global _indexer
    if _indexer is None:
        try:
            # Indexer should connect to ChromaDB and load the embedding model.
            # It may raise exceptions if ChromaDB is unreachable.
            _indexer = Indexer()
            logger.info("Indexer initialized.")
        except Exception as e:
            logger.error(f"Indexer initialization failed: {e}", exc_info=True)
            raise
    return _indexer


def get_firestore_client():
    """Lazy‑initialize Firestore client (placeholder)."""
    global _firestore_client
    if _firestore_client is None:
        try:
            from google.cloud import firestore
            _firestore_client = firestore.Client(project=settings.GCP_PROJECT_ID)
            logger.info("Firestore client initialized.")
        except Exception as e:
            logger.warning(f"Firestore client initialization failed (status updates disabled): {e}")
            _firestore_client = None
    return _firestore_client


# ----------------------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "service": "indexer", "project": settings.GCP_PROJECT_ID}


# ----------------------------------------------------------------------------
# Firestore Status Update (Placeholder – will be implemented later)
# ----------------------------------------------------------------------------
def update_status(
    document_id: str,
    status: str,
    message: Optional[str] = None,
    chunk_count: Optional[int] = None,
):
    db = get_firestore_client()
    if not db:
        return
    doc_ref = db.collection(settings.FIRESTORE_COLLECTION).document(document_id)
    data = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if message:
        data["message"] = message
    if chunk_count is not None:
        data["chunk_count"] = chunk_count
    if status == "failed":
        data["failed_at"] = datetime.now(timezone.utc).isoformat()
    elif status == "indexed":
        data["indexed_at"] = datetime.now(timezone.utc).isoformat()

    doc_ref.set(data, merge=True)
    logger.info(f"Firestore status updated for {document_id}: {status}")

# ----------------------------------------------------------------------------
# Pub/Sub Push Endpoint with Retries
# ----------------------------------------------------------------------------
@app.post("/index")
async def handle_pubsub(request: Request):
    """
    Pub/Sub push endpoint.
    - Reads the message, decodes base64 payload.
    - Downloads parsed JSON from GCS.
    - Indexes the document into ChromaDB.
    - Updates Firestore status.
    - Returns 204 on success, 500 on retryable errors, 400 on invalid messages.
    """
    # 1. Parse and validate the Pub/Sub envelope
    try:
        envelope = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse Pub/Sub request body: {e}")
        return Response(status_code=400, content="Invalid JSON")

    if not envelope or "message" not in envelope:
        logger.warning("Received invalid Pub/Sub message (missing 'message' field)")
        return Response(status_code=400, content="Missing 'message' field")

    # Decode the message data (base64 encoded)
    try:
        raw_data = envelope["message"].get("data")
        if not raw_data:
            logger.warning("Pub/Sub message has empty data field")
            return Response(status_code=400, content="Empty message data")
        decoded = base64.b64decode(raw_data).decode("utf-8")
        message = json.loads(decoded)
    except (base64.binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.error(f"Failed to decode or parse Pub/Sub message: {e}")
        return Response(status_code=400, content="Invalid message encoding")

    document_id = message.get("document_id")
    gcs_uri = message.get("gcs_uri")
    if not document_id or not gcs_uri:
        logger.error(f"Missing document_id or gcs_uri in message: {message}")
        return Response(status_code=400, content="Missing document_id or gcs_uri")

    logger.info(f"Received indexing request for document: {document_id} from {gcs_uri}")

    # Update status: indexing started
    update_status(document_id, "indexing", "Started indexing")

    try:
        # 2. Download parsed JSON from GCS (with retries)
        gcs_client = get_gcs_client()
        parsed_dict = gcs_client.download_json(gcs_uri)
        if parsed_dict is None:
            logger.error(f"Parsed JSON not found in GCS: {gcs_uri}")
            update_status(document_id, "failed", f"Missing GCS blob: {gcs_uri}")
            # Return 404 to Pub/Sub? Better to return 400 (bad request) or 404? 
            # Pub/Sub will retry 5xx errors. 4xx are considered fatal and not retried.
            return Response(status_code=400, content=f"Blob not found: {gcs_uri}")

        # 3. Convert to ParsedDocument
        try:
            parsed_doc = ParsedDocument(**parsed_dict)
        except Exception as e:
            logger.error(f"Failed to deserialize ParsedDocument for {document_id}: {e}")
            update_status(document_id, "failed", f"Invalid JSON structure: {str(e)}")
            return Response(status_code=400, content="Invalid document schema")

        # 4. Index into ChromaDB (with retries)
        indexer = get_indexer()
        # The indexer may raise exceptions (network, DB full, etc.)
        chunk_count = indexer.index_document(parsed_doc, document_id)

        # After indexing
        db = firestore.Client(project=settings.GCP_PROJECT_ID)
        doc_ref = db.collection(settings.FIRESTORE_COLLECTION).document(document_id)
        doc_ref.set({
            "status": "indexed",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,  # you need to capture this from indexer.index_document
        }, merge=True)
        logger.info(f"Status updated in Firestore for {document_id}: indexed")

        # 5. Update status: success
        update_status(document_id, "indexed", "Successfully indexed", chunk_count=chunk_count)
        logger.info(f"Successfully indexed {chunk_count} chunks for {document_id}")

        # Return 204 No Content (success, no retry)
        return Response(status_code=204)

    except Exception as e:
        # We treat most exceptions as retryable (5xx) so Pub/Sub will redeliver.
        # However, if the error is clearly client-side (e.g., validation), we return 4xx.
        # For now, we log and return 500 to trigger a retry.
        logger.error(f"Indexing failed for {document_id}: {e}", exc_info=True)
        update_status(document_id, "failed", f"Indexing error: {str(e)}")
        # Return 500 to trigger Pub/Sub retry (exponential backoff).
        return Response(status_code=500, content=f"Indexing failed: {str(e)}")


# ----------------------------------------------------------------------------
# Optional: Shutdown handler for cleanup
# ----------------------------------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown – close any persistent connections."""
    global _indexer
    if _indexer is not None:
        # If Indexer has a close method, call it.
        if hasattr(_indexer, "close"):
            try:
                _indexer.close()
                logger.info("Indexer closed gracefully.")
            except Exception as e:
                logger.warning(f"Error closing Indexer: {e}")
    logger.info("Indexer service shutting down.")