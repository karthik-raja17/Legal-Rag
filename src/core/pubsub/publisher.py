"""
Google Cloud Pub/Sub publisher for asynchronous document processing.
Includes retry logic, comprehensive error handling, and structured logging.
"""
import json
import logging
from typing import Dict, Optional

from google.cloud import pubsub_v1
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


class Publisher:
    """
    A production‑grade Pub/Sub publisher for sending document processing messages.

    Features:
        - Automatic retries with exponential backoff for transient errors.
        - Detailed logging for all operations.
        - Input validation and clear error messages.
        - Support for custom message attributes (useful for filtering/routing).
        - Configurable publish timeout.
        - Graceful handling of API errors.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize the Pub/Sub publisher.

        Args:
            timeout_seconds: Timeout for the publish call (blocking future result).

        Raises:
            ValueError: If required settings (project ID, topic ID) are missing.
        """
        self.project_id = settings.GCP_PROJECT_ID
        self.topic_id = settings.PUBSUB_TOPIC_ID
        self.timeout_seconds = timeout_seconds

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID is not set in settings.")
        if not self.topic_id:
            raise ValueError("PUBSUB_TOPIC_ID is not set in settings.")

        # Initialize the publisher client with default settings.
        # For higher throughput, you can customize batch settings here.
        self.publisher = pubsub_v1.PublisherClient(
            # Optional: set custom batch settings
            # batch_settings=pubsub_v1.types.BatchSettings(
            #     max_messages=100,
            #     max_bytes=1024 * 1024,  # 1 MB
            #     max_latency=0.05,       # 50 ms
            # )
        )
        self.topic_path = self.publisher.topic_path(self.project_id, self.topic_id)

        # Optionally, verify the topic exists (fails fast if misconfigured).
        try:
            self.publisher.get_topic(request={"topic": self.topic_path})
            logger.info(f"Publisher initialized for topic: {self.topic_path}")
        except exceptions.NotFound:
            logger.error(f"Topic '{self.topic_path}' does not exist. Please create it.")
            raise
        except Exception as e:
            logger.error(f"Failed to verify topic '{self.topic_path}': {e}")
            # Don't raise here – the client might just be lacking permissions,
            # but the publish itself will fail later anyway.
            # Optionally re‑raise if you want to fail fast.
            # raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (
                exceptions.ServiceUnavailable,
                exceptions.InternalServerError,
                exceptions.DeadlineExceeded,
                exceptions.RetryError,
                exceptions.TooManyRequests,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def publish(
        self,
        document_id: str,
        gcs_uri: str,
        attributes: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Publish a message to the Pub/Sub topic.

        Args:
            document_id: Unique identifier for the document.
            gcs_uri: URI of the parsed JSON in Cloud Storage.
            attributes: Optional key‑value pairs for message routing/filtering
                        (e.g., {"priority": "high", "source": "web"}).

        Returns:
            The message ID assigned by Pub/Sub.

        Raises:
            ValueError: If required arguments are invalid.
            RuntimeError: If publish fails after retries.
        """
        # Input validation
        if not document_id:
            raise ValueError("document_id cannot be empty or None.")
        if not gcs_uri:
            raise ValueError("gcs_uri cannot be empty or None.")
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"gcs_uri must start with 'gs://', got: {gcs_uri}")

        # Construct the message payload
        payload = {
            "document_id": document_id,
            "gcs_uri": gcs_uri,
            "timestamp": None,  # Pub/Sub automatically adds a timestamp
        }
        # Optionally add a timestamp for manual tracking
        import datetime
        timestamp = datetime.datetime.now(datetime.UTC).isoformat() + "Z"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            # Publish the message (non‑blocking, returns a Future)
            future = self.publisher.publish(
                self.topic_path,
                data,
                **attributes or {},  # Unpack attributes as keyword arguments
            )

            # Wait for the publish to complete (with timeout).
            # This is blocking; for non‑blocking, return the future instead.
            message_id = future.result(timeout=self.timeout_seconds)

            logger.info(
                f"Published message to {self.topic_path} | "
                f"document_id={document_id} | message_id={message_id} | "
                f"attributes={attributes}"
            )
            return message_id

        except exceptions.GoogleAPIError as e:
            logger.error(f"Pub/Sub API error for document {document_id}: {e}")
            raise RuntimeError(f"Failed to publish message for {document_id}") from e
        except TimeoutError as e:
            logger.error(f"Publish timeout for document {document_id}: {e}")
            raise RuntimeError(f"Publish timeout for {document_id}") from e
        except Exception as e:
            logger.error(f"Unexpected error publishing document {document_id}: {e}")
            raise

    def publish_async(self, document_id: str, gcs_uri: str, attributes: Optional[Dict[str, str]] = None):
        """
        Asynchronous version: returns the Future without waiting.

        Useful if you want to fire‑and‑forget and handle the result elsewhere.
        """
        if not document_id or not gcs_uri:
            raise ValueError("document_id and gcs_uri are required.")

        payload = {"document_id": document_id, "gcs_uri": gcs_uri}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        return self.publisher.publish(self.topic_path, data, **attributes or {})