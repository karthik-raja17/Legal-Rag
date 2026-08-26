import os
from typing import List
import google.auth.transport.requests
import google.oauth2.id_token
import requests

BGE_SERVICE_URL = os.getenv(
    "BGE_EMBEDDER_URL",
    "https://legal-bge-embedder-xjcapjczfa-od.a.run.app",
)

class BGEEmbedderClient:

    def __init__(self, endpoint_url: str = BGE_SERVICE_URL):
        self.base_url = endpoint_url.rstrip("/")
        self.endpoint = f"{self.base_url}/embed"

    def _get_headers(self) -> dict:
        # Only skip auth for local testing
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return {"Content-Type": "application/json"}
        # Always fetch token for production
        auth_req = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(auth_req, self.base_url)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def embed_documents(
        self, texts: List[str], batch_size: int = 16
    ) -> List[List[float]]:
        all_embeddings = []
        headers = self._get_headers()
        safe_batch = min(batch_size, 16)

        for i in range(0, len(texts), safe_batch):
            batch = texts[i : i + safe_batch]
            prefixed = [f"Represent this document for retrieval: {t}" for t in batch]
            response = requests.post(
                self.endpoint,
                json={"inputs": prefixed, "normalize": True, "truncate": True},
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            all_embeddings.extend(response.json())
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        headers = self._get_headers()
        prefixed = [
            f"Represent this question for searching relevant passages: {query}"
        ]
        response = requests.post(
            self.endpoint,
            json={"inputs": prefixed, "normalize": True, "truncate": True},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()[0]