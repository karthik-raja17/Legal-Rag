"""
Test FastAPI endpoints in-process with TestClient / AsyncClient for English contract RAG.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from httpx import AsyncClient, ASGITransport
from src.app.main import app


async def test_api():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        print("1. Testing GET /health...")
        resp = await ac.get("/health")
        print("Health response:", resp.status_code, resp.json())
        assert resp.status_code == 200

        print("\n2. Testing GET /api/documents/count...")
        resp = await ac.get("/api/documents/count")
        print("Count response:", resp.status_code, resp.json())
        assert resp.status_code == 200
        assert resp.json().get("count") >= 1

        print("\n3. Testing GET /api/dropdown-options...")
        resp = await ac.get("/api/dropdown-options")
        print("Dropdown response:", resp.status_code, resp.json())
        assert resp.status_code == 200

        print("\n4. Testing POST /query for annual fee...")
        query_payload = {
            "query": "What is the annual service fee and monthly payment under this agreement?",
            "generate": True,
            "hybrid": True,
            "rewrite": False,
            "top_k": 3,
        }
        resp = await ac.post("/query", json=query_payload)
        print("Query status:", resp.status_code)
        data = resp.json()
        print(f"Answer: {data.get('answer')}")
        print(f"Citations count: {len(data.get('citations', {}))}")
        assert resp.status_code == 200
        assert data.get("answer") is not None
        assert len(data.get("retrieved_chunks", [])) > 0

        print("\n5. Testing POST /query for contract duration...")
        query_payload_dur = {
            "query": "What is the initial term and duration of the contract?",
            "generate": True,
            "hybrid": True,
            "rewrite": False,
            "top_k": 3,
        }
        resp_dur = await ac.post("/query", json=query_payload_dur)
        data_dur = resp_dur.json()
        print(f"Answer duration: {data_dur.get('answer')}")
        assert resp_dur.status_code == 200
        assert data_dur.get("answer") is not None

        print("\n6. Testing POST /query for governing law and jurisdiction...")
        query_payload_law = {
            "query": "What state law governs this agreement and where are disputes resolved?",
            "generate": True,
            "hybrid": True,
            "rewrite": False,
            "top_k": 3,
        }
        resp_law = await ac.post("/query", json=query_payload_law)
        data_law = resp_law.json()
        print(f"Answer governing law: {data_law.get('answer')}")
        assert resp_law.status_code == 200
        assert data_law.get("answer") is not None

        print("\n🎉 All FastAPI endpoints verified successfully!")


if __name__ == "__main__":
    asyncio.run(test_api())
