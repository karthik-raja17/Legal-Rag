"""
Test FastAPI endpoints in-process with TestClient / AsyncClient.
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
        assert "Lentilly" in resp.json().get("sites", [])

        print("\n4. Testing POST /query for rent...")
        query_payload = {
            "query": "Quel est le montant de la redevance ou du loyer annuel ?",
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
        assert "12 500" in data.get("answer", "") or "12500" in data.get("answer", "")

        print("\n5. Testing POST /query for duration...")
        query_payload_dur = {
            "query": "Quelle est la durée du bail ?",
            "generate": True,
            "hybrid": True,
            "rewrite": False,
            "top_k": 3,
        }
        resp_dur = await ac.post("/query", json=query_payload_dur)
        data_dur = resp_dur.json()
        print(f"Answer duration: {data_dur.get('answer')}")
        assert "30" in data_dur.get("answer", "") or "trente" in data_dur.get("answer", "")

        print("\n6. Testing POST /query for penalties...")
        query_payload_pen = {
            "query": "Quel est le montant de la pénalité par jour de retard ?",
            "generate": True,
            "hybrid": True,
            "rewrite": False,
            "top_k": 3,
        }
        resp_pen = await ac.post("/query", json=query_payload_pen)
        data_pen = resp_pen.json()
        print(f"Answer penalty: {data_pen.get('answer')}")
        assert "50" in data_pen.get("answer", "")

        print("\n🎉 All FastAPI endpoints verified successfully!")

if __name__ == "__main__":
    asyncio.run(test_api())
