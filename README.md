# Legal RAG Engine — Local French Contract Intelligence

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-blueviolet)](https://github.com/facebookresearch/faiss)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(qwen2.5%3A7b)-black)](https://ollama.com/)

A 100% **local, privacy-first Retrieval-Augmented Generation (RAG) system** for **French legal contracts** (commercial leases, NDAs, photovoltaic agreements, construction contracts).

- **Vector Store**: FAISS HNSW (`M=24`, `efConstruction=100`, `efSearch=100`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, L2 normalized cosine similarity)
- **Local LLM**: Ollama (`qwen2.5:7b` or local model of choice)
- **Hybrid Retrieval**: Dense FAISS + `rank-bm25` Okapi with Reciprocal Rank Fusion (RRF)
- **Local Storage & Cache**: Local filesystem storage with JSON metadata and BM25 index caching
- **Grounded Legal Generation**: Strict French contract Q&A with exact clause citations (`[1]`, `[2]`, etc.) and numeric fidelity.

---

## 🏛 Architecture

```mermaid
flowchart TD
  PDF[Local French Contract PDF] --> Parser[PDFParser: PyMuPDF + Regex Structure]
  Parser --> Chunker[DocumentChunker: Hierarchical Clauses & Breadcrumbs]
  Chunker --> Embedder[LocalEmbedder: all-MiniLM-L6-v2 384d]
  Embedder --> FAISS[(FAISS HNSW Index M=24 efC=100 efS=100)]
  Chunker --> BM25[(Local BM25 Okapi Cache)]
  
  Query[User Question] --> API[FastAPI /query or CLI]
  API --> Rewriter[Query Rewriter via Ollama]
  Rewriter --> Retriever[HybridRetriever: FAISS Dense + BM25 RRF]
  FAISS -.-> Retriever
  BM25 -.-> Retriever
  Retriever --> Reranker[Local Reranker / Cross-Encoder]
  Reranker --> Ollama[Ollama Local LLM: qwen2.5:7b]
  Ollama --> Answer[Grounded Legal Answer with Citations [1], [2]]
```

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com/) installed and running locally

```bash
# Pull and start your local Ollama model
ollama pull qwen2.5:7b
ollama serve
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Key configuration in `.env`:
```env
VECTOR_STORE_TYPE=faiss
FAISS_INDEX_DIR=./data/faiss_index
HNSW_M=24
HNSW_EF_CONSTRUCTION=100
HNSW_EF_SEARCH=100

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

LOCAL_STORAGE_DIR=./data/storage
BM25_CACHE_DIR=./data/bm25_cache
```

---

## 📂 Ingesting Documents Locally

### Ingest a single PDF:
```bash
python scripts/local_ingest.py --pdf data/test_contract.pdf --doc-id bail_01 --site Lentilly
```

### Ingest an entire directory of PDFs:
```bash
python scripts/local_ingest.py --dir data/contracts/ --site Site_Provence
```

---

## 🖥 Running the FastAPI Server

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8080 --reload
```

### API Endpoints:
- `GET /health` : Liveness and status check for FAISS, Ollama, and Local Storage.
- `GET /dropdown-options` : List indexed documents and sites for UI selection.
- `POST /parse` : Upload and index a PDF directly into FAISS.
- `POST /query` : Query the RAG engine with hybrid search, context expansion, and Ollama answer generation.

### Example Query with `curl`:
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quelle est la durée du bail et le montant du loyer ?",
    "top_k": 3,
    "generate": true,
    "hybrid": true
  }'
```

---

## 🧪 Testing & Verification

Run the automated local verification suite:
```bash
# 1. Test FAISS HNSW + Embeddings + Ollama End-to-End:
python scripts/test_local_rag.py

# 2. Test FastAPI Endpoints:
python scripts/test_fastapi_endpoints.py
```
