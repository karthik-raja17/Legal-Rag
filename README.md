# Legal Contract RAG Engine — CUAD Benchmark & Local Contract Intelligence

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-blueviolet)](https://github.com/facebookresearch/faiss)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-Qwen3--Embedding--0.6B%20(512d%20MRL)-yellow)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(qwen2.5%3A7b)-black)](https://ollama.com/)

A 100% **local, privacy-first Retrieval-Augmented Generation (RAG) system** for **English legal commercial contracts** evaluated on the **CUAD (Contract Understanding Atticus Dataset)** benchmark (Affiliate, Co-Branding, Development, Licensing, Service agreements, etc.).

- **Vector Store**: FAISS HNSW (`M=24`, `efConstruction=100`, `efSearch=100`)
- **Embeddings**: `Qwen/Qwen3-Embedding-0.6B` using **Matryoshka Representation Learning (MRL)** truncated to **512 dimensions** with L2 normalization
- **Local LLM**: Ollama (`qwen2.5:7b` or local model of choice)
- **Hybrid Retrieval**: Dense FAISS + `rank-bm25` Okapi with Reciprocal Rank Fusion (RRF)
- **English Contract NLP**: English spaCy model (`en_core_web_sm`), standard contract structure detection (`ARTICLE`, `SECTION`, `EXHIBIT`, `SCHEDULE`), and 40+ CUAD legal clause classifications
- **Local Storage & Cache**: Local filesystem storage with JSON metadata and BM25 index caching
- **Grounded Legal Generation**: Strict contract Q&A with exact clause citations (`[1]`, `[2]`, etc.) and numeric fidelity.

---

## 🏛 Architecture

```mermaid
flowchart TD
  PDF[English Contract PDF / CUAD Dataset] --> Parser[PDFParser: PyMuPDF + Regex Structure]
  Parser --> Chunker[DocumentChunker: Hierarchical Clauses & Breadcrumbs]
  Chunker --> Embedder[LocalEmbedder: Qwen3-Embedding-0.6B 512d MRL]
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
python -m spacy download en_core_web_sm
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
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIMENSION=512

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

LOCAL_STORAGE_DIR=./data/storage
BM25_CACHE_DIR=./data/bm25_cache
```

---

## 📂 Ingesting Documents & CUAD Dataset

### Ingest sample English contract:
```bash
python scripts/create_test_pdf.py
python scripts/local_ingest.py --pdf data/sample_contract.pdf --doc-id msa_acme_01 --site "Delaware Headquarters"
```

### Ingest CUAD agreements:
```bash
python scripts/local_ingest.py --dir data/cuad/pdfs/Part_I/Affiliate_Agreements --site "Affiliate Agreements"
```

---

## 🧪 Testing and Verification

Run end-to-end local RAG tests:
```bash
python scripts/test_local_rag.py
```

Run FastAPI in-process tests:
```bash
python scripts/test_fastapi_endpoints.py
```

---

## 🌐 Running the Web Server

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8080 --reload
```

Then open `http://localhost:8080` in your browser.
