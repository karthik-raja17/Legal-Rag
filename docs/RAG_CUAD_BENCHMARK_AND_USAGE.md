# Legal Contract RAG Engine: CUAD Benchmark, Pipeline Architecture & Complete Usage Guide

This document provides a comprehensive technical overview of the local legal contract RAG engine, explaining the architectural decisions, mathematical formulation of information retrieval (IR) metrics, step-by-step usage for individual pipeline scripts, and how to execute the all-in-one evaluation runner.

---

## 1. System Architecture & Design Rationale

```mermaid
flowchart TD
  PDF[English Contract PDFs / CUAD Dataset] --> Ingest[scripts/pipeline/ingest.py]
  Ingest --> LightParser[Lightweight Stateful Regex Parser: PyMuPDF + Breadcrumbs]
  LightParser --> DocStore[(Local SQLite DocStore: Parent Sections)]
  LightParser --> LeafChunks[(Leaf Chunks: Breadcrumbed Clauses)]
  
  LeafChunks --> Indexer[scripts/pipeline/index_documents.py]
  Indexer --> Embedder[LocalEmbedder: Qwen3-Embedding-0.6B with MRL 512d]
  Embedder --> FAISS[(FAISS HNSW Index: M=24, efC=100, efS=100)]
  LeafChunks --> BM25[(Local BM25 Okapi Disk Cache)]
  
  Query[Legal Question / CUAD QA Prompt] --> Eval[scripts/eval/evaluate_retrieval_metrics.py or FastAPI /query]
  Eval --> Dense[Dense FAISS Search]
  Eval --> Keyword[BM25 Okapi Search]
  Dense --> RRF[Reciprocal Rank Fusion - Top 60 Candidates]
  Keyword --> RRF
  RRF --> Reranker[Local Cross-Encoder: ms-marco-MiniLM-L-6-v2]
  Reranker --> TopK[Final Top 10 Ranked Chunks]
  TopK --> ContextBuilder[ContextBuilder: Small-to-Big Expansion via DocStore]
  DocStore -.-> ContextBuilder
  ContextBuilder --> Ollama[Local LLM: Ollama qwen2.5:7b]
  Ollama --> Answer[Grounded Legal Answer with Bracketed Citations [1], [2]]
```

### Why These Components?

1. **Lightweight Stateful Regex Parser (<0.03s per document)**:
   - Replaces heavy, multi-minute subprocess pipelines with pure PyMuPDF text extraction + stateful regex hierarchy detection (`HEADING_REGEX`, `SUBCLAUSE_REGEX`, `PREAMBLE_REGEX`).
   - Guarantees clause hierarchy survival (`ARTICLE 2 > SECTION 2.1 > (b)`) while reducing ingestion time by over 3,000x.
   - Preserves table detection via fast conditional `pdfplumber` (~0.05s).

2. **Small-to-Big Retrieval & SQLite DocStore**:
   - Embeds fine-grained leaf chunks for sharp semantic search while storing full parent section bodies in `./data/docstore.sqlite`.
   - The LLM receives complete, unfragmented context without bloating the FAISS vector database metadata.

1. **`Qwen/Qwen3-Embedding-0.6B` with Matryoshka Representation Learning (MRL)**:
   - **512 Dimensions**: Qwen3 natively supports MRL truncation (`truncate_dim=512`). Truncating from 1024 to 512 dimensions cuts vector memory in half, speeds up FAISS index build and search by 2x, and preserves over 99% of dense semantic similarity.
   - **Unit L2 Normalization**: Vectors are unit-normalized upon encoding, enabling inner-product (`faiss.IndexHNSWFlat(512, faiss.METRIC_INNER_PRODUCT)`) to compute exact cosine similarity with maximum hardware throughput.

2. **FAISS HNSW Vector Store ($M=24, efConstruction=100, efSearch=100$)**:
   - `M=24`: Graph connectivity per node balanced for high recall and fast insertion.
   - `efConstruction=100` & `efSearch=100`: High exploration depth ensuring near-exact nearest-neighbor recall even with large collections of legal clauses.

3. **Hybrid Dense + BM25Okapi + Reciprocal Rank Fusion (RRF)**:
   - **Dense Embeddings** excel at semantic synonyms (e.g. mapping "how long does this deal last" to "Term and Duration").
   - **BM25Okapi** excels at exact legal keywords, alphanumeric clause references, specific party names, dates, and currency values.
   - **RRF ($k=60$)** fuses dense and keyword ranks into a robust score without needing cross-metric score normalization.

4. **Two-Stage Reranking (Top 60 Candidates $\rightarrow$ Cross-Encoder $\rightarrow$ Top 10)**:
   - Stage 1 retrieves **60 candidate clauses** via hybrid search to maximize recall.
   - Stage 2 applies a **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) which jointly attends over `(query, chunk_text)` token pairs to score deep legal relevance, surfacing the exact answering clause into the **Top 10**.

5. **Local Ollama LLM (`qwen2.5:7b`) with Citations**:
   - Generates answers strictly grounded in retrieved context with bracketed citations (`[1]`, `[2]`), copies numbers/dates faithfully, and explicitly refuses unmentioned queries.

---

## 2. Information Retrieval Metrics & Mathematical Definitions

The evaluation engine (`scripts/eval/evaluate_retrieval_metrics.py`) evaluates retrieval performance against the **CUAD (Contract Understanding Atticus Dataset)** benchmark across 41 categories using positive QA pairs.

### Relevance Definition
For a retrieved chunk $C$ and ground-truth answer span $A$:
$$\text{rel}(C, A) = \begin{cases} 1 & \text{if } A \subseteq C \lor C \subseteq A \lor \text{TokenOverlap}(C, A) \ge 0.65 \\ 0 & \text{otherwise} \end{cases}$$

### Metric Formulations

#### 1. Recall @ $K$ ($K \in \{1, 5, 10\}$)
Measures whether at least one relevant ground-truth clause was successfully retrieved within the top $K$ positions:
$$\text{Recall}@K = \begin{cases} 1.0 & \text{if } \sum_{i=1}^{K} \text{rel}_i \ge 1 \\ 0.0 & \text{otherwise} \end{cases}$$
The Macro Recall@$K$ across $Q$ queries is:
$$\text{Macro Recall}@K = \frac{1}{|Q|} \sum_{q \in Q} \text{Recall}_q@K$$

#### 2. Precision @ $K$ ($K \in \{1, 5, 10\}$)
Measures the proportion of retrieved chunks in the top $K$ that are relevant:
$$\text{Precision}@K = \frac{1}{K} \sum_{i=1}^{K} \text{rel}_i$$

#### 3. Mean Reciprocal Rank (MRR)
Measures how high up in the ranking the **first** relevant chunk appears:
$$\text{RR}_q = \begin{cases} \frac{1}{\text{rank}_1} & \text{if first relevant chunk is at rank } \text{rank}_1 \le 10 \\ 0 & \text{if no relevant chunk found in top 10} \end{cases}$$
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}_q$$

#### 4. Normalized Discounted Cumulative Gain (nDCG @ $K$)
Measures the ranking quality with logarithmic position discounting:
$$\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
$$\text{IDCG}@K = \sum_{i=1}^{\min(R, K)} \frac{2^1 - 1}{\log_2(i + 1)}$$
$$\text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K} \quad (\text{if IDCG}=0, \text{nDCG}=0)$$

#### 5. Mean Average Precision (MAP)
Calculates the mean of Average Precision scores, rewarding systems that place all relevant items high in the ranking:
$$\text{AP}_q = \frac{1}{\min(R, K)} \sum_{k=1}^{K} \text{Precision}@k \cdot \text{rel}_k$$
$$\text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}_q$$

---

## 3. Directory Layout & Created Scripts

```
.
├── Makefile                               # make run, make test, make ingest, make ingest-cuad
├── README.md                              # Project overview and quick start
├── docs/
│   └── RAG_CUAD_BENCHMARK_AND_USAGE.md    # Complete architecture and usage documentation
├── data/
│   ├── cuad/
│   │   ├── annotations/
│   │   │   ├── CUAD_v1.json               # Full CUAD v1 dataset (510 contracts, 41 categories)
│   │   │   ├── train_cuad.json            # 80% Train split (408 contracts)
│   │   │   └── test_cuad.json             # 20% Test split (102 contracts)
│   │   └── pdfs/                          # Raw CUAD PDF agreements (Affiliate, Co-Branding, etc.)
│   ├── faiss_index/                       # Persisted FAISS HNSW 512d index & chunk metadata
│   ├── storage/                           # Parsed JSON documents & raw uploaded PDFs
│   ├── bm25_cache/                        # Persisted BM25Okapi inverted index
│   └── eval/                              # Evaluation reports and benchmark JSON results
├── scripts/
│   ├── run_all.py                         # 🌟 Master All-in-One Pipeline & Benchmark Runner
│   ├── create_test_pdf.py                 # Generates sample synthetic English commercial contract
│   ├── test_local_rag.py                  # End-to-end local FAISS + Qwen3 + Ollama verification
│   ├── test_fastapi_endpoints.py          # In-process FastAPI test suite (/health, /query, etc.)
│   ├── pipeline/
│   │   ├── ingest.py                      # Step 1: PDF text, hierarchy & table ingestion
│   │   └── index_documents.py             # Step 2: Chunker, Qwen3 512d embedder & FAISS indexing
│   ├── eval/
│   │   └── evaluate_retrieval_metrics.py  # Step 3: Recall@K, Precision@K, MRR, nDCG@K, MAP benchmark
│   └── dataset/
│       └── split_cuad_docs.py             # Splits CUAD annotations into 80/20 train/test sets
└── src/
    ├── app/
    │   └── main.py                        # FastAPI web server and REST API
    ├── config/
    │   └── settings.py                    # Application configuration (Pydantic)
    └── core/
        ├── embedding/local_embedder.py    # Qwen3-Embedding-0.6B (MRL 512d truncation)
        ├── indexer/faiss_client.py        # FAISS HNSW vector database
        ├── llm/ollama_client.py           # Local Ollama client
        ├── parser/                        # PyMuPDF extractor, regex hierarchy, tables & NLP
        ├── retrieval/                     # Hybrid retriever, cross-encoder reranker, query rewriting
        └── storage/local_storage.py       # Local filesystem document storage
```

---

## 4. How to Use Individual Scripts

### 1. Ingestion (`scripts/pipeline/ingest.py`)
Parses PDF documents into structured JSON objects saved under `data/storage/parsed/`.
Supports scanning all 3 CUAD folders (`Part_I`, `Part_II`, `Part_III`) automatically.

```bash
# Ingest all CUAD contracts across Part_I, Part_II, Part_III:
python scripts/pipeline/ingest.py --cuad

# Ingest specific parts (e.g. only Part_I and Part_II):
python scripts/pipeline/ingest.py --parts Part_I Part_II

# Ingest a limited batch of CUAD contracts (e.g. 20 contracts):
python scripts/pipeline/ingest.py --cuad --limit 20

# Ingest the sample synthetic English contract:
python scripts/pipeline/ingest.py --pdf data/sample_contract.pdf

# Ingest a custom directory of PDFs:
python scripts/pipeline/ingest.py --dir /path/to/my/contracts/

# Force re-parsing of previously cached PDFs:
python scripts/pipeline/ingest.py --cuad --force
```

### 2. Indexing (`scripts/pipeline/index_documents.py`)
Reads parsed JSONs from `data/storage/parsed/`, generates 512-dim MRL embeddings with `Qwen3-Embedding-0.6B`, and indexes them into the FAISS HNSW index (`data/faiss_index/`).

```bash
# Index all parsed documents currently in storage:
python scripts/pipeline/index_documents.py

# Reset FAISS index before indexing:
python scripts/pipeline/index_documents.py --reset

# Index only specific document IDs:
python scripts/pipeline/index_documents.py --doc-ids msa_acme_01 cuad_creditcards_affiliate
```

### 3. Retrieval Metrics Benchmark (`scripts/eval/evaluate_retrieval_metrics.py`)
Evaluates **Recall@1,5,10**, **Precision@1,5,10**, **MRR**, **nDCG@1,5,10**, and **MAP** across positive CUAD QA pairs.

```bash
# Run standard Hybrid evaluation with Cross-Encoder Reranker (Top 60 -> Top 10):
python scripts/eval/evaluate_retrieval_metrics.py --mode hybrid --candidate-k 60 --top-k 10

# Run Dense-only evaluation:
python scripts/eval/evaluate_retrieval_metrics.py --mode dense --top-k 10

# Run BM25-only evaluation:
python scripts/eval/evaluate_retrieval_metrics.py --mode bm25 --top-k 10

# Evaluate against test split:
python scripts/eval/evaluate_retrieval_metrics.py --annotation data/cuad/annotations/test_cuad.json

# Save report to a custom path:
python scripts/eval/evaluate_retrieval_metrics.py --output data/eval/benchmark_run_01.json
```

---

## 5. How to Use the All-In-One Master Runner (`scripts/run_all.py`)

The master runner orchestrates the entire pipeline from scratch in a single command:
1. Ingests PDF contracts.
2. Indexes chunks with 512d Qwen3 embeddings into FAISS HNSW.
3. Evaluates Recall@1/5/10, Precision@1/5/10, MRR, nDCG@1/5/10, MAP (Top 60 $\rightarrow$ Reranker $\rightarrow$ Top 10).
4. Runs live end-to-end Ollama Q&A with bracketed citations.

```bash
# Ingest 10 CUAD contracts, index, evaluate retrieval metrics, and run QA demo:
python scripts/run_all.py --cuad --limit 10 --candidate-k 60 --top-k 10

# Ingest and benchmark a single contract:
python scripts/run_all.py --pdf data/sample_contract.pdf --reset-index

# Run benchmark on all currently indexed contracts without re-ingesting:
python scripts/run_all.py --limit 0 --candidate-k 60 --top-k 10
```

---

## 6. Running the Local Web Server & API

Start the local FastAPI server:
```bash
make run
# Or: uvicorn src.app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080` in your browser for the Web UI, or access `http://localhost:8080/docs` for the interactive Swagger documentation.

---

## 7. Tuning & Optimizing Performance

To tune retrieval performance for your specific contract collection, adjust these parameters in `.env`:

| Parameter | Recommended Value | Impact |
|---|---|---|
| `EMBEDDING_DIMENSION` | `512` | MRL truncation dimension. 512 saves 50% RAM with near-lossless accuracy. |
| `HNSW_M` | `24` | HNSW connectivity. Increase to `32` for denser connection graphs. |
| `HNSW_EF_CONSTRUCTION` | `100` | Index build depth. |
| `HNSW_EF_SEARCH` | `100` | Query search depth. Increase to `150` for higher recall. |
| `HYBRID_DENSE_WEIGHT` | `0.6` | Dense semantic weight in RRF fusion. |
| `HYBRID_BM25_WEIGHT` | `0.4` | Keyword/exact match weight in RRF fusion. |
| `RERANKER_CANDIDATE_K` | `60` | Number of candidate clauses retrieved in Stage 1 before cross-encoder reranking. |
| `HYBRID_TOP_K` | `10` | Final number of highest-scoring clauses returned to the LLM. |

