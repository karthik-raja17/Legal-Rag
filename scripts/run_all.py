#!/usr/bin/env python3
"""
Master Orchestrator: Complete Legal RAG Pipeline & Benchmark Runner
Executes:
  1. Ingestion of contracts (PyMuPDF + Structure + Tables + English NLP)
  2. Indexing into FAISS HNSW (Qwen3-Embedding-0.6B, MRL 512d)
  3. Retrieval Metrics Evaluation (Recall@1/5/10, Precision@1/5/10, MRR, nDCG@1/5/10, MAP with top 60 -> Reranker -> top 10)
  4. End-to-end Ollama RAG Generation test with bracketed citations
"""
import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import settings
from scripts.pipeline.ingest import run_ingest
from scripts.pipeline.index_documents import run_indexing
from scripts.eval.evaluate_retrieval_metrics import evaluate_cuad_retrieval
from src.core.indexer.faiss_client import FAISSClient
from src.core.embedding.local_embedder import LocalEmbedder
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.reranker import LocalReranker
from src.core.llm.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline.run_all")


async def execute_sample_generation():
    """Run a live end-to-end generation test on the indexed knowledge base."""
    print("\n" + "=" * 70)
    print(" 🤖 END-TO-END RAG QUESTION ANSWERING DEMO (OLLAMA)")
    print("=" * 70)

    faiss_client = FAISSClient()
    embedder = LocalEmbedder()
    retriever = HybridRetriever(vector_client=faiss_client, embedder=embedder)
    reranker = LocalReranker(enabled=True)
    ollama = OllamaClient()

    if not ollama.health_check():
        print("⚠️ Ollama is not accessible on localhost:11434. Skipping generation demo.")
        return

    sample_questions = [
        "What are the fees, payment terms, or compensation structure under the agreement?",
        "What are the termination conditions and notice periods?",
        "Which state or country's law governs this agreement?",
    ]

    for q in sample_questions:
        print(f"\n❓ Question: {q}")
        t0 = time.time()

        # Retrieve top 60 -> Rerank to top 10
        candidates = await retriever.hybrid_search(q, top_k=settings.RERANKER_CANDIDATE_K)
        top_chunks = await reranker.rerank(q, candidates, top_n=settings.HYBRID_TOP_K)

        context_parts = []
        for idx, c in enumerate(top_chunks, start=1):
            context_parts.append(f"[{idx}] {c['text']}")
        context_str = "\n\n".join(context_parts)

        prompt = f"Contract Context:\n{context_str}\n\nQuestion: {q}\n\nAnswer:"
        system = (
            "You are a professional legal contract assistant. "
            "Answer directly and precisely using ONLY the provided context. "
            "Cite sources in brackets [1], [2]. "
            "Preserve exact numbers, dates, and currency figures."
        )

        answer = await ollama.agenerate(prompt=prompt, system=system, temperature=0.1)
        elapsed = time.time() - t0

        print(f"💡 Answer ({elapsed:.2f}s):\n{answer}\n")
        print("-" * 70)


async def main_async(args):
    start_time = time.time()

    print("=" * 70)
    print(" ⚡ LEGAL RAG ENGINE — MASTER PIPELINE & BENCHMARK RUNNER")
    print("=" * 70)
    print(f" • Embedding Model  : {settings.EMBEDDING_MODEL_NAME} (MRL {settings.EMBEDDING_DIMENSION}d)")
    print(f" • Vector Store     : FAISS HNSW (M={settings.HNSW_M}, efC={settings.HNSW_EF_CONSTRUCTION}, efS={settings.HNSW_EF_SEARCH})")
    print(f" • Reranker Pipeline: Top {args.candidate_k} Candidates -> Local Cross-Encoder -> Top {args.top_k}")
    print(f" • LLM Generator    : Ollama ({settings.OLLAMA_MODEL})")
    print("=" * 70)

    # 1. INGESTION
    print("\n📦 STEP 1: DOCUMENT INGESTION")
    parsed_ids = run_ingest(
        pdf_path=args.pdf,
        dir_path=args.dir,
        cuad=args.cuad,
        parts=args.parts,
        limit=args.limit,
        force=args.force_parse
    )
    print(f"✅ Ingestion complete: {len(parsed_ids)} documents parsed.")

    # 2. INDEXING
    print("\n🔍 STEP 2: FAISS HNSW INDEXING (512d MRL)")
    total_chunks = run_indexing(
        doc_ids=parsed_ids if not args.index_all else None,
        reset_index=args.reset_index,
    )
    print(f"✅ Indexing complete: {total_chunks} total chunks in FAISS vector store.")

    # 3. BENCHMARK RETRIEVAL METRICS
    print("\n📊 STEP 3: RETRIEVAL EVALUATION BENCHMARK (CUAD QA)")
    eval_results = await evaluate_cuad_retrieval(
        annotation_path=args.annotation,
        retrieval_mode=args.retrieval_mode,
        use_reranker=not args.no_rerank,
        candidate_k=args.candidate_k,
        top_k_final=args.top_k,
        max_queries=args.max_eval_queries,
        output_path=args.output,
    )

    # 4. SAMPLE GENERATION DEMO
    if not args.skip_generation:
        await execute_sample_generation()

    total_time = time.time() - start_time
    print(f"🎉 MASTER PIPELINE COMPLETED IN {total_time:.2f}s!\n")


def main():
    parser = argparse.ArgumentParser(description="Master runner for Legal RAG ingestion, indexing, and CUAD evaluation.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to single PDF to ingest.")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of PDFs to ingest.")
    parser.add_argument("--cuad", action="store_true", help="Ingest from data/cuad/pdfs/ (all Part_I, Part_II, Part_III).")
    parser.add_argument("--parts", nargs="+", choices=["Part_I", "Part_II", "Part_III"], default=None, help="Specific CUAD parts to ingest.")
    parser.add_argument("--limit", type=int, default=5, help="Number of contract PDFs to ingest (default: 5).")
    parser.add_argument("--force-parse", action="store_true", help="Force re-parsing of PDFs.")
    parser.add_argument("--reset-index", action="store_true", help="Reset existing FAISS index before indexing.")
    parser.add_argument("--index-all", action="store_true", help="Index all parsed documents in storage.")
    parser.add_argument("--annotation", type=str, default="data/cuad/annotations/train_cuad.json", help="Path to CUAD annotation json.")
    parser.add_argument("--retrieval-mode", type=str, choices=["hybrid", "dense", "bm25"], default="hybrid", help="Retrieval mode.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking.")
    parser.add_argument("--candidate-k", type=int, default=60, help="Number of candidate chunks before reranking (default: 60).")
    parser.add_argument("--top-k", type=int, default=10, help="Final top-k chunks evaluated (default: 10).")
    parser.add_argument("--max-eval-queries", type=int, default=None, help="Limit number of QA queries evaluated.")
    parser.add_argument("--output", type=str, default="data/eval/cuad_retrieval_metrics.json", help="Output evaluation report path.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip live Ollama generation demo.")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

