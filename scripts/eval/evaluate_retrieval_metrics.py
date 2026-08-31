#!/usr/bin/env python3
"""
Step 3: Comprehensive Retrieval Metrics Evaluation
Evaluates Recall@1,5,10, Precision@1,5,10, MRR, nDCG@1,5,10, and MAP
against CUAD training/test QA pairs for indexed legal contracts.
Supports Hybrid dense+BM25 retrieval (top 60 candidates) followed by Cross-Encoder Reranking (top 10).
"""
import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import settings
from src.core.indexer.faiss_client import FAISSClient
from src.core.embedding.local_embedder import LocalEmbedder
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.reranker import LocalReranker
from src.core.docstore import LocalDocStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("eval.retrieval_metrics")


def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for robust legal span matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_chunk_relevant(chunk_text: str, answers: List[Dict[str, Any]], threshold: float = 0.65) -> bool:
    """
    Check if a retrieved chunk contains or substantially overlaps with any ground-truth answer.
    """
    norm_chunk = normalize_text(chunk_text)
    if not norm_chunk:
        return False

    chunk_tokens = set(norm_chunk.split())

    for ans in answers:
        raw_ans = ans.get("text", "")
        norm_ans = normalize_text(raw_ans)
        if not norm_ans:
            continue

        # 1. Exact or substring match
        if norm_ans in norm_chunk or norm_chunk in norm_ans:
            return True

        # 2. Token overlap (Jaccard / intersection over minimum)
        ans_tokens = set(norm_ans.split())
        if not ans_tokens:
            continue

        intersection = chunk_tokens.intersection(ans_tokens)
        min_len = min(len(chunk_tokens), len(ans_tokens))
        if min_len > 0 and (len(intersection) / min_len) >= threshold:
            return True

        if len(ans_tokens) > 0 and (len(intersection) / len(ans_tokens)) >= 0.70:
            return True

    return False


# --- FIX: Semantic Fallback for Debugging (Strict Scoring) ---
_eval_embedder = None

def get_eval_embedder():
    global _eval_embedder
    if _eval_embedder is None:
        _eval_embedder = LocalEmbedder()
    return _eval_embedder


def is_chunk_relevant_hybrid(chunk_text: str, answers: List[Dict], threshold: float = 0.65) -> bool:
    """
    Hybrid checker: Strict lexical check first. If lexical fails, check semantic similarity
    but ALWAYS return False to keep final scores strict. Only logs semantic matches for debugging.
    """
    # 1. Strict Lexical Check (Claire's requirement)
    if is_chunk_relevant(chunk_text, answers, threshold):
        return True
    
    # 2. Semantic Check (Debug only - does NOT affect score)
    embedder = get_eval_embedder()
    for ans in answers:
        ans_text = ans.get("text", "")
        if not ans_text:
            continue
        try:
            chunk_emb = embedder.embed_query(chunk_text)
            ans_emb = embedder.embed_query(ans_text)
            similarity = float(np.dot(chunk_emb, ans_emb))  # Cosine (L2 normalized)
            
            if similarity > 0.85:
                # Log the false negative to a debug file
                with open("semantic_false_negatives.log", "a", encoding="utf-8") as f:
                    f.write(f"SEMANTIC MATCH ONLY (SIM={similarity:.3f}): '{chunk_text[:50]}...' vs '{ans_text[:50]}...'\n")
                break
        except Exception:
            pass
    
    return False  # Strict evaluator fails safely for scoring


def compute_dcg_at_k(relevance: List[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevance[:k], start=1):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)
    return dcg


def compute_ndcg_at_k(relevance: List[int], total_relevant: int, k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    dcg = compute_dcg_at_k(relevance, k)
    ideal_rel = [1] * min(total_relevant, k)
    idcg = compute_dcg_at_k(ideal_rel, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_average_precision(relevance: List[int], total_relevant: int, k: int = 10) -> float:
    """Average Precision (AP) for a single query."""
    if total_relevant == 0:
        return 0.0

    score = 0.0
    num_relevant_found = 0

    for i, rel in enumerate(relevance[:k], start=1):
        if rel > 0:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / i
            score += precision_at_i

    return score / min(total_relevant, k)


def compute_reciprocal_rank(relevance: List[int], k: int = 10) -> float:
    """Reciprocal Rank (RR): 1 / rank of first relevant chunk."""
    for i, rel in enumerate(relevance[:k], start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


async def evaluate_cuad_retrieval(
    annotation_path: str = "data/cuad/annotations/train_cuad.json",
    retrieval_mode: str = "hybrid",  # "hybrid", "dense", "bm25"
    use_reranker: bool = True,
    candidate_k: int = 60,
    top_k_final: int = 10,
    max_queries: Optional[int] = None,
    output_path: str = "data/eval/cuad_retrieval_metrics.json",
) -> Dict[str, Any]:
    """
    Run evaluation across CUAD positive QA pairs on indexed contracts.
    """
    faiss_client = FAISSClient()
    collection_info = faiss_client.get_collection_info()
    logger.info(f"📊 Evaluating against FAISS Collection: {collection_info['name']} ({collection_info['count']} total chunks)")

    if collection_info["count"] == 0:
        logger.error("FAISS index is empty! Run ingestion and indexing first.")
        return {}

    embedder = LocalEmbedder()
    retriever = HybridRetriever(vector_client=faiss_client, embedder=embedder)
    reranker = LocalReranker(enabled=use_reranker) if use_reranker else None
    docstore = LocalDocStore()

    # 1. Get list of indexed document IDs in FAISS
    chunks_data = faiss_client.get_all_chunks(limit=100000, include=["metadatas"])
    indexed_doc_ids = set()
    for meta in chunks_data.get("metadatas", []):
        doc_id = meta.get("document_id")
        if doc_id:
            indexed_doc_ids.add(doc_id)

    logger.info(f"📑 Total unique indexed documents: {len(indexed_doc_ids)}")

    # 2. Load CUAD annotation dataset
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    # 3. Extract positive QA pairs for indexed documents
    test_queries: List[Dict[str, Any]] = []

    for doc in cuad_data.get("data", []):
        title = doc.get("title", "")
        matching_doc_id = None
        for doc_id in indexed_doc_ids:
            if doc_id == title or doc_id.startswith(title) or title.startswith(doc_id):
                matching_doc_id = doc_id
                break

        if not matching_doc_id:
            continue

        for paragraph in doc.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                if not qa.get("is_impossible", False) and len(qa.get("answers", [])) > 0:
                    test_queries.append({
                        "doc_id": matching_doc_id,
                        "qa_id": qa.get("id"),
                        "question": qa.get("question"),
                        "answers": qa.get("answers", []),
                    })

    logger.info(f"🎯 Extracted {len(test_queries)} positive QA pairs for indexed contracts.")

    if not test_queries:
        logger.warning("No matching QA pairs found for current indexed documents! Using sample test questions.")
        test_queries = [
            {
                "doc_id": "msa_acme_01",
                "qa_id": "sample_term",
                "question": "What is the term and duration of the agreement?",
                "answers": [{"text": "initial period of thirty-six (36) months"}],
            },
            {
                "doc_id": "msa_acme_01",
                "qa_id": "sample_fees",
                "question": "What are the fees and payment terms under this contract?",
                "answers": [{"text": "annual service fee of $120,000 USD, payable in equal monthly installments of $10,000 USD"}],
            },
            {
                "doc_id": "msa_acme_01",
                "qa_id": "sample_governing_law",
                "question": "Which state law governs this contract and where are disputes resolved?",
                "answers": [{"text": "laws of the State of Delaware"}, {"text": "Wilmington, Delaware"}],
            },
            {
                "doc_id": "msa_acme_01",
                "qa_id": "sample_ip",
                "question": "Who owns the intellectual property and deliverables developed under this contract?",
                "answers": [{"text": "work made for hire and shall be the exclusive property of Client"}],
            },
            {
                "doc_id": "msa_acme_01",
                "qa_id": "sample_indemnity",
                "question": "What are the indemnification obligations and late delivery penalties?",
                "answers": [{"text": "indemnify, defend, and hold harmless Client"}, {"text": "$150 USD per business day"}],
            },
        ]

    if max_queries and max_queries > 0:
        test_queries = test_queries[:max_queries]

    logger.info(
        f"🚀 Running retrieval evaluation on {len(test_queries)} queries "
        f"(mode={retrieval_mode}, candidate_k={candidate_k}, reranker={'ON' if use_reranker else 'OFF'}, final_top_k={top_k_final})..."
    )

    # Metrics Accumulators
    recalls_1, recalls_5, recalls_10 = [], [], []
    precisions_1, precisions_5, precisions_10 = [], [], []
    mrrs = []
    ndcgs_1, ndcgs_5, ndcgs_10 = [], [], []
    maps = []

    per_query_results = []
    t_start = time.time()

    for idx, q_item in enumerate(test_queries, start=1):
        doc_id = q_item["doc_id"]
        question = q_item["question"]
        answers = q_item["answers"]

        filter_meta = {"document_id": doc_id} if doc_id else None
        
        # 1. Retrieve Candidate Chunks (top candidate_k)
        retrieval_limit = candidate_k if use_reranker else top_k_final
        if retrieval_mode == "hybrid":
            candidates = await retriever.hybrid_search(question, top_k=retrieval_limit, filter_metadata=filter_meta)
        elif retrieval_mode == "dense":
            candidates = await retriever._dense_search(question, top_k=retrieval_limit, filter_metadata=filter_meta)
        elif retrieval_mode == "bm25":
            candidates = await retriever._bm25_search(question, top_k=retrieval_limit, filter_metadata=filter_meta)
        else:
            candidates = await retriever.hybrid_search(question, top_k=retrieval_limit, filter_metadata=filter_meta)

        # 2. Apply Cross-Encoder Reranker (top 60 -> top 10)
        if use_reranker and reranker and candidates:
            final_chunks = await reranker.rerank(question, candidates, top_n=top_k_final)
        else:
            final_chunks = candidates[:top_k_final]

        # Graded binary relevance vector: check both leaf text and parent section text
        relevance_vector = []
        for c in final_chunks:
            leaf_text = c.get("text", "")
            parent_id = (c.get("metadata") or {}).get("parent_id")
            parent_text = docstore.get(parent_id) if parent_id else None
            is_rel = is_chunk_relevant_hybrid(leaf_text, answers) or (parent_text and is_chunk_relevant_hybrid(parent_text, answers))
            relevance_vector.append(1 if is_rel else 0)
        # Pad to top_k_final
        relevance_vector += [0] * (top_k_final - len(relevance_vector))

        total_relevant = max(1, sum(1 for c in final_chunks if is_chunk_relevant(c.get("text", ""), answers)))

        # 1. Recall @ 1, 5, 10
        r1 = 1.0 if sum(relevance_vector[:1]) > 0 else 0.0
        r5 = 1.0 if sum(relevance_vector[:5]) > 0 else 0.0
        r10 = 1.0 if sum(relevance_vector[:10]) > 0 else 0.0
        recalls_1.append(r1)
        recalls_5.append(r5)
        recalls_10.append(r10)

        # 2. Precision @ 1, 5, 10
        p1 = sum(relevance_vector[:1]) / 1.0
        p5 = sum(relevance_vector[:5]) / 5.0
        p10 = sum(relevance_vector[:10]) / 10.0
        precisions_1.append(p1)
        precisions_5.append(p5)
        precisions_10.append(p10)

        # 3. MRR
        rr = compute_reciprocal_rank(relevance_vector, k=top_k_final)
        mrrs.append(rr)

        # 4. nDCG @ 1, 5, 10
        ndcg1 = compute_ndcg_at_k(relevance_vector, total_relevant, 1)
        ndcg5 = compute_ndcg_at_k(relevance_vector, total_relevant, 5)
        ndcg10 = compute_ndcg_at_k(relevance_vector, total_relevant, 10)
        ndcgs_1.append(ndcg1)
        ndcgs_5.append(ndcg5)
        ndcgs_10.append(ndcg10)

        # 5. MAP
        ap = compute_average_precision(relevance_vector, total_relevant, k=top_k_final)
        maps.append(ap)

        per_query_results.append({
            "qa_id": q_item["qa_id"],
            "doc_id": doc_id,
            "question": question,
            "relevance_vector": relevance_vector[:top_k_final],
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
            "precision@1": p1,
            "precision@5": p5,
            "precision@10": p10,
            "mrr": rr,
            "ndcg@1": ndcg1,
            "ndcg@5": ndcg5,
            "ndcg@10": ndcg10,
            "map": ap,
        })

    elapsed_total = time.time() - t_start

    # Summary Macro Metrics
    metrics_summary = {
        "dataset": annotation_path,
        "total_queries_evaluated": len(test_queries),
        "retrieval_mode": retrieval_mode,
        "reranker_enabled": use_reranker,
        "reranker_candidate_k": candidate_k,
        "final_top_k": top_k_final,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "elapsed_seconds": round(elapsed_total, 2),
        "queries_per_second": round(len(test_queries) / max(elapsed_total, 0.001), 2),
        "macro_metrics": {
            "Recall@1": float(np.mean(recalls_1)),
            "Recall@5": float(np.mean(recalls_5)),
            "Recall@10": float(np.mean(recalls_10)),
            "Precision@1": float(np.mean(precisions_1)),
            "Precision@5": float(np.mean(precisions_5)),
            "Precision@10": float(np.mean(precisions_10)),
            "MRR": float(np.mean(mrrs)),
            "nDCG@1": float(np.mean(ndcgs_1)),
            "nDCG@5": float(np.mean(ndcgs_5)),
            "nDCG@10": float(np.mean(ndcgs_10)),
            "MAP": float(np.mean(maps)),
        }
    }

    # Save output to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": metrics_summary,
            "queries": per_query_results,
        }, f, indent=2, ensure_ascii=False)

    # Print Formatted Table
    print("\n" + "=" * 70)
    print(" 🏆 CUAD RETRIEVAL BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    print(f" Dataset             : {annotation_path}")
    print(f" Embedding Model     : {settings.EMBEDDING_MODEL_NAME} (MRL {settings.EMBEDDING_DIMENSION}d)")
    print(f" Retrieval Pipeline  : {retrieval_mode.upper()} (Candidates={candidate_k} -> Reranker={'ON (' + settings.RERANKER_MODEL_NAME + ')' if use_reranker else 'OFF'} -> Final Top {top_k_final})")
    print(f" Evaluated Queries   : {len(test_queries)}")
    print(f" Benchmark Latency   : {elapsed_total:.2f}s ({metrics_summary['queries_per_second']} queries/sec)")
    print("-" * 70)
    print(f"  {'Metric':<20} | {'Score':<10} | {'Description'}")
    print("-" * 70)
    print(f"  {'Recall@1':<20} | {metrics_summary['macro_metrics']['Recall@1']:.4f}     | Top-1 span hit rate")
    print(f"  {'Recall@5':<20} | {metrics_summary['macro_metrics']['Recall@5']:.4f}     | Top-5 span hit rate")
    print(f"  {'Recall@10':<20} | {metrics_summary['macro_metrics']['Recall@10']:.4f}     | Top-10 span hit rate")
    print(f"  {'Precision@1':<20} | {metrics_summary['macro_metrics']['Precision@1']:.4f}     | Relevant fraction in Top 1")
    print(f"  {'Precision@5':<20} | {metrics_summary['macro_metrics']['Precision@5']:.4f}     | Relevant fraction in Top 5")
    print(f"  {'Precision@10':<20} | {metrics_summary['macro_metrics']['Precision@10']:.4f}     | Relevant fraction in Top 10")
    print(f"  {'MRR':<20} | {metrics_summary['macro_metrics']['MRR']:.4f}     | Mean Reciprocal Rank (1/rank)")
    print(f"  {'nDCG@1':<20} | {metrics_summary['macro_metrics']['nDCG@1']:.4f}     | Discounted gain @ 1")
    print(f"  {'nDCG@5':<20} | {metrics_summary['macro_metrics']['nDCG@5']:.4f}     | Discounted gain @ 5")
    print(f"  {'nDCG@10':<20} | {metrics_summary['macro_metrics']['nDCG@10']:.4f}     | Discounted gain @ 10")
    print(f"  {'MAP':<20} | {metrics_summary['macro_metrics']['MAP']:.4f}     | Mean Average Precision")
    print("=" * 70)
    print(f" Detailed results saved to: {output_path}\n")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Legal RAG retrieval on CUAD benchmark QA pairs.")
    parser.add_argument("--annotation", type=str, default="data/cuad/annotations/train_cuad.json", help="Path to CUAD json annotation file.")
    parser.add_argument("--mode", type=str, choices=["hybrid", "dense", "bm25"], default="hybrid", help="Retrieval mode to evaluate.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking.")
    parser.add_argument("--candidate-k", type=int, default=60, help="Number of candidate chunks retrieved before reranking.")
    parser.add_argument("--top-k", type=int, default=10, help="Final top-k chunks returned.")
    parser.add_argument("--max-queries", type=int, default=None, help="Maximum number of queries to evaluate.")
    parser.add_argument("--output", type=str, default="data/eval/cuad_retrieval_metrics.json", help="Output JSON path for evaluation report.")
    args = parser.parse_args()

    asyncio.run(evaluate_cuad_retrieval(
        annotation_path=args.annotation,
        retrieval_mode=args.mode,
        use_reranker=not args.no_rerank,
        candidate_k=args.candidate_k,
        top_k_final=args.top_k,
        max_queries=args.max_queries,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()

