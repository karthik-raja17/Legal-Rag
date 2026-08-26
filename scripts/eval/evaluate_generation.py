#!/usr/bin/env python3
"""
Production‑grade generation evaluation for Legal RAG.
- Uses Gemini 2.5‑Flash as the evaluator judge (faithfulness, relevance, semantic equivalence, numeric correctness, refusal).
- Caches all results (queries, Gemini judgments) to disk.
- Parallelizes HTTP requests with semaphore concurrency control.
- Retries on transient rate limits (429) and network errors.
- Outputs detailed metrics and a pass/fail summary.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)

from src.config.settings import settings

# ======================= CONFIGURATION =======================
PARSER_URL = os.getenv("PARSER_URL", "https://your-parser-service-url/query")
GOLDEN_PATH = os.getenv("GOLDEN_PATH", "data/golden_with_chunks_bge.jsonl")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/generation_eval_results_prod.json")
CACHE_PATH = os.getenv("CACHE_PATH", "data/eval_cache.json")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "5.0"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "1200"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================= INIT VERTEX JUDGE =======================
vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
judge_model = GenerativeModel("gemini-2.5-flash")

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# ======================= DATA LOADING =======================
def load_data():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    positive, hard_neg_static, absent, unmatched_present = [], [], [], []
    for e in entries:
        if e.get("answer_quote") == "ABSENT_DU_CONTRAT":
            absent.append(e)
        elif not e.get("chunk_ids"):
            unmatched_present.append(e)
        else:
            has_unmatched = any(d.get("method") == "unmatched" for d in e.get("chunk_match_details", []))
            if has_unmatched:
                hard_neg_static.append(e)
            else:
                positive.append(e)
    return positive, hard_neg_static, absent, unmatched_present

# ======================= CACHE =======================
class EvalCache:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info(f"Loaded cache with {len(self.data)} entries.")
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def flush(self):
        self.save()

cache = EvalCache(CACHE_PATH)

# ======================= ASYNC HTTP HELPER =======================
async def query_rag_async(session: aiohttp.ClientSession, question: str, doc_id: Optional[str] = None) -> Dict:
    key = f"rag_{question}_{doc_id}"
    cached = cache.get(key)
    if cached:
        return cached

    payload = {
        "query": question,
        "generate": True,
        "hybrid": True,
        "rerank": True,
        "expand": False,
        "auto_optimize": False,
        "top_k": 5,
    }
    if doc_id:
        payload["document_id"] = doc_id

    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with session.post(
                PARSER_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                cache.set(key, result)
                return result
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Attempt {attempt+1} failed for {question[:30]}...: {e}")
            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
    raise RuntimeError(f"Failed to query after {RETRY_ATTEMPTS} attempts: {question}")

# ======================= GEMINI JUDGE =======================
async def gemini_judge_async(
    query: str,
    context: str,
    ground_truth_quote: str,
    ground_truth_summary: str,
    generated_answer: str,
    is_absent: bool,
) -> Dict:
    judge_key = f"judge_{hash(query + generated_answer)}"
    cached = cache.get(judge_key)
    if cached:
        return cached

    truncated_context = context[:2500]
    truncated_answer = generated_answer[:1000] if generated_answer else ""

    prompt = f"""
You are a strict legal evaluator. Evaluate the Generated Answer based ONLY on the provided context and ground truth.

Return a valid JSON object matching this schema:
{{
  "faithfulness_score": <integer from 1 to 5, where 5 means fully supported by retrieved context and 1 means invented/contradictory>,
  "refusal_correct": <boolean: true if absent query and assistant properly refused/stated absence, false otherwise. If not absent, return null>,
  "numeric_correct": <boolean: true if numbers/dates/durations are preserved accurately, false if wrong, null if no numbers>,
  "relevance_score": <integer from 1 to 5: directness in answering the query>,
  "semantic_equivalence": <integer from 1 to 5: similarity of meaning to ground truth summary, null if absent query>
}}

**User Query:** {query[:200]}
**Retrieved Context:** {truncated_context}
**Ground Truth Quote:** {ground_truth_quote[:300] if ground_truth_quote else "N/A"}
**Ground Truth Summary:** {ground_truth_summary[:300] if ground_truth_summary else "N/A"}
**Generated Answer:** {truncated_answer}
"""

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = judge_model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    max_output_tokens=2048,
                ),
                safety_settings=SAFETY_SETTINGS,
            )

            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("Empty candidate returned by Gemini judge.")

            result = json.loads(response.text.strip())
            defaults = {
                "faithfulness_score": 3,
                "refusal_correct": None,
                "numeric_correct": None,
                "relevance_score": 3,
                "semantic_equivalence": None,
            }
            for k, v in defaults.items():
                if k not in result:
                    result[k] = v
            cache.set(judge_key, result)
            return result
        except Exception as e:
            logger.warning(f"Gemini judge attempt {attempt+1} failed: {e}")
            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))

    logger.error(f"Gemini judge permanently failed for query: {query[:50]}...")
    return {
        "faithfulness_score": 3,
        "refusal_correct": False if is_absent else None,
        "numeric_correct": None,
        "relevance_score": 3,
        "semantic_equivalence": None,
    }

# ======================= PROCESS SINGLE ENTRY =======================
async def process_entry(entry: Dict, session: aiohttp.ClientSession, category: str) -> Dict:
    question = entry["question"]
    doc_id = entry.get("doc_id")
    result = await query_rag_async(session, question, doc_id)
    answer = result.get("answer", "")
    retrieved_chunks = result.get("retrieved_chunks", [])
    context = "\n".join([c.get("text", "") for c in retrieved_chunks])

    judge_scores = await gemini_judge_async(
        question,
        context,
        entry.get("answer_quote", ""),
        entry.get("answer_summary", ""),
        answer,
        is_absent=(category == "absent"),
    )

    faith_score = judge_scores.get("faithfulness_score", 3) / 5.0

    out = {
        "type": category,
        "question": question,
        "answer": answer,
        "faithfulness": faith_score,
        "refusal_correct": judge_scores.get("refusal_correct"),
        "numeric_correct": judge_scores.get("numeric_correct"),
        "relevance_score": judge_scores.get("relevance_score"),
        "semantic_equivalence": judge_scores.get("semantic_equivalence"),
    }

    if category == "positive":
        ground_ids = set(entry.get("chunk_ids", []))
        primary_retrieved_id = retrieved_chunks[0].get("id") if retrieved_chunks else None
        top3_retrieved_ids = {c.get("id") for c in retrieved_chunks[:3]}

        citation_correct = bool(primary_retrieved_id in ground_ids or (top3_retrieved_ids & ground_ids))
        hard_neg_fooled = not citation_correct

        out["citation_correct"] = citation_correct
        out["hard_negative_fooled"] = hard_neg_fooled

    return out

# ======================= MAIN EVALUATION =======================
async def main():
    global GOLDEN_PATH, OUTPUT_PATH, MAX_CONCURRENT

    parser = argparse.ArgumentParser(description="Evaluate Legal RAG generation quality.")
    parser.add_argument("--golden", default=GOLDEN_PATH, help="Path to golden dataset JSONL.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output JSON file.")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent queries.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions per category.")
    args = parser.parse_args()

    GOLDEN_PATH = args.golden
    OUTPUT_PATH = args.output
    MAX_CONCURRENT = args.concurrent

    logger.info(f"📂 Loading dataset from {GOLDEN_PATH}...")
    positive, hard_neg_static, absent, unmatched_present = load_data()
    logger.info(f"✅ Positive: {len(positive)}")
    logger.info(f"✅ Hard Negatives (static unmatched): {len(hard_neg_static)}")
    logger.info(f"✅ Absent: {len(absent)}")
    logger.info(f"✅ Unmatched Present: {len(unmatched_present)}")

    if args.limit:
        positive = positive[:args.limit]
        hard_neg_static = hard_neg_static[:args.limit]
        absent = absent[:args.limit]
        unmatched_present = unmatched_present[:args.limit]
        logger.info(f"⚠️ Limited to {args.limit} questions per category for testing.")

    all_entries = []
    for e in absent:
        all_entries.append((e, "absent"))
    for e in positive:
        all_entries.append((e, "positive"))
    for e in unmatched_present:
        all_entries.append((e, "unmatched_present"))
    for e in hard_neg_static:
        all_entries.append((e, "positive"))

    total = len(all_entries)
    logger.info(f"📊 Total questions to evaluate: {total}")

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_with_semaphore(entry, category):
        async with sem:
            async with aiohttp.ClientSession() as session:
                return await process_entry(entry, session, category)

    async def safe_process(entry, category):
        try:
            return await process_with_semaphore(entry, category)
        except Exception as e:
            logger.error(f"❌ Failed for question: {entry['question'][:60]}... Error: {e}")
            return {
                "type": category,
                "question": entry["question"],
                "error": str(e),
            }

    tasks = [asyncio.create_task(safe_process(entry, category)) for entry, category in all_entries]

    results = []
    completed = 0
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        completed += 1
        if completed % 10 == 0 or completed == total:
            cache.flush()
            elapsed = time.time() - start_time
            logger.info(f"📊 Progress: {completed}/{total} completed, elapsed {elapsed:.1f}s")

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Evaluation completed in {elapsed:.1f} seconds.")

    cache.flush()

    # ---- AGGREGATE METRICS ----
    logger.info("\n" + "=" * 70)
    logger.info("📊 FINAL GENERATION EVALUATION REPORT")
    logger.info("=" * 70)

    absent_res = [r for r in results if r.get("type") == "absent" and "error" not in r]
    pos_res = [r for r in results if r.get("type") == "positive" and "error" not in r]
    unmatched_res = [r for r in results if r.get("type") == "unmatched_present" and "error" not in r]

    refusal_correct = [r["refusal_correct"] for r in absent_res if r.get("refusal_correct") is not None]
    refusal_rate = float(np.mean(refusal_correct)) if refusal_correct else 0.0

    citation_correct = [r["citation_correct"] for r in pos_res if r.get("citation_correct") is not None]
    citation_accuracy = float(np.mean(citation_correct)) if citation_correct else 0.0
    hard_neg_fooled = [r["hard_negative_fooled"] for r in pos_res if r.get("hard_negative_fooled") is not None]
    hard_neg_distraction_rate = float(np.mean(hard_neg_fooled)) if hard_neg_fooled else 0.0

    present_res = pos_res + unmatched_res
    faith_scores = [r["faithfulness"] for r in present_res if r.get("faithfulness") is not None]
    num_correct = [r["numeric_correct"] for r in present_res if r.get("numeric_correct") is not None]
    rel_scores = [r["relevance_score"] for r in present_res if r.get("relevance_score") is not None]
    sem_scores = [r["semantic_equivalence"] for r in present_res if r.get("semantic_equivalence") is not None]

    avg_faith = float(np.mean(faith_scores)) if faith_scores else 0.0
    avg_num = float(np.mean(num_correct)) if num_correct else 0.0
    avg_rel = float(np.mean(rel_scores)) if rel_scores else 0.0
    avg_sem = float(np.mean(sem_scores)) if sem_scores else 0.0

    logger.info(f"✅ Absent Clause Refusal Rate:          {refusal_rate:.2%}  [Target: 100%]")
    logger.info(f"✅ Citation Accuracy (Positive):        {citation_accuracy:.2%}  [Target: ≥ 90%]")
    logger.info(f"✅ Hard Negative Distraction Rate:      {hard_neg_distraction_rate:.2%}  [Target: ≤ 10%]")
    logger.info(f"✅ Faithfulness (LLM-as-judge):         {avg_faith:.4f}  [Target: ≥ 0.95]")
    logger.info(f"✅ Numeric/Entity Correctness:          {avg_num:.2%}  [Target: ≥ 95%]")
    logger.info(f"✅ Answer Relevance (1-5):              {avg_rel:.2f}  [Target: ≥ 4.5]")
    logger.info(f"✅ Semantic Equivalence (1-5):          {avg_sem:.2f}  [Target: ≥ 4.0]")

    # ---- PASS/FAIL DECISION ----
    logger.info("\n" + "=" * 70)
    pass_conditions = (
        refusal_rate == 1.0 and
        citation_accuracy >= 0.90 and
        hard_neg_distraction_rate <= 0.10 and
        avg_faith >= 0.95 and
        avg_num >= 0.95 and
        avg_rel >= 4.5 and
        avg_sem >= 4.0
    )

    if pass_conditions:
        logger.info("✅ PRODUCTION STATUS: PASS.")
        logger.info("   🚀 The Legal RAG pipeline is safe, faithful, and accurate.")
    else:
        logger.info("❌ PRODUCTION STATUS: FAIL.")
        logger.info("   Address the following gaps:")
        if refusal_rate < 1.0:
            logger.info("      - Refusal failure on absent clauses.")
        if citation_accuracy < 0.90:
            logger.info("      - Insufficient citation accuracy on positive pairs.")
        if hard_neg_distraction_rate > 0.10:
            logger.info("      - Model impacted by hard distractors.")
        if avg_faith < 0.95:
            logger.info("      - Faithfulness score below threshold.")
        if avg_num < 0.95:
            logger.info("      - Numeric or key entity inaccuracies.")
        if avg_rel < 4.5:
            logger.info("      - Low answer relevance scores.")
        if avg_sem < 4.0:
            logger.info("      - Semantic drift relative to ground truth.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"\n📁 Detailed results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())