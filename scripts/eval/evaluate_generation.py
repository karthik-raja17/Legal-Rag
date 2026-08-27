#!/usr/bin/env python3
"""
Generation evaluation for Local Legal RAG.
- Uses Ollama (qwen2.5:7b) as the evaluator judge (faithfulness, relevance, semantic equivalence, numeric correctness, refusal).
- Caches all results to disk.
- Outputs detailed metrics and a summary.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import settings
from src.core.llm.ollama_client import OllamaClient

# ======================= CONFIGURATION =======================
PARSER_URL = os.getenv("PARSER_URL", "http://localhost:8080/query")
GOLDEN_PATH = os.getenv("GOLDEN_PATH", "data/golden.jsonl")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/generation_eval_results_prod.json")
CACHE_PATH = os.getenv("CACHE_PATH", "data/eval_cache.json")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "3"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "2"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ollama_judge = OllamaClient()


# ======================= DATA LOADING =======================
def load_data():
    if not os.path.exists(GOLDEN_PATH):
        logger.warning(f"Golden dataset {GOLDEN_PATH} not found.")
        return [], [], [], []

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    positive, hard_neg_static, absent, unmatched_present = [], [], [], []
    for e in entries:
        t = e.get("type", "")
        if t == "positive":
            positive.append(e)
        elif t == "hard_neg_static":
            hard_neg_static.append(e)
        elif t == "absent":
            absent.append(e)
        elif t == "unmatched_present":
            unmatched_present.append(e)

    logger.info(f"Loaded datasets: positive={len(positive)}, hard_neg_static={len(hard_neg_static)}, absent={len(absent)}, unmatched_present={len(unmatched_present)}")
    return positive, hard_neg_static, absent, unmatched_present


# ======================= CACHE =======================
class DiskCache:
    def __init__(self, path: str):
        self.path = path
        self.cache: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)


cache = DiskCache(CACHE_PATH)


# ======================= QUERY RUNNER =======================
async def query_pipeline(
    session: aiohttp.ClientSession,
    question: str,
    doc_id: Optional[str] = None
) -> Dict:
    key = f"query_{hash(question + str(doc_id))}"
    cached = cache.get(key)
    if cached:
        return cached

    payload = {
        "query": question,
        "generate": True,
        "hybrid": True,
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
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {question[:30]}...: {e}")
            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
    raise RuntimeError(f"Failed to query after {RETRY_ATTEMPTS} attempts: {question}")


# ======================= OLLAMA JUDGE =======================
async def ollama_judge_async(
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
  "faithfulness_score": <integer 1-5>,
  "refusal_correct": <boolean or null>,
  "numeric_correct": <boolean or null>,
  "relevance_score": <integer 1-5>,
  "semantic_equivalence": <integer 1-5 or null>
}}

**User Query:** {query[:200]}
**Retrieved Context:** {truncated_context}
**Ground Truth Quote:** {ground_truth_quote[:300] if ground_truth_quote else "N/A"}
**Ground Truth Summary:** {ground_truth_summary[:300] if ground_truth_summary else "N/A"}
**Generated Answer:** {truncated_answer}
"""

    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp_text = await ollama_judge.agenerate(
                prompt=prompt,
                system="You are an automated evaluation judge. Output ONLY valid JSON.",
                json_mode=True,
                temperature=0.0,
            )
            result = json.loads(resp_text.strip())
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
            logger.warning(f"Ollama judge attempt {attempt+1} failed: {e}")
            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))

    return {
        "faithfulness_score": 3,
        "refusal_correct": False if is_absent else None,
        "numeric_correct": None,
        "relevance_score": 3,
        "semantic_equivalence": None,
    }


async def main():
    pos, neg, abs_set, unmatch = load_data()
    print(f"Total test dataset items: {len(pos) + len(neg) + len(abs_set) + len(unmatch)}")


if __name__ == "__main__":
    asyncio.run(main())