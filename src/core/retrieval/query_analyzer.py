"""
Query complexity analyzer for adaptive retrieval parameters (English & CUAD contracts).

Supports two modes:
- heuristic: fast rule-based estimation (word count, legal terms, numbers, question type)
- llm: uses Ollama to classify complexity and suggest parameters (JSON output)
"""
import json
import logging
import re
from typing import Dict, Any, Optional

from src.core.llm.ollama_client import OllamaClient
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.enabled = settings.QUERY_ANALYZER_ENABLED
        self.mode = settings.QUERY_ANALYZER_MODE
        self.llm = ollama_client

        # English legal contract keywords (aligned with CUAD benchmark)
        self.legal_keywords = [
            "indemnif", "indemnity", "hold harmless", "liability", "damages",
            "termination", "expire", "cancel", "default", "breach",
            "governing law", "jurisdiction", "venue", "dispute", "arbitration",
            "confidential", "non-disclosure", "proprietary", "trade secret",
            "intellectual property", "ip", "patent", "trademark", "copyright",
            "non-compete", "non-solicit", "exclusivity", "restriction",
            "payment", "fee", "compensation", "royalty", "reimbursement", "price",
            "warranty", "warranties", "representation", "covenant", "obligation",
            "force majeure", "audit", "assignment", "assignability", "amendment",
            "severability", "entire agreement", "effective date", "term", "renewal"
        ]

        # Question type hints
        self.question_words = {
            "simple": ["who", "when", "where", "what date", "how much", "how many", "is there", "does"],
            "moderate": ["what is", "which", "what are", "what obligations"],
            "complex": ["how", "why", "under what conditions", "in the event of", "what remedies", "exceptions to"]
        }

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity and return suggested retrieval parameters.
        """
        if not self.enabled:
            return self._default_analysis()

        if self.mode == "llm":
            try:
                return self._analyze_llm(query)
            except Exception as e:
                logger.warning(f"LLM analysis failed ({e}), falling back to heuristic")
                return self._analyze_heuristic(query)

        return self._analyze_heuristic(query)

    def _default_analysis(self) -> Dict[str, Any]:
        return {
            "complexity": "medium",
            "suggested_top_k": 5,
            "suggested_expand": False,
            "suggested_rerank": False,
            "raw_score": 0.0,
            "word_count": 0,
            "legal_count": 0,
            "has_numbers": False,
            "question_type": "unknown"
        }

    def _analyze_heuristic(self, query: str) -> Dict[str, Any]:
        words = re.findall(r'\b\w+\b', query.lower())
        word_count = len(words)
        lower_q = query.lower()
        legal_count = sum(1 for kw in self.legal_keywords if kw in lower_q)
        has_numbers = bool(re.search(r'\d+', query))

        qtype = "moderate"
        for q_type, terms in self.question_words.items():
            if any(term in lower_q for term in terms):
                qtype = q_type
                break

        score = 0.0
        if word_count > 6:
            score += 0.5
        if word_count > 15:
            score += 0.5
        if legal_count >= 2:
            score += 0.5
        if legal_count >= 4:
            score += 0.5
        if has_numbers:
            score += 0.5
        if qtype == "complex":
            score += 0.5
        elif qtype == "moderate":
            score += 0.25

        if score <= 1.0:
            complexity = "low"
        elif score <= 2.5:
            complexity = "medium"
        else:
            complexity = "high"

        top_k_map = {
            "low": settings.QUERY_ANALYZER_TOP_K_LOW,
            "medium": settings.QUERY_ANALYZER_TOP_K_MEDIUM,
            "high": settings.QUERY_ANALYZER_TOP_K_HIGH
        }
        expand_map = settings.QUERY_ANALYZER_EXPAND_MAP
        rerank_map = settings.QUERY_ANALYZER_RERANK_MAP

        return {
            "complexity": complexity,
            "suggested_top_k": top_k_map.get(complexity, 5),
            "suggested_expand": expand_map.get(complexity, False),
            "suggested_rerank": rerank_map.get(complexity, False),
            "raw_score": score,
            "word_count": word_count,
            "legal_count": legal_count,
            "has_numbers": has_numbers,
            "question_type": qtype
        }

    def _analyze_llm(self, query: str) -> Dict[str, Any]:
        if self.llm is None:
            self.llm = OllamaClient()

        prompt = f"""
You are an expert legal query analyzer.
Analyze the following query and return a valid JSON object with the following fields:
- "complexity": "low", "medium", or "high"
- "suggested_top_k": integer (between 3 and 10)
- "suggested_expand": boolean
- "suggested_rerank": boolean

Query: {query}
"""
        response_text = self.llm.generate(
            prompt=prompt,
            json_mode=True,
            temperature=0.0,
            max_tokens=128
        )
        data = json.loads(response_text)
        return {
            "complexity": data.get("complexity", "medium"),
            "suggested_top_k": data.get("suggested_top_k", 5),
            "suggested_expand": data.get("suggested_expand", False),
            "suggested_rerank": data.get("suggested_rerank", False),
            "raw_score": None,
            "word_count": None,
            "legal_count": None,
            "has_numbers": None,
            "question_type": None
        }