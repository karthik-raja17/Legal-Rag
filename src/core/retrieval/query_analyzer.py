"""
Query complexity analyzer for adaptive retrieval parameters.

Supports two modes:
- heuristic: fast rule‑based estimation (word count, legal terms, numbers, question type)
- llm: uses Gemini to classify complexity and suggest parameters (more accurate, higher latency)

Unified interface: analyze(query) returns complexity, suggested_top_k, expand, rerank.
"""
import json
import logging
import re
from typing import Dict, Any, Optional

from vertexai.generative_models import GenerativeModel
from src.settings import settings

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    def __init__(self):
        self.enabled = settings.QUERY_ANALYZER_ENABLED
        self.mode = settings.QUERY_ANALYZER_MODE
        self.llm_model_name = settings.QUERY_ANALYZER_LLM_MODEL

        # French legal keywords (extend as needed)
        self.legal_keywords = [
            "obligation", "indemnité", "garantie", "responsabilité", "assurance",
            "confidentialité", "résiliation", "préavis", "pénalité", "délai",
            "frais", "prix", "révision", "loyer", "caution", "dépôt",
            "tribunal", "arbitrage", "exécution", "livraison", "performance",
            "contrat", "clause", "article", "annexe", "partie"
        ]
        # Question type hints
        self.question_words = {
            "simple": ["qui", "quand", "où", "quel", "quelle", "combien"],
            "moderate": ["quoi", "que", "qu'est-ce", "quel est", "quelle est"],
            "complex": ["comment", "pourquoi", "dans quelles conditions", "en cas de"]
        }

        # Cached LLM instance (lazy init)
        self._llm = None

    def _get_llm(self) -> Optional[GenerativeModel]:
        """Lazy initialise the Gemini model for LLM analysis."""
        if self.mode != "llm":
            return None
        if self._llm is None:
            try:
                self._llm = GenerativeModel(self.llm_model_name)
            except Exception as e:
                logger.error(f"Failed to initialise LLM for query analysis: {e}")
                # Fall back to heuristic
                self.mode = "heuristic"
        return self._llm

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query and return complexity and suggested parameters.

        Returns:
            dict with keys:
                complexity (str): 'low', 'medium', 'high'
                suggested_top_k (int)
                suggested_expand (bool)
                suggested_rerank (bool)
                raw_score (float): heuristic score (only in heuristic mode)
                word_count (int)
                legal_count (int)
                has_numbers (bool)
                question_type (str)
        """
        if not self.enabled:
            return self._default_analysis()

        if self.mode == "llm":
            try:
                return self._analyze_llm(query)
            except Exception as e:
                logger.warning(f"LLM analysis failed, falling back to heuristic: {e}")
                # Fall back to heuristic
                return self._analyze_heuristic(query)

        return self._analyze_heuristic(query)

    def _default_analysis(self) -> Dict[str, Any]:
        """Return neutral defaults when analyzer is disabled."""
        return {
            "complexity": "medium",
            "suggested_top_k": 5,
            "suggested_expand": False,
            "suggested_rerank": True,
            "raw_score": 0.0,
            "word_count": 0,
            "legal_count": 0,
            "has_numbers": False,
            "question_type": "unknown"
        }

    def _analyze_heuristic(self, query: str) -> Dict[str, Any]:
        """
        Fast rule‑based analysis.
        """
        words = re.findall(r'\b\w+\b', query.lower())
        word_count = len(words)

        # Legal keyword count
        legal_count = sum(1 for w in words if w in self.legal_keywords)

        # Numbers
        has_numbers = bool(re.search(r'\d+', query))

        # Question type
        qtype = "moderate"
        lower_q = query.lower()
        for q_type, terms in self.question_words.items():
            if any(term in lower_q for term in terms):
                qtype = q_type
                break

        # Complexity score (0..4)
        score = 0.0
        if word_count > 5:
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

        # Map score to complexity
        if score <= 1.0:
            complexity = "low"
        elif score <= 2.5:
            complexity = "medium"
        else:
            complexity = "high"

        # Map to suggestions
        top_k_map = {
            "low": settings.QUERY_ANALYZER_TOP_K_LOW,
            "medium": settings.QUERY_ANALYZER_TOP_K_MEDIUM,
            "high": settings.QUERY_ANALYZER_TOP_K_HIGH
        }
        expand_map = settings.QUERY_ANALYZER_EXPAND_MAP
        rerank_map = settings.QUERY_ANALYZER_RERANK_MAP

        return {
            "complexity": complexity,
            "suggested_top_k": top_k_map[complexity],
            "suggested_expand": expand_map[complexity],
            "suggested_rerank": rerank_map[complexity],
            "raw_score": score,
            "word_count": word_count,
            "legal_count": legal_count,
            "has_numbers": has_numbers,
            "question_type": qtype
        }

    def _analyze_llm(self, query: str) -> Dict[str, Any]:
        """
        Use Gemini to analyse query complexity and suggest parameters.
        Falls back to heuristic if LLM call fails or returns invalid data.
        """
        llm = self._get_llm()
        if llm is None:
            return self._analyze_heuristic(query)

        prompt = f"""
Tu es un expert en analyse de requêtes juridiques pour un système de recherche.
Analyse la requête suivante et retourne un JSON valide avec les champs :
- "complexity": "low", "medium" ou "high"
- "suggested_top_k": un entier entre 1 et 10
- "suggested_expand": boolean (true/false) – indique si la requête bénéficierait d'une expansion sémantique
- "suggested_rerank": boolean (true/false) – recommande l'utilisation d'un reranker (pour la précision)

Critères :
- low: question factuelle simple, peu de termes juridiques, pas de condition.
- medium: question qui combine plusieurs concepts, quelques termes juridiques.
- high: question complexe avec conditions, références croisées, nombreux termes juridiques.

Requête : {query}

Réponds uniquement avec le JSON, sans autre texte.
"""
        try:
            response = llm.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            text = response.text.strip()
            # Extract JSON from markdown if present
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text)
            # Validate and fill defaults
            return {
                "complexity": data.get("complexity", "medium"),
                "suggested_top_k": data.get("suggested_top_k", 5),
                "suggested_expand": data.get("suggested_expand", False),
                "suggested_rerank": data.get("suggested_rerank", True),
                "raw_score": None,
                "word_count": None,
                "legal_count": None,
                "has_numbers": None,
                "question_type": None
            }
        except Exception as e:
            logger.warning(f"LLM analysis parsing failed: {e}, falling back to heuristic")
            return self._analyze_heuristic(query)