"""
Layer 4: Semantic Enrichment & NLP (English & CUAD Contracts)
Extracts named entities (parties/organizations, persons, locations, dates, monetary amounts)
and classifies legal clauses using spaCy (en_core_web_sm) and legal regex heuristics.
"""
import logging
import hashlib
import json
import os
import re
from typing import Optional, Dict, Any, List
from filelock import FileLock

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from src.config.settings import settings

logger = logging.getLogger(__name__)

# English Date & Amount regexes
ENGLISH_DATE_PATTERN = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}|"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|"
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
ENGLISH_AMOUNT_PATTERN = re.compile(
    r"\b(\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|dollars?|EUR|€|GBP|£))\b",
    re.IGNORECASE,
)


class SemanticEnricher:
    """
    Enriches parsed legal documents with:
    - Named entities (organizations/parties, dates, amounts, persons, locations)
    - Clause type classification based on legal contract taxonomies (CUAD categories).
    """

    def __init__(
        self,
        language: str = "en",
        cache_dir: Optional[str] = None,
        entity_chunk_size: int = 100000,
        clause_min_length: int = 40,
        use_spacy_ner: bool = True,
        **kwargs
    ):
        self.language = language
        self.cache_dir = cache_dir
        self.entity_chunk_size = entity_chunk_size
        self.clause_min_length = clause_min_length
        self.use_spacy_ner = use_spacy_ner

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "ocr_cache.json")
            self.cache_lock_file = self.cache_file + ".lock"
            self._ensure_cache_file()

        # Initialize spaCy English model
        self.nlp = None
        if self.use_spacy_ner and SPACY_AVAILABLE:
            try:
                model_name = f"{language}_core_web_sm" if language == "en" else f"{language}_core_news_sm"
                self.nlp = spacy.load(model_name)
                keep = [p for p in ("tok2vec", "ner") if self.nlp.has_pipe(p)]
                self.nlp.select_pipes(enable=keep)
                logger.info(f"Loaded spaCy model: {model_name} (active pipes: {keep})")
            except OSError:
                logger.warning(f"spaCy model not found. Run: python -m spacy download {language}_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None

        self.clause_classifier = self._keyword_classifier

    def _ensure_cache_file() -> None:
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _get_content_hash(self, structure: Dict, raw_text: str) -> str:
        structure_sig = json.dumps({
            "text_preview": raw_text[:5000]
        }, sort_keys=True)
        return hashlib.sha256(structure_sig.encode("utf-8")).hexdigest()

    def _load_from_cache(self, content_hash: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            return cache.get(content_hash)
        except Exception:
            return None

    def _save_to_cache(self, content_hash: str, result: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        try:
            with FileLock(self.cache_lock_file, timeout=10):
                cache = {}
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        cache = json.load(f)
                cache[content_hash] = result
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write semantic cache: {e}")

    def enrich(
        self,
        structure: Dict,
        raw_text: str,
        elements: Dict,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        result = {
            "structure": structure,
            "entities": {},
            "clauses": [],
            "warnings": [],
            "error": None,
        }

        content_hash = self._get_content_hash(structure, raw_text)

        if not force_reprocess:
            cached = self._load_from_cache(content_hash)
            if cached:
                result["entities"] = cached.get("entities", {})
                result["clauses"] = cached.get("clauses", [])
                self._inject_clause_into_structure(structure, result["clauses"])
                result["structure"] = structure
                return result

        try:
            # 1. Extract Entities
            entities = self._extract_entities(raw_text)
            result["entities"] = entities

            # 2. Extract and classify clauses
            clauses = self._extract_clauses(structure)
            result["clauses"] = clauses
            self._inject_clause_into_structure(structure, clauses)

            # Save to cache
            self._save_to_cache(content_hash, {
                "entities": entities,
                "clauses": clauses,
            })

        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def _extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        entities = {
            "organizations": [],
            "persons": [],
            "dates": [],
            "amounts": [],
            "locations": [],
        }

        for match in ENGLISH_DATE_PATTERN.finditer(text):
            entities["dates"].append({"text": match.group(0), "type": "date"})
        for match in ENGLISH_AMOUNT_PATTERN.finditer(text):
            entities["amounts"].append({"text": match.group(0), "type": "amount"})

        if not self.nlp:
            return self._dedupe_entities(entities)

        chunks = [text[i:i+self.entity_chunk_size] for i in range(0, len(text), self.entity_chunk_size)]
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                doc = self.nlp(chunk)
                for ent in doc.ents:
                    label = ent.label_.lower()
                    if label in ("org", "organization"):
                        entities["organizations"].append({"text": ent.text, "type": "organization"})
                    elif label in ("person", "per"):
                        entities["persons"].append({"text": ent.text, "type": "person"})
                    elif label in ("loc", "location", "gpe"):
                        entities["locations"].append({"text": ent.text, "type": "location"})
                    elif label in ("money", "amount") and not any(e["text"] == ent.text for e in entities["amounts"]):
                        entities["amounts"].append({"text": ent.text, "type": "amount"})
                    elif label == "date" and not any(e["text"] == ent.text for e in entities["dates"]):
                        entities["dates"].append({"text": ent.text, "type": "date"})
            except Exception as e:
                logger.warning(f"Entity extraction chunk failed: {e}")

        return self._dedupe_entities(entities)

    def _dedupe_entities(self, entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        for key in entities:
            seen = set()
            unique = []
            for e in entities[key]:
                if e["text"] not in seen:
                    seen.add(e["text"])
                    unique.append(e)
            entities[key] = unique
        return entities

    def _extract_clauses(self, structure: Dict) -> List[Dict]:
        clauses = []

        def traverse(node):
            section_type = node.get("section_type", "")
            text = node.get("text", "")
            if section_type in ("paragraph", "subparagraph", "clause", "section") and text and len(text) >= self.clause_min_length:
                clause_type = self.clause_classifier(text)
                clauses.append({
                    "section_id": node.get("section_id"),
                    "text": text,
                    "heading": node.get("heading", ""),
                    "type": clause_type,
                })
            for child in node.get("children", []):
                traverse(child)

        if structure and "root" in structure:
            traverse(structure["root"])

        return clauses

    def _keyword_classifier(self, text: str) -> str:
        """
        Classifies legal text into core CUAD / contract categories.
        """
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["indemnif", "hold harmless", "defend and hold"]):
            return "indemnification"
        if any(kw in text_lower for kw in ["terminat", "right to cancel", "expiration of term"]):
            return "termination"
        if any(kw in text_lower for kw in ["governing law", "jurisdiction", "venue", "laws of the state"]):
            return "governing_law"
        if any(kw in text_lower for kw in ["confidential", "non-disclosure", "proprietary information"]):
            return "confidentiality"
        if any(kw in text_lower for kw in ["intellectual property", "patent", "copyright", "trademark", "work made for hire"]):
            return "ip_assignment"
        if any(kw in text_lower for kw in ["non-compete", "non compete", "covenant not to compete", "restrictive covenant"]):
            return "non_compete"
        if any(kw in text_lower for kw in ["non-solicit", "non solicit", "solicitation of employees"]):
            return "non_solicitation"
        if any(kw in text_lower for kw in ["payment", "fee", "compensation", "invoice", "royalty", "reimbursement", "price"]):
            return "payment_term"
        if any(kw in text_lower for kw in ["term of agreement", "effective date", "renewal", "duration"]):
            return "duration"
        if any(kw in text_lower for kw in ["limitation of liability", "consequential damages", "cap on liability"]):
            return "limitation_of_liability"
        if any(kw in text_lower for kw in ["warranty", "warranties", "representation and warranty", "as is"]):
            return "warranties"
        if any(kw in text_lower for kw in ["force majeure", "acts of god", "natural disaster"]):
            return "force_majeure"
        if any(kw in text_lower for kw in ["audit right", "inspection of books", "books and records"]):
            return "audit_rights"
        if any(kw in text_lower for kw in ["assignment", "assignability", "successor and assign"]):
            return "assignment"
        if any(kw in text_lower for kw in ["shall", "must", "agrees to", "covenants to"]):
            return "obligation"
        return "general"

    def _inject_clause_into_structure(self, structure: Dict, clauses: List[Dict]):
        clause_map = {c["section_id"]: c for c in clauses}

        def traverse(node):
            sec_id = node.get("section_id")
            if sec_id and sec_id in clause_map:
                node["clause_type"] = clause_map[sec_id]["type"]
                if "clauses" not in node:
                    node["clauses"] = []
                node["clauses"].append({
                    "text": clause_map[sec_id]["text"],
                    "type": clause_map[sec_id]["type"],
                })
            for child in node.get("children", []):
                traverse(child)

        if structure and "root" in structure:
            traverse(structure["root"])