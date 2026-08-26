"""
Layer 4: Semantic Enrichment & NLP
Adds entities, clause classification, and semantic metadata.
Integrated with caching and configurable NLP models.
"""
import logging
import hashlib
import json
import os
import re
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from filelock import FileLock

# Optional imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from google.cloud import aiplatform
    from vertexai.language_models import TextEmbeddingModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

from src.config.settings import settings

logger = logging.getLogger(__name__)

FRENCH_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}\s+(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
    r"septembre|octobre|novembre|d[ée]cembre)\s+\d{4}|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
    re.IGNORECASE,
)
FRENCH_AMOUNT_PATTERN = re.compile(
    r"\b(\d{1,3}(?:[ \u202f.]\d{3})*(?:,\d{2})?\s?(?:€|EUR|euros?)|"
    r"(?:€|EUR)\s?\d{1,3}(?:[ \u202f.]\d{3})*(?:,\d{2})?)\b",
    re.IGNORECASE,
)


class SemanticEnricher:
    """
    Enriches parsed document with:
    - Named entities (organizations, dates, amounts, persons, locations)
    - Clause type classification (obligation, payment, guarantee, termination, etc.)
    - Section embeddings for RAG (using Vertex AI)

    Caches entities and embeddings per document hash (derived from text + structure).
    """

    def __init__(
        self,
        language: str = "fr",
        cache_dir: Optional[str] = None,
        embedding_batch_size: int = 10,
        entity_chunk_size: int = 100000,  # characters per chunk
        clause_min_length: int = 50,
        use_vertex_embeddings: bool = True,
        use_spacy_ner: bool = True,
    ):
        """
        Args:
            language: Language code for spaCy ('fr' or 'en').
            cache_dir: Directory to store semantic enrichment cache.
            embedding_batch_size: Number of texts to embed per batch (Vertex AI).
            entity_chunk_size: Max characters per chunk for spaCy NER (to avoid memory issues).
            clause_min_length: Minimum text length to consider as a clause.
            use_vertex_embeddings: Whether to generate embeddings (requires Vertex AI).
            use_spacy_ner: Whether to run spaCy NER.
        """
        self.language = language
        self.cache_dir = cache_dir
        self.embedding_batch_size = embedding_batch_size
        self.entity_chunk_size = entity_chunk_size
        self.clause_min_length = clause_min_length
        self.use_vertex_embeddings = use_vertex_embeddings
        self.use_spacy_ner = use_spacy_ner

        # Initialize cache
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "ocr_cache.json")
            self.cache_lock_file = self.cache_file + ".lock"
            self._ensure_cache_file()

        # Initialize spaCy
        self.nlp = None
        if self.use_spacy_ner and SPACY_AVAILABLE:
            try:
                model_name = f"{language}_core_news_sm" if language == "fr" else "en_core_web_sm"
                self.nlp = spacy.load(model_name)
                # Keep only the pipes NER actually depends on. `ner` listens to
                # `tok2vec`'s output via a shared listener in these pipelines --
                # disabling tok2vec alongside it can break or silently degrade NER.
                # select_pipes(enable=...) is the safe way to do this: it disables
                # everything except what you list, without touching listener wiring.
                keep = [p for p in ("tok2vec", "ner") if self.nlp.has_pipe(p)]
                self.nlp.select_pipes(enable=keep)
                logger.info(f"Loaded spaCy model: {model_name} (active pipes: {keep})")
            except OSError:
                logger.warning(f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}")
                self.nlp = None
        elif not SPACY_AVAILABLE:
            logger.warning("spaCy not installed – NER disabled")
        else:
            logger.info("spaCy NER disabled")

        # Initialize Vertex AI embeddings
        self.embedding_model = None
        if self.use_vertex_embeddings and VERTEX_AI_AVAILABLE:
            try:
                aiplatform.init(
                    project=settings.GCP_PROJECT_ID,
                    location=settings.GCP_LOCATION
                )
                model_name = getattr(settings, "VERTEX_AI_EMBEDDING_MODEL", "textembedding-gecko@003")
                self.embedding_model = TextEmbeddingModel.from_pretrained(model_name)
                logger.info(f"Initialized Vertex AI embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Vertex AI embedding initialization failed: {e}")
                self.embedding_model = None
        elif not VERTEX_AI_AVAILABLE:
            logger.warning("Vertex AI SDK not installed – embeddings disabled")

        # Clause classifier – can be a simple keyword classifier or a Vertex AI endpoint
        self.clause_classifier = None
        self._init_clause_classifier()

    def _init_clause_classifier(self):
        """Initialize the clause classifier (keyword by default, or custom Vertex endpoint)."""
        # By default, use keyword-based classification.
        # You can override by setting a custom classifier function.
        self.clause_classifier = self._keyword_classifier

        # If you have a Vertex AI endpoint for clause classification, uncomment:
        # from google.cloud import aiplatform
        # self.clause_classifier = self._vertex_ai_clause_classifier

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------

    def _ensure_cache_file(self) -> None:
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w") as f:
                json.dump({}, f)

    def _get_content_hash(self, structure: Dict, raw_text: str) -> str:
        """Hash based on raw text and structure (section IDs and headings)."""
        # Use a subset of structure to avoid too much variation
        structure_sig = json.dumps({
            "sections": self._extract_structure_signature(structure),
            "text_preview": raw_text[:5000]
        }, sort_keys=True)
        return hashlib.sha256(structure_sig.encode("utf-8")).hexdigest()

    def _extract_structure_signature(self, structure: Dict) -> List[Dict]:
        """Extract section IDs and headings for hash stability."""
        sig = []
        def traverse(node):
            if node.get("section_id"):
                sig.append({
                    "id": node["section_id"],
                    "heading": node.get("heading", ""),
                })
            for child in node.get("children", []):
                traverse(child)
        if structure and "root" in structure:
            traverse(structure["root"])
        return sig

    def _load_from_cache(self, content_hash: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        try:
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
            if content_hash in cache:
                logger.info(f"Semantic cache hit for hash {content_hash[:8]}...")
                return cache[content_hash]
            return None
        except Exception as e:
            logger.warning(f"Failed to read semantic cache: {e}")
            return None

    def _save_to_cache(self, pdf_hash: str, result: Dict[str, Any]) -> None:
        """Save the extraction result to the cache. Locked to survive concurrent requests
        on the same Cloud Run instance -- without this, two simultaneous parses can
        interleave read-modify-write cycles and corrupt/drop cache entries."""
        if not self.cache_dir:
            return
        try:
            with FileLock(self.cache_lock_file, timeout=10):
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                cache[pdf_hash] = result
                with open(self.cache_file, "w") as f:
                    json.dump(cache, f, indent=2)
            logger.info(f"Cached result for PDF hash {pdf_hash[:8]}...")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to write cache: {e}")
        except TimeoutError:
            logger.warning(f"Cache lock timeout for hash {pdf_hash[:8]} – skipping cache write")

    # -------------------------------------------------------------------------
    # Main enrichment entry point
    # -------------------------------------------------------------------------

    def enrich(
        self,
        structure: Dict,
        raw_text: str,
        elements: Dict,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply semantic enrichment to the document.

        Args:
            structure: The hierarchical structure from Layer 2.
            raw_text: Full raw text from Layer 1.
            elements: Extracted tables/figures from Layer 3 (optional).
            force_reprocess: Ignore cache and recompute.

        Returns:
            Dict with keys:
            - structure: enriched structure (with entities, embeddings, clause types)
            - entities: global entities dict
            - clauses: list of clause dicts
            - warnings: list of warnings
            - error: optional error message
        """
        result = {
            "structure": structure,
            "entities": {},
            "clauses": [],
            "warnings": [],
            "error": None,
        }

        # Compute cache key
        content_hash = self._get_content_hash(structure, raw_text)

        if not force_reprocess:
            cached = self._load_from_cache(content_hash)
            if cached:
                # Restore entities and clauses, and inject them into structure
                result["entities"] = cached.get("entities", {})
                result["clauses"] = cached.get("clauses", [])
                # Re-inject clause types and embeddings into structure (if present)
                self._inject_cached_data_into_structure(structure, cached)
                result["structure"] = structure
                logger.info(f"Loaded semantic enrichment from cache: {len(result['clauses'])} clauses")
                return result

        try:
            # 1. Entity extraction
            entities = self._extract_entities(raw_text) if raw_text else {}
            result["entities"] = entities
            logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")

            # 1b. Also run entity extraction over table cell text so figures like
            # payment amounts in tables aren't invisible to the entity index.
            table_text = self._flatten_table_text(elements.get("tables", []))
            if table_text:
                table_entities = self._extract_entities(table_text)
                for key in entities:
                    entities[key].extend(table_entities.get(key, []))
                entities = self._dedupe_entities(entities)
                result["entities"] = entities


            # 2. Clause extraction and classification
            clauses = self._extract_clauses(structure)
            if clauses:
                result["clauses"] = clauses
                logger.info(f"Extracted {len(clauses)} clauses")
                # Inject clause types into structure
                self._inject_clause_into_structure(structure, clauses)

            # 3. Embeddings for sections (if enabled)
            if self.embedding_model:
                self._add_embeddings(structure)

            # 4. Cache the enriched data (without storing the full structure to save space)
            cache_data = {
                "entities": entities,
                "clauses": clauses,
                # Store only section-level data for injection on cache load
                "section_enrichments": self._extract_section_enrichments(structure),
            }
            self._save_to_cache(content_hash, cache_data)

            result["structure"] = structure

        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    # -------------------------------------------------------------------------
    # Entity Extraction
    # -------------------------------------------------------------------------

    def _extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using spaCy (PER/ORG/LOC) plus regex for dates/amounts,
        since the French spaCy models don't have DATE/MONEY labels."""
        entities = {
            "organizations": [],
            "persons": [],
            "dates": [],
            "amounts": [],
            "locations": [],
        }

        # Regex-based dates and amounts run regardless of spaCy availability.
        for match in FRENCH_DATE_PATTERN.finditer(text):
            entities["dates"].append({"text": match.group(0), "type": "date"})
        for match in FRENCH_AMOUNT_PATTERN.finditer(text):
            entities["amounts"].append({"text": match.group(0), "type": "amount"})

        if not self.nlp:
            entities = self._dedupe_entities(entities)
            return entities

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
                    # Note: no date/money elif here -- fr_core_news_* models don't emit
                    # these labels, so any date/money handling belongs to the regex pass above.
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

    def _flatten_table_text(self, tables: List[Dict]) -> str:
        """Flatten table cell values into plain text so entity extraction can see them."""
        parts = []
        for table in tables:
            for row in table.get("data", []):
                if isinstance(row, dict):
                    parts.extend(str(v) for v in row.values() if v)
                elif isinstance(row, list):
                    parts.extend(str(v) for v in row if v)
        return " ".join(parts)

    # -------------------------------------------------------------------------
    # Clause Extraction and Classification
    # -------------------------------------------------------------------------

    def _extract_clauses(self, structure: Dict) -> List[Dict]:
        """Traverse structure and extract nodes that are paragraphs/subparagraphs."""
        clauses = []

        def traverse(node):
            section_type = node.get("section_type", "")
            text = node.get("text", "")
            if section_type in ("paragraph", "subparagraph") and text and len(text) >= self.clause_min_length:
                # Classify the clause
                clause_type = self.clause_classifier(text) if callable(self.clause_classifier) else "general"
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
        """Simple keyword-based clause classification."""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["obligation", "s'engage", "doit", "devra"]):
            return "obligation"
        if any(kw in text_lower for kw in ["paiement", "payer", "facture", "tarif", "prix"]):
            return "payment_term"
        if any(kw in text_lower for kw in ["garantie", "garentie", "caution"]):
            return "guarantee"
        if any(kw in text_lower for kw in ["résiliation", "terminaison", "annulation", "fin"]):
            return "termination"
        if any(kw in text_lower for kw in ["conformité", "conforme", "norme", "réglement"]):
            return "compliance"
        if any(kw in text_lower for kw in ["durée", "période", "délai", "échéance"]):
            return "duration"
        if any(kw in text_lower for kw in ["responsabilité", "responsable"]):
            return "liability"
        if any(kw in text_lower for kw in ["droit", "autoriser", "permission"]):
            return "right"
        return "general"

    def _vertex_ai_clause_classifier(self, text: str) -> str:
        """Placeholder for calling a Vertex AI endpoint for clause classification."""
        # In production, you'd call a deployed model endpoint.
        # For now, fall back to keyword.
        return self._keyword_classifier(text)

    # -------------------------------------------------------------------------
    # Inject enrichment data into structure
    # -------------------------------------------------------------------------

    def _inject_clause_into_structure(self, structure: Dict, clauses: List[Dict]):
        """Add clause_type and clause list to each corresponding node."""
        clause_map = {c["section_id"]: c for c in clauses}

        def traverse(node):
            sec_id = node.get("section_id")
            if sec_id and sec_id in clause_map:
                node["clause_type"] = clause_map[sec_id]["type"]
                # Add a clauses array (though usually only one per node)
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

    def _inject_cached_data_into_structure(self, structure: Dict, cache_data: Dict):
        """Re‑inject clause types and embeddings from cached data."""
        # Re‑inject clause types
        clauses = cache_data.get("clauses", [])
        self._inject_clause_into_structure(structure, clauses)

        # Re‑inject embeddings
        if "section_enrichments" in cache_data:
            enrichments = {item["section_id"]: item for item in cache_data["section_enrichments"]}

            def traverse(node):
                sec_id = node.get("section_id")
                if sec_id and sec_id in enrichments:
                    node["embedding"] = enrichments[sec_id].get("embedding")
                for child in node.get("children", []):
                    traverse(child)

            if structure and "root" in structure:
                traverse(structure["root"])

    def _extract_section_enrichments(self, structure: Dict) -> List[Dict]:
        """Extract section enrichments (embeddings, clause types) for caching."""
        enrichments = []

        def traverse(node):
            sec_id = node.get("section_id")
            if sec_id:
                enrich = {"section_id": sec_id}
                if "embedding" in node:
                    enrich["embedding"] = node["embedding"]
                if "clause_type" in node:
                    enrich["clause_type"] = node["clause_type"]
                if "clauses" in node:
                    enrich["clauses"] = node["clauses"]
                if enrich != {"section_id": sec_id}:
                    enrichments.append(enrich)
            for child in node.get("children", []):
                traverse(child)

        if structure and "root" in structure:
            traverse(structure["root"])
        return enrichments

    # -------------------------------------------------------------------------
    # Embeddings Generation
    # -------------------------------------------------------------------------

    def _add_embeddings(self, structure: Dict):
        """Generate embeddings for sections with sufficient text, batched."""
        if not self.embedding_model:
            return

        # Collect texts to embed with their node references
        nodes_to_embed = []

        def traverse(node):
            text = node.get("text", "")
            if text and len(text) >= 50:
                # Truncate to model's max token limit (approx 2048 chars for gecko)
                truncated = text[:2048]
                nodes_to_embed.append((node, truncated))
            for child in node.get("children", []):
                traverse(child)

        if structure and "root" in structure:
            traverse(structure["root"])

        if not nodes_to_embed:
            return

        # Batch embeddings
        total = len(nodes_to_embed)
        for i in range(0, total, self.embedding_batch_size):
            batch = nodes_to_embed[i:i+self.embedding_batch_size]
            texts = [item[1] for item in batch]
            try:
                # Generate embeddings in a single call
                embeddings = self.embedding_model.get_embeddings(texts)
                for (node, _), emb_obj in zip(batch, embeddings):
                    # Check if embedding object has values attribute
                    if hasattr(emb_obj, "values"):
                        node["embedding"] = emb_obj.values
                    else:
                        node["embedding"] = emb_obj  # fallback
                logger.info(f"Generated embeddings for {len(batch)} sections (batch {i//self.embedding_batch_size + 1})")
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
                # Try individual embeddings on failure
                for node, text in batch:
                    try:
                        emb = self.embedding_model.get_embeddings([text])[0]
                        node["embedding"] = emb.values if hasattr(emb, "values") else emb
                    except Exception as e2:
                        logger.warning(f"Failed to embed section {node.get('section_id')}: {e2}")