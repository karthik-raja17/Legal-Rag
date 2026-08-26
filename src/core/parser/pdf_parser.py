"""
Main PDF Parser Orchestrator
Coordinates all parsing layers for legal contract extraction.
Production‑grade with caching, error handling, and configurable components.
"""
import logging
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

from src.core.parser.text_extractor import TextExtractor
from src.core.parser.structure_extractor import StructureExtractor
from src.core.parser.element_extractor import ElementExtractor
from src.core.parser.semantic_enricher import SemanticEnricher

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Container for fully parsed document output."""
    document_id: str
    metadata: Dict[str, Any]
    structure: Dict[str, Any]          # The hierarchical document tree
    elements: Dict[str, List[Dict]]    # Tables, figures, clauses, entities
    raw_text: str
    processing_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ocr_used: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary (JSON serializable)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class PDFParser:
    """
    Orchestrates the complete parsing pipeline for legal contracts.
    Manages fallback paths and error handling across all layers.
    Supports caching and configurable components.
    """

    def __init__(
        self,
        use_ocr: bool = True,
        use_dedoc: bool = True,
        extract_tables: bool = True,
        extract_figures: bool = True,
        semantic_enrichment: bool = True,
        language: str = "fr",
        cache_dir: Optional[str] = None,
        ocr_cache_dir: Optional[str] = None,
        structure_cache_dir: Optional[str] = None,
        element_cache_dir: Optional[str] = None,
        semantic_cache_dir: Optional[str] = None,
        dedoc_url: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            use_ocr: Enable Google Document AI OCR fallback for scanned PDFs
            use_dedoc: Enable Dedoc for structure extraction
            extract_tables: Enable table extraction
            extract_figures: Enable figure/graph extraction
            semantic_enrichment: Enable NLP + Vertex AI enrichment
            language: Document language (fr/en)
            cache_dir: Top-level cache directory (creates subdirectories per layer)
            ocr_cache_dir: Cache directory for OCR (overrides cache_dir)
            structure_cache_dir: Cache directory for structure (overrides cache_dir)
            element_cache_dir: Cache directory for elements (overrides cache_dir)
            semantic_cache_dir: Cache directory for semantic enrichment (overrides cache_dir)
            **kwargs: Additional parameters passed to individual extractors
        """
        self.use_ocr = use_ocr
        self.use_dedoc = use_dedoc
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures
        self.semantic_enrichment = semantic_enrichment
        self.language = language

        # Determine cache directories
        if cache_dir:
            ocr_cache_dir = ocr_cache_dir or f"{cache_dir}/ocr"
            structure_cache_dir = structure_cache_dir or f"{cache_dir}/structure"
            element_cache_dir = element_cache_dir or f"{cache_dir}/elements"
            semantic_cache_dir = semantic_cache_dir or f"{cache_dir}/semantic"

        # Initialize layer components
        self.text_extractor = TextExtractor(
            use_ocr=use_ocr,
            cache_dir=ocr_cache_dir,
            **{k: v for k, v in kwargs.items() if k in ['ocr_threshold_chars_per_page', 'always_use_ocr']}
        )

        # In pdf_parser.py, inside __init__, after setting self.structure_extractor:
        logger.info(f"StructureExtractor initialized with dedoc_url: {dedoc_url}")
        self.structure_extractor = StructureExtractor(
            use_dedoc=use_dedoc,
            language=language,
            cache_dir=structure_cache_dir,
            dedoc_url=dedoc_url,
            **{k: v for k, v in kwargs.items() if k in ['dedoc_parameters', 'dedoc_timeout']}
        )

        self.element_extractor = ElementExtractor(
            extract_tables=extract_tables,
            extract_figures=extract_figures,
            cache_dir=element_cache_dir,
            **{k: v for k, v in kwargs.items() if k in ['table_methods', 'max_pages_for_ocr']}
        )

        self.semantic_enricher = SemanticEnricher(
            language=language,
            cache_dir=semantic_cache_dir,
            **{k: v for k, v in kwargs.items()
               if k in ['embedding_batch_size', 'entity_chunk_size', 'clause_min_length',
                        'use_vertex_embeddings', 'use_spacy_ner']}
        ) if semantic_enrichment else None

        # Validation: warn if required libraries missing
        if use_dedoc and not self.structure_extractor.use_dedoc:
            logger.warning("Dedoc requested but not available – structure extraction will use regex fallback")

        if use_ocr and not self.text_extractor.use_ocr:
            logger.warning("OCR requested but not available – only PyMuPDF will be used")

    def parse(self, pdf_content: bytes, document_id: str = "unknown") -> ParsedDocument:
        """
        Main entry point for parsing a PDF document.
        
        Args:
            pdf_content: Raw PDF bytes
            document_id: Unique identifier for the document
            
        Returns:
            ParsedDocument with all extracted information
        """
        start_time = time.time()
        errors = []
        warnings = []

        logger.info(f"Starting parsing for document: {document_id}")

        # Initialize intermediate results
        raw_text = ""
        ocr_used = False
        structure = {"root": None}
        elements = {"tables": [], "figures": [], "entities": {}, "clauses": []}
        metadata = {"parsed_at": datetime.utcnow().isoformat(), "language": self.language}

        # ==================== LAYER 1: TEXT EXTRACTION ====================
        try:
            logger.info(f"Layer 1: Extracting text for {document_id}...")
            text_result = self.text_extractor.extract(pdf_content, document_id=document_id)
            
            if text_result.get("error"):
                errors.append(f"Text extraction error: {text_result['error']}")
            else:
                raw_text = text_result.get("text", "")
                ocr_used = text_result.get("ocr_used", False)
                if ocr_used:
                    warnings.append("OCR was used – text may contain recognition errors")
                
            if not raw_text:
                warnings.append("No text could be extracted from the document")
                # We still continue to get whatever structure might be possible
                
        except Exception as e:
            errors.append(f"Text extraction critical failure: {str(e)}")
            logger.error(f"Layer 1 critical error: {e}", exc_info=True)

        # ==================== LAYER 2: STRUCTURE EXTRACTION ====================
        try:
            logger.info("Layer 2: Extracting document structure...")
            # Prepare layer1_result for structure extractor (it expects the full dict)
            layer1_result = {
                "text": raw_text,
                "pages": text_result.get("pages", []) if 'text_result' in locals() else [],
                "ocr_used": ocr_used,
                "error": text_result.get("error") if 'text_result' in locals() else None,
            }
            structure_result = self.structure_extractor.extract(
                pdf_content=pdf_content,
                layer1_result=layer1_result,
                force_reprocess=False
            )
            
            if structure_result.get("error"):
                errors.append(f"Structure extraction error: {structure_result['error']}")
                # Use fallback flat structure
                structure = self.structure_extractor._create_flat_structure(raw_text)
            else:
                structure = structure_result.get("structure", {"root": None})
                if not structure.get("root"):
                    structure = self.structure_extractor._create_flat_structure(raw_text)
                    warnings.append("Structure extraction returned empty – using flat structure")
            
            if structure_result.get("warnings"):
                warnings.extend(structure_result["warnings"])
                
        except Exception as e:
            errors.append(f"Structure extraction critical failure: {str(e)}")
            logger.error(f"Layer 2 critical error: {e}", exc_info=True)
            # Fallback to flat structure
            try:
                structure = self.structure_extractor._create_flat_structure(raw_text)
            except Exception:
                structure = {"root": None}

        # ==================== LAYER 3: ELEMENT EXTRACTION ====================
        try:
            logger.info("Layer 3: Extracting tables and figures...")
            # Pass layer1_result for better page-level context
            elements_result = self.element_extractor.extract(
                pdf_content=pdf_content,
                structure=structure,
                layer1_result=layer1_result if 'layer1_result' in locals() else None,
                force_reprocess=False
            )
            
            elements["tables"] = elements_result.get("tables", [])
            elements["figures"] = elements_result.get("figures", [])
            
            if elements_result.get("warnings"):
                warnings.extend(elements_result["warnings"])
            if elements_result.get("errors"):
                errors.extend(elements_result["errors"])
                
        except Exception as e:
            errors.append(f"Element extraction critical failure: {str(e)}")
            logger.error(f"Layer 3 critical error: {e}", exc_info=True)

        # ==================== LAYER 4: SEMANTIC ENRICHMENT ====================
        enriched_structure = structure
        if self.semantic_enricher:
            try:
                logger.info("Layer 4: Applying semantic enrichment...")
                enrichment_result = self.semantic_enricher.enrich(
                    structure=structure,
                    raw_text=raw_text,
                    elements=elements,
                    force_reprocess=False
                )
                
                enriched_structure = enrichment_result.get("structure", structure)
                if enrichment_result.get("error"):
                    errors.append(f"Semantic enrichment error: {enrichment_result['error']}")
                
                # Add extracted entities and clauses to elements
                elements["entities"] = enrichment_result.get("entities", {})
                elements["clauses"] = enrichment_result.get("clauses", [])
                
                if enrichment_result.get("warnings"):
                    warnings.extend(enrichment_result["warnings"])
                    
            except Exception as e:
                errors.append(f"Semantic enrichment critical failure: {str(e)}")
                logger.error(f"Layer 4 critical error: {e}", exc_info=True)

        # ==================== BUILD FINAL RESULT ====================
        processing_time = time.time() - start_time
        
        # Extract metadata from structure
        metadata.update(self._extract_metadata(enriched_structure, raw_text))
        metadata["ocr_used"] = ocr_used
        metadata["word_count"] = len(raw_text.split())
        metadata["char_count"] = len(raw_text)
        
        # Add counts to metadata
        metadata["table_count"] = len(elements.get("tables", []))
        metadata["figure_count"] = len(elements.get("figures", []))
        metadata["clause_count"] = len(elements.get("clauses", []))
        metadata["entity_count"] = sum(len(v) for v in elements.get("entities", {}).values())

        return ParsedDocument(
            document_id=document_id,
            metadata=metadata,
            structure=enriched_structure,
            elements=elements,
            raw_text=raw_text,
            processing_time=processing_time,
            errors=errors,
            warnings=warnings,
            ocr_used=ocr_used
        )

    def _extract_metadata(self, structure: Dict, raw_text: str) -> Dict[str, Any]:
        """Extract basic metadata from the parsed structure."""
        metadata = {}
        
        # Try to extract from structure if available
        if structure and "root" in structure:
            root = structure.get("root", {})
            if root.get("heading"):
                metadata["title"] = root["heading"]
        
        # If title not found, try to find first heading
        if not metadata.get("title") and structure and "root" in structure:
            first_heading = self._find_first_heading(structure["root"])
            if first_heading:
                metadata["title"] = first_heading
        
        return metadata

    def _find_first_heading(self, node: Dict) -> Optional[str]:
        """Find the first heading in the structure tree."""
        if node.get("heading"):
            return node["heading"]
        for child in node.get("children", []):
            result = self._find_first_heading(child)
            if result:
                return result
        return None

    def parse_batch(self, documents: List[Dict[str, Any]]) -> List[ParsedDocument]:
        """
        Parse multiple documents in batch.
        
        Args:
            documents: List of dicts with 'content' (bytes) and 'id' (str)
            
        Returns:
            List of ParsedDocument objects
        """
        results = []
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(results)}")
            pdf_bytes = doc.get("content")
            if not pdf_bytes:
                logger.warning(f"Skipping document {doc_id} – no content provided")
                continue
            try:
                result = self.parse(pdf_bytes, doc_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch parse failed for {doc_id}: {e}", exc_info=True)
                # Create error result
                results.append(ParsedDocument(
                    document_id=doc_id,
                    metadata={"parsed_at": datetime.utcnow().isoformat()},
                    structure={},
                    elements={},
                    raw_text="",
                    processing_time=0.0,
                    errors=[f"Batch parse error: {str(e)}"],
                    warnings=[]
                ))
        return results

    def parse_to_json(self, pdf_content: bytes, document_id: str = "unknown") -> str:
        """Parse and return result as JSON string."""
        result = self.parse(pdf_content, document_id)
        return result.to_json()

    def parse_to_file(self, pdf_content: bytes, document_id: str, output_dir: str) -> str:
        """
        Parse and save result to a JSON file.
        
        Args:
            pdf_content: Raw PDF bytes
            document_id: Document identifier
            output_dir: Directory to save JSON file
            
        Returns:
            Path to the saved file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        result = self.parse(pdf_content, document_id)
        filepath = os.path.join(output_dir, f"{document_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        logger.info(f"Saved parsed document to {filepath}")
        return filepath