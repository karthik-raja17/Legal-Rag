"""
Layer 3: Specialized Element Extraction
Extracts tables, figures, and other structured elements from PDFs.
Supports mapping to hierarchical sections from Layer 2.
"""
import io
import logging
import hashlib
import json
import os
import tempfile
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
from filelock import FileLock

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Optional: tabula-py (Java-based, slower but accurate)
try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

logger = logging.getLogger(__name__)


class ElementExtractor:
    """
    Extracts specialized elements from PDFs:
    - Tables (using Camelot + pdfplumber fallback + optional tabula)
    - Figures/Graphs (using Unstructured.io or Pillow – placeholder)

    Designed to work with Layer 1 (TextExtractor) and Layer 2 (StructureExtractor)
    outputs to enrich the document hierarchy with element references.
    """

    def __init__(
        self,
        extract_tables: bool = True,
        extract_figures: bool = False,
        table_methods: List[str] = None,
        cache_dir: Optional[str] = None,
        max_pages_for_ocr: int = 50,  # For figure extraction with vision models
    ):
        """
        Args:
            extract_tables: Whether to extract tables.
            extract_figures: Whether to attempt figure/graph extraction.
            table_methods: Ordered list of methods to try: ['camelot_lattice', 'camelot_stream', 'pdfplumber', 'tabula'].
                           Defaults to all available.
            cache_dir: Directory to cache extracted elements (JSON).
            max_pages_for_ocr: Max pages to process for figure extraction (to limit cost).
        """
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures
        self.cache_dir = cache_dir
        self.max_pages_for_ocr = max_pages_for_ocr

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "ocr_cache.json")
            self.cache_lock_file = self.cache_file + ".lock"
            self._ensure_cache_file()

        # Determine available table extraction methods
        self.table_methods = table_methods or [
            "camelot_lattice",
            "camelot_stream",
            "pdfplumber",
            "tabula"
        ]
        # Filter to those available
        self.available_methods = []
        for method in self.table_methods:
            if method.startswith("camelot") and CAMELOT_AVAILABLE:
                self.available_methods.append(method)
            elif method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                self.available_methods.append(method)
            elif method == "tabula" and TABULA_AVAILABLE:
                self.available_methods.append(method)

        if not self.available_methods and extract_tables:
            logger.warning("No table extraction libraries available – install camelot, pdfplumber, or tabula")

        # For figure extraction, we'll use a placeholder; you can integrate Unstructured.io later
        self.figure_extractor = None  # To be initialized if extract_figures

    def _ensure_cache_file(self) -> None:
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w") as f:
                json.dump({}, f)

    def _get_content_hash(self, pdf_content: bytes) -> str:
        """Hash the PDF bytes for caching."""
        return hashlib.sha256(pdf_content).hexdigest()

    def _load_from_cache(self, content_hash: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        try:
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
            if content_hash in cache:
                logger.info(f"Element cache hit for hash {content_hash[:8]}...")
                return cache[content_hash]
            return None
        except Exception as e:
            logger.warning(f"Failed to read element cache: {e}")
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

    def extract(
        self,
        pdf_content: bytes,
        structure: Dict,
        layer1_result: Optional[Dict] = None,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract tables and figures, and map them to the document structure.

        Args:
            pdf_content: Raw PDF bytes.
            structure: The hierarchy from Layer 2 (contains pages and sections).
            layer1_result: Optional output from TextExtractor (for page-level text and confidence).
            force_reprocess: Ignore cache and re-extract.

        Returns:
            Dict with keys:
            - tables: List of table dicts with page, data, headers, section_id, method
            - figures: List of figure dicts (placeholder)
            - warnings: List of warnings
            - errors: List of errors
            - element_count: Total number of elements extracted
        """
        result = {
            "tables": [],
            "figures": [],
            "warnings": [],
            "errors": [],
            "element_count": 0,
        }

        # Check cache
        content_hash = self._get_content_hash(pdf_content)
        if not force_reprocess:
            cached = self._load_from_cache(content_hash)
            if cached:
                # Merge cached data with structure mapping (section ids may change)
                # We'll re-map tables to sections based on page numbers
                cached_tables = cached.get("tables", [])
                if cached_tables:
                    result["tables"] = self._map_tables_to_sections(cached_tables, structure)
                    result["element_count"] += len(result["tables"])
                if cached.get("figures"):
                    result["figures"] = cached["figures"]
                    result["element_count"] += len(result["figures"])
                result["warnings"] = cached.get("warnings", [])
                logger.info(f"Loaded {len(result['tables'])} tables and {len(result['figures'])} figures from cache")
                return result

        # Perform extraction
        try:
            if self.extract_tables:
                raw_tables = self._extract_tables(pdf_content)
                if raw_tables:
                    # Map tables to sections using structure
                    mapped_tables = self._map_tables_to_sections(raw_tables, structure)
                    result["tables"] = mapped_tables
                    result["element_count"] += len(mapped_tables)
                    logger.info(f"Extracted and mapped {len(mapped_tables)} tables")
                else:
                    result["warnings"].append("No tables extracted")

            if self.extract_figures:
                figures = self._extract_figures(pdf_content, layer1_result)
                result["figures"] = figures
                result["element_count"] += len(figures)

            # Cache the raw (unmapped) tables and figures; mapping will be done on each load
            # to handle structure changes. We'll cache the raw extraction.
            self._save_to_cache(content_hash, {
                "tables": raw_tables if self.extract_tables else [],
                "figures": result["figures"] if self.extract_figures else [],
                "warnings": result["warnings"],
            })

        except Exception as e:
            logger.error(f"Element extraction failed: {e}", exc_info=True)
            result["errors"].append(str(e))

        return result

    # -------------------------------------------------------------------------
    # Table Extraction
    # -------------------------------------------------------------------------

    def _extract_tables(self, pdf_content: bytes) -> List[Dict]:
        """
        Extract tables using the configured methods in order.
        Returns a list of table dicts with page, data, headers, method.
        """
        tables = []
        # Save PDF to temp file for libraries that need a file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name

        try:
            for method in self.available_methods:
                try:
                    if method == "camelot_lattice":
                        extracted = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
                        for table in extracted:
                            tables.append({
                                "page": table.page,
                                "data": table.df.to_dict(orient="records"),
                                "headers": table.df.columns.tolist(),
                                "method": "camelot_lattice",
                                "raw_table": table,  # Keep for later if needed
                            })
                        if tables:
                            logger.info(f"Camelot lattice extracted {len(tables)} tables")
                            break  # Success, stop trying other methods

                    elif method == "camelot_stream":
                        extracted = camelot.read_pdf(tmp_path, pages="all", flavor="stream")
                        for table in extracted:
                            tables.append({
                                "page": table.page,
                                "data": table.df.to_dict(orient="records"),
                                "headers": table.df.columns.tolist(),
                                "method": "camelot_stream",
                                "raw_table": table,
                            })
                        if tables:
                            logger.info(f"Camelot stream extracted {len(tables)} tables")
                            break

                    elif method == "pdfplumber":
                        with pdfplumber.open(tmp_path) as pdf:
                            for page_num, page in enumerate(pdf.pages):
                                extracted = page.extract_tables()
                                for table_data in extracted:
                                    if table_data and len(table_data) > 1:
                                        headers = table_data[0] if table_data[0] else None
                                        rows = table_data[1:] if len(table_data) > 1 else []
                                        if headers:
                                            data = [
                                                {headers[i]: row[i] if i < len(row) else None
                                                 for i in range(len(headers))}
                                                for row in rows
                                            ]
                                        else:
                                            data = rows
                                        tables.append({
                                            "page": page_num + 1,
                                            "data": data,
                                            "headers": headers,
                                            "method": "pdfplumber",
                                        })
                        if tables:
                            logger.info(f"pdfplumber extracted {len(tables)} tables")
                            break

                    elif method == "tabula":
                        # tabula works with file path
                        extracted = tabula.read_pdf(tmp_path, pages="all", multiple_tables=True)
                        for df in extracted:
                            # Convert DataFrame to dict
                            data = df.to_dict(orient="records")
                            headers = df.columns.tolist()
                            tables.append({
                                "page": None,  # tabula doesn't easily give page number; we'll try to infer later
                                "data": data,
                                "headers": headers,
                                "method": "tabula",
                            })
                        if tables:
                            # Try to assign page numbers via heuristics? For now, set page=1
                            for t in tables:
                                t["page"] = 1  # placeholder
                            logger.info(f"tabula extracted {len(tables)} tables")
                            break

                except Exception as e:
                    logger.warning(f"Table extraction with {method} failed: {e}")
                    continue

            # Clean up: remove raw_table if present to keep JSON serializable
            for t in tables:
                t.pop("raw_table", None)

            return tables

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Table-to-Section Mapping
    # -------------------------------------------------------------------------

    def _map_tables_to_sections(self, tables: List[Dict], structure: Dict) -> List[Dict]:
        """
        Assign each table to the closest section based on page number.
        Returns enriched table dicts with 'section_id' and 'section_heading'.
        """
        if not tables or not structure:
            return tables

        # Build a mapping from page number to section ID (using leaf nodes)
        page_section_map = self._build_page_section_map(structure)

        mapped = []
        for table in tables:
            page_num = table.get("page")
            if not page_num:
                # Try to infer from position? For now, set to root
                table["section_id"] = "root"
                table["section_heading"] = "Document"
            else:
                # Find the deepest section that covers this page
                section_id = page_section_map.get(page_num)
                if section_id:
                    # Get the section heading from structure
                    heading = self._find_section_heading(structure, section_id)
                    table["section_id"] = section_id
                    table["section_heading"] = heading or "Unknown"
                else:
                    # Fallback to root
                    table["section_id"] = "root"
                    table["section_heading"] = "Document"
            mapped.append(table)
        return mapped

    def _build_page_section_map(self, structure: Dict) -> Dict[int, str]:
        """
        Traverse the structure tree and map each leaf section to its page number
        (if available). Returns dict {page_num: section_id}.
        """
        page_map = {}
        root = structure.get("root")
        if not root:
            return page_map

        def traverse(node, page_assigned=None):
            # If node has a page number, use it
            node_page = node.get("page")
            if node_page:
                page_assigned = node_page
            # Assign this node to page_assigned if we have one
            if page_assigned is not None and node.get("section_id"):
                # We'll store the deepest node for each page
                page_map[page_assigned] = node["section_id"]
            for child in node.get("children", []):
                traverse(child, page_assigned)

        traverse(root)
        return page_map

    def _find_section_heading(self, structure: Dict, section_id: str) -> Optional[str]:
        """Find the heading of a section by its ID."""
        root = structure.get("root")
        if not root:
            return None

        def search(node):
            if node.get("section_id") == section_id:
                return node.get("heading")
            for child in node.get("children", []):
                result = search(child)
                if result:
                    return result
            return None

        return search(root)

    # -------------------------------------------------------------------------
    # Figure Extraction (Placeholder)
    # -------------------------------------------------------------------------

    def _extract_figures(self, pdf_content: bytes, layer1_result: Optional[Dict]) -> List[Dict]:
        """
        Extract figures/graphs. Placeholder – can be extended with:
        - Unstructured.io (via partition_pdf)
        - Google Vision API
        - Custom image detection using OpenCV/Pillow
        """
        # This is a stub – you can integrate a vision model here.
        # For now, we'll return an empty list with a warning if figures requested.
        if self.extract_figures:
            logger.warning("Figure extraction is not fully implemented – returning empty list")
        return []

    # -------------------------------------------------------------------------
    # Convenience: Save/load raw tables for debugging
    # -------------------------------------------------------------------------

    def save_tables_to_csv(self, tables: List[Dict], output_dir: str) -> None:
        """Save extracted tables as CSV files for inspection."""
        os.makedirs(output_dir, exist_ok=True)
        for i, table in enumerate(tables):
            data = table.get("data", [])
            if not data:
                continue
            headers = table.get("headers", [])
            import csv
            filepath = os.path.join(output_dir, f"table_{i+1}_page_{table.get('page', 1)}.csv")
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                for row in data:
                    if isinstance(row, dict):
                        writer.writerow([row.get(h, "") for h in headers])
                    else:
                        writer.writerow(row)
            logger.info(f"Saved table {i+1} to {filepath}")