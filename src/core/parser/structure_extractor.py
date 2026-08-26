"""
Layer 2: Structure Extraction
Uses Dedoc for document hierarchy with custom post-processing for French legal documents.
Provides a robust fallback using regex and formatting heuristics.
"""
import io
import logging
import re
import hashlib
import json
import os
import tempfile
from typing import Optional, Dict, Any, List, Tuple
from filelock import FileLock
import requests

import google.auth.transport.requests
import google.oauth2.id_token

# Dedoc import (graceful fallback)
try:
    from dedoc import DedocManager
    from dedoc.data_structures import Document, Node
    DEDOC_AVAILABLE = True
except ImportError:
    DEDOC_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Dedoc not installed - structure extraction will fall back to regex")

from src.settings import settings

logger = logging.getLogger(__name__)


class StructureExtractor:
    """
    Extracts hierarchical document structure (sections, subsections, articles)
    using Dedoc (primary) or a regex/formatting-based fallback.

    Designed to consume output from Layer 1 (TextExtractor) and produce a
    normalized tree suitable for RAG ingestion.
    """

    # French legal numbering patterns – used for both Dedoc post-processing
    # and fallback extraction.
    FRENCH_PATTERNS = {
        "article": re.compile(r"^(Article|ART\.?|Art\.?)\s+(\d+|[IVXLCDM]+)", re.IGNORECASE),
        "section": re.compile(r"^(Section|SECTION)\s+(\d+|[IVXLCDM]+)", re.IGNORECASE),
        "subsection": re.compile(r"^(Sous-section|Sous-Section)\s+(\d+|[IVXLCDM]+)", re.IGNORECASE),
        "paragraph": re.compile(r"^([A-Z]\.|[0-9]+\.|[IVXLCDM]+\.)", re.IGNORECASE),
        "subparagraph": re.compile(r"^([a-z]\.|[0-9]+\)|[IVXLCDM]+\))", re.IGNORECASE),
        "clause": re.compile(r"^(Clause|CLAUSE)\s+(\d+|[IVXLCDM]+)", re.IGNORECASE),
    }

    # Mapping from pattern type to hierarchy level (0 = top)
    LEVEL_MAP = {
        "article": 0,
        "section": 1,
        "clause": 1,
        "subsection": 2,
        "paragraph": 3,
        "subparagraph": 4,
    }

    def __init__(
        self,
        use_dedoc: bool = True,
        language: str = "fr",
        cache_dir: Optional[str] = None,
        dedoc_parameters: Optional[Dict] = None,
        dedoc_url: Optional[str] = None,
        dedoc_timeout: int = 3600,
    ):
        """
        Args:
            use_dedoc: Whether to attempt Dedoc extraction (falls back to regex if unavailable).
            language: Language hint for Dedoc ('fr', 'en', etc.).
            cache_dir: Optional directory to cache extracted structures (JSON).
            dedoc_parameters: Additional parameters to pass to Dedoc (e.g., 'structure_type').
            dedoc_url: Base URL of a remote Dedoc service (e.g. the Cloud Run dedoc-service).
                       If set, this takes priority over the local DedocManager import.
            dedoc_timeout: Request timeout (seconds) for the remote Dedoc call.
        """
        # Same block, extend it to also pass language:
        self.language = language
        self.dedoc_parameters = dedoc_parameters or {
            "structure_type": "tree",
            "language": "fra" if language == "fr" else "eng",
        }
        self.dedoc_url = dedoc_url.strip() if dedoc_url and dedoc_url.strip() else None
        self.dedoc_timeout = dedoc_timeout

        # deployed dedoc-service on Cloud Run is actually meant to be used.
        self.use_dedoc_remote = use_dedoc and self.dedoc_url is not None
        self.use_dedoc_local = use_dedoc and not self.use_dedoc_remote and DEDOC_AVAILABLE
        self.use_dedoc = self.use_dedoc_remote or self.use_dedoc_local

        logger.info(f"StructureExtractor remote mode: use_dedoc_remote={self.use_dedoc_remote}, dedoc_url={self.dedoc_url}")

        if self.use_dedoc_remote:
            self.dedoc_manager = None
            logger.info(f"Dedoc configured in remote mode: {self.dedoc_url}")
        elif self.use_dedoc_local:
            self.dedoc_manager = DedocManager()
            logger.info("Dedoc initialized in local mode")
        else:
            self.dedoc_manager = None
            if use_dedoc:
                logger.warning("Dedoc requested but neither dedoc_url nor local package available – using regex fallback")

        # Optional cache
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "structure_cache.json")
            self.cache_lock_file = self.cache_file + ".lock"
            self._ensure_cache_file()

    def _ensure_cache_file(self) -> None:
        """Create an empty cache file if it doesn't exist."""
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w") as f:
                json.dump({}, f)

    def _get_content_hash(self, pdf_content: bytes, raw_text: str) -> str:
        """Compute a hash based on PDF content + raw text to detect changes."""
        combined = pdf_content + raw_text.encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def _load_from_cache(self, content_hash: str) -> Optional[Dict]:
        """Load cached structure or None if not found."""
        if not self.cache_dir:
            return None
        try:
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
            if content_hash in cache:
                logger.info(f"Structure cache hit for hash {content_hash[:8]}...")
                return cache[content_hash]
            logger.info(f"Structure cache miss for hash {content_hash[:8]}...")
            return None
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read structure cache: {e}")
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
        layer1_result: Dict[str, Any],
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract document structure with hierarchy.

        Args:
            pdf_content: Raw PDF bytes (for Dedoc and hashing).
            layer1_result: Output from TextExtractor (dict with 'pages', 'text', etc.).
            force_reprocess: If True, ignore cache and re-extract.

        Returns:
            Dict with keys:
            - structure: hierarchical document tree (root node)
            - error: optional error message
            - warnings: list of warnings
            - total_sections: number of sections extracted
        """
        result = {
            "structure": {"root": None},
            "error": None,
            "warnings": [],
            "total_sections": 0,
        }

        raw_text = layer1_result.get("text", "")
        if not raw_text and not pdf_content:
            result["error"] = "No text or PDF content provided"
            return result

        # Check cache (unless forced)
        content_hash = self._get_content_hash(pdf_content, raw_text)
        if not force_reprocess:
            cached = self._load_from_cache(content_hash)
            if cached:
                result["structure"] = cached
                result["total_sections"] = self._count_nodes(cached.get("root", {}))
                return result

        try:
            # Primary: try Dedoc
            if self.use_dedoc:
                structure = self._extract_with_dedoc(pdf_content, raw_text)
                if structure and structure.get("root"):
                    # Normalize French numbering
                    structure = self._normalize_french_structure(structure)
                    result["structure"] = structure
                    result["total_sections"] = self._count_nodes(structure.get("root", {}))
                    self._save_to_cache(content_hash, structure)
                    logger.info(f"Structure extracted with Dedoc: {result['total_sections']} sections")
                    return result
                else:
                    result["warnings"].append("Dedoc extraction returned no structure – falling back")

            # Fallback: regex-based extraction
            logger.info("Using fallback regex structure extraction")
            structure = self._extract_with_regex(raw_text, layer1_result.get("pages", []))
            if structure and structure.get("root"):
                structure = self._normalize_french_structure(structure)
                result["structure"] = structure
                result["total_sections"] = self._count_nodes(structure.get("root", {}))
                result["warnings"].append("Used regex-based structure (less accurate)")
                self._save_to_cache(content_hash, structure)
                logger.info(f"Structure extracted with regex: {result['total_sections']} sections")
                return result
            else:
                # Last resort: flat structure
                structure = self._create_flat_structure(raw_text)
                result["structure"] = structure
                result["total_sections"] = 0
                result["warnings"].append("Could not detect hierarchy – flat structure used")
                logger.warning("Flat structure used – hierarchy may be missing")
                self._save_to_cache(content_hash, structure)
                return result

        except Exception as e:
            logger.error(f"Structure extraction failed: {e}", exc_info=True)
            result["error"] = str(e)
            # Attempt to return flat structure as last resort
            try:
                flat = self._create_flat_structure(raw_text)
                result["structure"] = flat
                result["warnings"].append("Error during extraction – flat structure returned")
            except Exception:
                pass
            return result

    # -------------------------------------------------------------------------
    # Dedoc Integration
    # -------------------------------------------------------------------------

    def _extract_with_dedoc(self, pdf_content: bytes, raw_text: str) -> Optional[Dict]:
        """Extract structure using Dedoc (remote HTTP service preferred, local package as fallback)."""
        if not self.use_dedoc:
            return None

        if self.use_dedoc_remote:
            return self._extract_with_dedoc_remote(pdf_content)
        return self._extract_with_dedoc_local(pdf_content)

    def _get_dedoc_auth_header(self) -> Dict[str, str]:
        """
        dedoc-service is a private Cloud Run service (only legal-rag-sa has
        roles/run.invoker on it). Unauthenticated requests get rejected by
        Cloud Run's IAM layer -- Google Frontend returns a generic Apache-style
        403 before the request ever reaches the dedoc container. This fetches
        a Google-signed identity token scoped to dedoc_url as the audience,
        using the ambient credentials of whatever service account this
        container runs as (legal-rag-sa in Cloud Run; your local ADC when
        testing locally).
        """
        try:
            auth_req = google.auth.transport.requests.Request()
            token = google.oauth2.id_token.fetch_id_token(auth_req, self.dedoc_url)
            return {"Authorization": f"Bearer {token}"}
        except Exception as e:
            logger.warning(f"Could not fetch identity token for Dedoc call: {e}")
            return {}

    def _extract_with_dedoc_remote(self, pdf_content: bytes) -> Optional[Dict]:
        try:
            files = {"file": ("document.pdf", pdf_content, "application/pdf")}
            data = {k: str(v) for k, v in self.dedoc_parameters.items()}
            headers = self._get_dedoc_auth_header()
            logger.info(f"Calling Dedoc at {self.dedoc_url}/upload with data={data}, authed={'Authorization' in headers}")
            response = requests.post(
                f"{self.dedoc_url}/upload",
                files=files,
                data=data,
                headers=headers,
                timeout=self.dedoc_timeout,
            )
            logger.info(f"Dedoc response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Dedoc returned {response.status_code}: {response.text[:500]}")
                return None
            dedoc_result = response.json()
            logger.info(f"Dedoc response keys: {dedoc_result.keys() if isinstance(dedoc_result, dict) else 'not a dict'}")
            logger.info(f"Dedoc response preview: {str(dedoc_result)[:500]}")
            return self._convert_dedoc_to_structure(dedoc_result)
        except Exception as e:
            logger.error(f"Dedoc remote extraction error: {e}", exc_info=True)
            return None

    def _extract_with_dedoc_local(self, pdf_content: bytes) -> Optional[Dict]:
        """Extract structure using the local dedoc package (only available if installed)."""
        if not self.dedoc_manager:
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name
            try:
                dedoc_result = self.dedoc_manager.parse_document(
                    tmp_path,
                    parameters=self.dedoc_parameters,
                )
                return self._convert_dedoc_to_structure(dedoc_result)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Dedoc local extraction error: {e}", exc_info=True)
            return None

    def _convert_dedoc_to_structure(self, dedoc_result) -> Dict:
        logger.info(f"Converting Dedoc result of type {type(dedoc_result)}")
        if isinstance(dedoc_result, dict):
            logger.info(f"Dedoc result keys: {dedoc_result.keys()}")
        root = {
            "section_id": "root",
            "level": -1,
            "heading": "Document",
            "text": "",
            "section_type": "root",
            "children": [],
            "breadcrumb": [],
            "page": 1,
        }

        # The real dedoc REST API (v2.7, confirmed from a live response) nests
        # the structure tree at content.structure, not at the top level, and
        # each node's own children live under "subparagraphs", not "children" --
        # this doesn't match the SDK-object shape the original code assumed.
        dedoc_tree = None
        if isinstance(dedoc_result, dict) and "content" in dedoc_result:
            content = dedoc_result.get("content") or {}
            if isinstance(content, dict) and "structure" in content:
                dedoc_tree = content["structure"]
                logger.info("Dedoc result had content.structure")
        if dedoc_tree is None and hasattr(dedoc_result, "structure"):
            dedoc_tree = dedoc_result.structure
            logger.info("Dedoc result had 'structure' attribute")
        if dedoc_tree is None and isinstance(dedoc_result, dict) and "structure" in dedoc_result:
            dedoc_tree = dedoc_result["structure"]
            logger.info("Dedoc result had top-level 'structure' key")

        if dedoc_tree is not None:
            # The top node from Dedoc is itself the document root (paragraph_type
            # "root") -- we want ITS subparagraphs as our root's children, not
            # to wrap the whole tree one level too deep.
            root["children"] = self._traverse_dedoc(dedoc_tree.get("subparagraphs", []) if isinstance(dedoc_tree, dict) else dedoc_tree)
        else:
            logger.warning("Dedoc result has no recognizable structure field")
            root["children"] = []

        logger.info(f"Dedoc conversion produced {len(root['children'])} top-level children")
        return {"root": root}

    def _traverse_dedoc(self, node, depth: int = 0) -> List[Dict]:
        """
        Recursively traverse a Dedoc node and convert to our format.

        Real Dedoc REST API v2.7 schema (confirmed from a live response):
        each node is a dict with keys node_id, text, annotations, metadata
        (paragraph_type, page_id, line_id), subparagraphs (list of child nodes,
        NOT "children"). There is no "heading"/"title" key -- for nodes whose
        paragraph_type looks like a heading (e.g. "header"), we use the node's
        own text as the heading; for body text nodes we detect French legal
        section types (article/section/clause/etc.) from the text itself via
        the existing FRENCH_PATTERNS regexes.
        """
        children = []

        if isinstance(node, list):
            for item in node:
                children.extend(self._traverse_dedoc(item, depth))
            return children

        if isinstance(node, dict) and "node_id" in node:
            text = (node.get("text") or "").strip()
            metadata = node.get("metadata") or {}
            paragraph_type = metadata.get("paragraph_type", "raw_text")
            page = metadata.get("page_id")
            subparagraphs = node.get("subparagraphs") or []

            section_type = self._detect_section_type(text) if text else "paragraph"
            is_heading_like = paragraph_type in ("header", "root") or section_type != "paragraph"

            current = {
                "section_id": "",
                "level": depth,
                "heading": text[:100] if (is_heading_like and text) else "",
                "text": text,
                "section_type": section_type,
                "children": [],
                "breadcrumb": [],
                "page": (page + 1) if isinstance(page, int) else 1,  # dedoc pages are 0-indexed
            }
            if subparagraphs:
                current["children"] = self._traverse_dedoc(subparagraphs, depth + 1)
            children.append(current)
            return children

        # Fallback for any other shape (SDK Node objects, plain strings, etc.)
        if hasattr(node, "text") and hasattr(node, "subparagraphs"):
            text = node.text or ""
            section_type = self._detect_section_type(text) if text else "paragraph"
            current = {
                "section_id": "",
                "level": depth,
                "heading": text[:100] if section_type != "paragraph" else "",
                "text": text,
                "section_type": section_type,
                "children": [],
                "breadcrumb": [],
                "page": 1,
            }
            sub = getattr(node, "subparagraphs", None)
            if sub:
                current["children"] = self._traverse_dedoc(sub, depth + 1)
            children.append(current)
        elif node:
            children.append({
                "section_id": "",
                "level": depth,
                "heading": "",
                "text": str(node),
                "section_type": "paragraph",
                "children": [],
                "breadcrumb": [],
                "page": 1,
            })
        return children

    def _detect_section_type(self, text: str) -> str:
        """
        Classify a Dedoc node's text as a French legal section type using the
        same FRENCH_PATTERNS regexes the regex fallback path already relies on,
        so both paths produce consistent section_type values. Returns "paragraph"
        for anything that doesn't match a known heading pattern.
        """
        if not text:
            return "paragraph"
        stripped = text.strip()
        for type_name, pattern in self.FRENCH_PATTERNS.items():
            if pattern.match(stripped):
                return type_name
        return "paragraph"

    # -------------------------------------------------------------------------
    # French Legal Normalization
    # -------------------------------------------------------------------------

    def _normalize_french_structure(self, structure: Dict) -> Dict:
        """Normalize the hierarchy with French legal numbering."""
        root = structure.get("root")
        if not root:
            return structure

        # Assign section IDs and breadcrumbs recursively
        self._assign_ids(root, parent_path="")
        return structure

    def _assign_ids(self, node: Dict, parent_path: str) -> str:
        """
        Recursively assign section_id and breadcrumb.
        Returns the section_id of this node.
        """
        heading = node.get("heading", "")
        section_type = node.get("section_type", "paragraph")
        # Determine section number from heading
        num = self._extract_number(heading, section_type)
        if not num:
            num = "0"

        # Build ID and breadcrumb
        if parent_path:
            node["section_id"] = f"{parent_path}_{num}"
        else:
            node["section_id"] = f"sec_{num}" if num != "0" else "root"

        # Build breadcrumb from parent (stored in node for now)
        # We'll compute after children processed
        # For now, set a placeholder
        node["breadcrumb"] = [heading] if heading else []

        # Process children
        children = node.get("children", [])
        child_paths = []
        for child in children:
            child_paths.append(self._assign_ids(child, node["section_id"]))

        # Update breadcrumb by prepending parent's heading
        if parent_path:
            # Get parent breadcrumb from somewhere? We can store parent reference.
            # Simpler: recursively build breadcrumb later.
            pass

        return node["section_id"]

    def _extract_number(self, heading: str, section_type: str) -> str:
        if not heading:
            return "0"
        pattern = self.FRENCH_PATTERNS.get(section_type)
        if pattern:
            match = pattern.match(heading.strip())
            if match:
                # match.groups() returns a tuple -- must compare its length, not the
                # tuple itself. Patterns like "article" have 2 groups (word + number),
                # while "paragraph" has only 1 group (the number itself).
                if len(match.groups()) > 1:
                    return match.group(2) or "0"
                else:
                    return match.group(1) or "0"
        m = re.match(r"^(\d+|[IVXLCDM]+)\s*[\.\-\)]", heading)
        if m:
            return m.group(1)
        return "0"

    # -------------------------------------------------------------------------
    # Regex Fallback Extraction
    # -------------------------------------------------------------------------

    def _extract_with_regex(self, raw_text: str, pages: List[Dict]) -> Dict:
        """
        Fallback: extract hierarchy using regex patterns, walking page-by-page
        (when page data is available) so each node can be tagged with the page
        it started on -- required for Layer 3's table-to-section mapping.
        """
        root = {
            "section_id": "root",
            "level": -1,
            "heading": "Document",
            "text": "",
            "section_type": "root",
            "children": [],
            "breadcrumb": [],
            "page": 1,
        }

        stack = [root]

        # Walk page-by-page when we have per-page text, so we know which page
        # each line came from. Fall back to a single "page 1" pass otherwise.
        if pages:
            line_sources = [(p.get("page_num", i + 1), line)
                             for i, p in enumerate(pages)
                             for line in p.get("text", "").split("\n")]
        else:
            line_sources = [(1, line) for line in raw_text.split("\n")]

        for page_num, line in line_sources:
            line = line.strip()
            if not line:
                continue

            heading_info = self._detect_heading(line)
            if heading_info:
                level = heading_info["level"]
                section_type = heading_info["type"]
                heading_text = heading_info["text"]

                while len(stack) > level + 1:
                    stack.pop()
                parent = stack[-1]

                node = {
                    "section_id": "",
                    "level": level,
                    "heading": heading_text,
                    "text": "",
                    "section_type": section_type,
                    "children": [],
                    "breadcrumb": [],
                    "page": page_num,
                }
                parent["children"].append(node)
                stack.append(node)
            else:
                if stack:
                    node = stack[-1]
                    if "text" in node:
                        node["text"] += " " + line
                    else:
                        node["text"] = line

        self._remove_empty_nodes(root)
        return {"root": root}

    def _detect_heading(self, line: str) -> Optional[Dict]:
        """
        Detect if a line is a heading and determine its level.
        Returns dict with 'type', 'level', 'text'.
        """
        for type_name, pattern in self.FRENCH_PATTERNS.items():
            if pattern.match(line):
                return {
                    "type": type_name,
                    "level": self.LEVEL_MAP.get(type_name, 0),
                    "text": line
                }

        # Check for numbered headings (e.g., "1. Introduction")
        import re
        numbered = re.compile(r"^(\d+)\.\s+(.+)$")
        match = numbered.match(line)
        if match:
            return {
                "type": "numbered",
                "level": 0,  # treat as top-level
                "text": line
            }

        # Check for all-caps short lines (often section titles)
        if len(line) < 80 and line.isupper() and len(line) > 5:
            return {
                "type": "section",
                "level": 0,
                "text": line
            }

        return None

    def _remove_empty_nodes(self, node: Dict) -> None:
        """Remove children that have no text and no children."""
        if "children" in node:
            # Recursively clean children
            new_children = []
            for child in node.get("children", []):
                self._remove_empty_nodes(child)
                # Keep if it has text or children
                if child.get("text") or child.get("children"):
                    new_children.append(child)
            node["children"] = new_children

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    def _create_flat_structure(self, raw_text: str) -> Dict:
        """Create a flat structure with no hierarchy."""
        root = {
            "section_id": "root",
            "level": -1,
            "heading": "Document",
            "text": raw_text[:5000],  # Truncate for sanity
            "section_type": "root",
            "children": [],
            "breadcrumb": [],
            "page": 1,
        }
        return {"root": root}

    def _count_nodes(self, node: Dict) -> int:
        """Count total sections (including root)."""
        if not node:
            return 0
        count = 1
        for child in node.get("children", []):
            count += self._count_nodes(child)
        return count