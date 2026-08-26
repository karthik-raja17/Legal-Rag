"""
Structure-aware, clause-level chunking with TOC filtering and heading-body merging.
Production-grade for French legal contracts.
"""
import logging
import re
from typing import List, Dict, Optional, Any

from src.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Production‑grade chunker for legal documents.

    Features:
        - High-precision TOC filtering (requires leader dots or explicit dot-space patterns).
        - Safe heading-body merging without dropping tree descendants.
        - Hierarchical fallback splitting (Paragraph -> Sentence -> Token window).
        - Breadcrumb path retention for vector context.
        - Enforces minimum chunk size (>=50 chars) while avoiding artificial truncation.
    """

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens or getattr(settings, "MAX_CHUNK_TOKENS", 1900)
        self.chars_per_token_estimate = 3.3
        self.max_chars = int(self.max_tokens * self.chars_per_token_estimate)
        self.min_chunk_chars = 50
        logger.info(
            f"DocumentChunker initialized: max_tokens={self.max_tokens} "
            f"(~{self.max_chars} chars), min_chunk_chars={self.min_chunk_chars}"
        )

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token_estimate)

    def _is_toc_line(self, text: str) -> bool:
        """
        High-precision TOC detection. Avoids false positives on regular legal clauses.
        Matches:
            - Leader dots: 'Article 1 .............. 4' or 'Objet . . . . . . . 12'
            - Tab/Space heavy dot leaders ending in page numbers
        """
        stripped = text.strip()
        if not stripped:
            return False

        # 1. Standard leader dots (3 or more consecutive dots or dot-spaces) followed by optional page number
        if re.search(r'(\.{3,}|\.(\s*\.){2,})\s*\d*$', stripped):
            return True

        # 2. Repeated underscore / dash leaders
        if re.search(r'([_\-]{4,})\s*\d*$', stripped):
            return True

        # 3. Explicit SOMMAIRE / TABLE DES MATIERES header lines
        if re.match(r'^(sommaire|table\s+des\s+mati[eè]res)\b', stripped, re.IGNORECASE):
            return True

        return False

    def _filter_toc_text(self, text: str) -> str:
        """Strip TOC lines while preserving normal legal clause spacing."""
        if not text:
            return ""
        lines = text.split('\n')
        filtered = [line for line in lines if not self._is_toc_line(line)]
        return '\n'.join(filtered).strip()

    def _merge_heading_with_body(self, heading: str, body: str) -> str:
        """Combine heading and body, preventing redundant duplicates."""
        heading_clean = heading.strip()
        body_clean = body.strip()

        if not heading_clean:
            return body_clean
        if not body_clean:
            return heading_clean

        # If body already starts with the heading (case-insensitive check)
        if body_clean.lower().startswith(heading_clean.lower()):
            return body_clean

        return f"{heading_clean}\n\n{body_clean}"

    def chunk_document(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Main entry point: convert hierarchical structure tree into chunk list."""
        root = structure.get("root")
        if not root:
            logger.warning("Structure has no 'root' key; returning empty chunks.")
            return []

        chunks: List[Dict[str, Any]] = []
        self._traverse(root, [], chunks)
        logger.info(f"Chunked document into {len(chunks)} chunks")
        return chunks

    def _traverse(self, node: Dict[str, Any], breadcrumb: List[str], chunks: List[Dict[str, Any]]):
        heading = (node.get("heading") or "").strip()
        raw_text = node.get("text") or ""
        section_id = node.get("section_id") or "root"
        children = node.get("children") or []

        current_breadcrumb = breadcrumb + [heading] if heading else breadcrumb
        cleaned_text = self._filter_toc_text(raw_text)

        # Case 1: Node has substantive body text
        if cleaned_text:
            merged_text = self._merge_heading_with_body(heading, cleaned_text)
            if len(merged_text) >= self.min_chunk_chars:
                if self._estimate_tokens(merged_text) <= self.max_tokens:
                    chunk = self._make_chunk(node, merged_text, current_breadcrumb, heading, section_id)
                    if chunk:
                        chunks.append(chunk)
                else:
                    self._split_long_text(merged_text, node, current_breadcrumb, heading, section_id, chunks)

        # Case 2: Node is a heading-only container (no direct body text)
        elif heading and not children:
            # Only emit standalone heading if it is a leaf node and meaningful in length
            if len(heading) >= self.min_chunk_chars:
                chunk = self._make_chunk(node, heading, current_breadcrumb, heading, section_id)
                if chunk:
                    chunks.append(chunk)

        # Always traverse children so sub-sections/clauses are never lost
        for child in children:
            self._traverse(child, current_breadcrumb, chunks)

    def _make_chunk(
        self,
        node: Dict[str, Any],
        text: str,
        breadcrumb: List[str],
        heading: str,
        section_id: str
    ) -> Optional[Dict[str, Any]]:
        text_clean = text.strip()
        if not text_clean:
            return None

        # Build clean breadcrumb trail string (e.g., "Document > Article 2 > Section 2.1")
        clean_bc = [b for b in breadcrumb if b]
        bc_str = " > ".join(clean_bc) if clean_bc else (heading or "Document")

        return {
            "text": text_clean,
            "section_id": section_id,
            "heading": heading[:200] if heading else "",
            "breadcrumb": bc_str,
            "level": node.get("level", 0),
            "page": node.get("page", 1),
            "section_type": node.get("section_type", "paragraph"),
            "clause_type": node.get("clause_type", "general"),
            "embedding": node.get("embedding"),
        }

    def _split_long_text(
        self,
        text: str,
        node: Dict[str, Any],
        breadcrumb: List[str],
        heading: str,
        section_id: str,
        chunks: List[Dict[str, Any]]
    ):
        """
        Split text exceeding max_tokens.
        COLLECTS all parts first, then emits them with part_number and total_parts.
        """
        # 1. Break into atomic units (paragraphs -> sentences -> chars)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        # 2. Greedily pack units into batches within token limit
        batches = []
        current_batch: List[str] = []
        for unit in paragraphs:
            candidate = "\n\n".join(current_batch + [unit]) if current_batch else unit
            if self._estimate_tokens(candidate) <= self.max_tokens:
                current_batch.append(unit)
            else:
                if current_batch:
                    batches.append("\n\n".join(current_batch))
                # Handle a single unit that is still too long (force char split)
                if self._estimate_tokens(unit) > self.max_tokens:
                    for i in range(0, len(unit), self.max_chars):
                        slice_text = unit[i:i + self.max_chars].strip()
                        if len(slice_text) >= self.min_chunk_chars:
                            batches.append(slice_text)
                    current_batch = []
                else:
                    current_batch = [unit]

        if current_batch:
            batches.append("\n\n".join(current_batch))

        # 3. EMIT with total_parts count
        total = len(batches)
        for idx, batch_text in enumerate(batches, start=1):
            self._emit_split_chunk(
                chunk_text=batch_text,
                node=node,
                breadcrumb=breadcrumb,
                heading=heading,
                section_id=section_id,
                part_num=idx,
                total_parts=total,  # <-- Pass total count
                chunks=chunks
            )

    def _emit_split_chunk(
        self,
        chunk_text: str,
        node: Dict[str, Any],
        breadcrumb: List[str],
        heading: str,
        section_id: str,
        part_num: int,
        total_parts: int,  # New parameter
        chunks: List[Dict[str, Any]]
    ):
        parent_section_id = section_id.split("_part_")[0] if "_part_" in section_id else section_id
        split_heading = f"{heading} (part {part_num}/{total_parts})" if heading else f"(part {part_num}/{total_parts})"
        
        clean_bc = [b for b in breadcrumb if b]
        bc_str = " > ".join(clean_bc) if clean_bc else (heading or "Document")

        chunks.append({
            "text": chunk_text,
            "section_id": f"{section_id}_part_{part_num}",
            "heading": split_heading[:200],
            "breadcrumb": bc_str,
            "level": node.get("level", 0),
            "page": node.get("page", 1),
            "section_type": node.get("section_type", "paragraph"),
            "clause_type": node.get("clause_type", "general"),
            "embedding": None,
            
            # ----- NEW CONTEXT-WINDOW METADATA -----
            "parent_section_id": parent_section_id,  # Groups all parts
            "part_number": part_num,                 # Current index (1-based)
            "total_parts": total_parts,              # Total count
            "is_part": total_parts > 1,              # Flag
        })