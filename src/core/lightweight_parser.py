"""
Lightweight Legal Contract Parser
Replaces Dedoc/Camelot with PyMuPDF + Stateful Regex Breadcrumbs.
Enforces clause hierarchy survival (SECTION 8.2 > (b)) without AST bloat.
Supports both direct PDF parsing and pre-extracted raw text parsing.
"""
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# 1. Strict Heading Hierarchy Patterns (CUAD & General Legal)
HEADING_REGEX = re.compile(
    r'(?m)^(?:\s*)(?P<type>ARTICLE|SECTION|SCHEDULE|EXHIBIT|CLAUSE|APPENDIX|ANNEX)\s+(?P<num>[\d\w\.\-]+)(?:\s*[:\-\.]?\s*(?P<title>[^\n]+))?',
    re.IGNORECASE
)
SUBCLAUSE_REGEX = re.compile(
    r'(?m)^(?:\s*)(?P<num>\([a-z\d]+\)|\([ivxlcdm]+\)|\([A-Z]\)|[a-z]\.|\([0-9]+\))',
    re.IGNORECASE
)
PREAMBLE_REGEX = re.compile(r'^(RECITALS|WITNESSETH|BACKGROUND|PREAMBLE)', re.IGNORECASE)


def parse_and_chunk_text(
    full_text: str,
    doc_id: Optional[str] = None,
    max_leaf_chars: int = 1200,
    min_text_length: int = 50
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Builds hierarchical breadcrumbs from raw contract text and chunks into Small-to-Big format.
    Returns (chunks, full_raw_text).
    """
    doc_prefix = f"{doc_id}_" if doc_id else ""

    if len(full_text.strip()) < min_text_length:
        flat_chunk = {
            "leaf_id": f"{doc_prefix}chunk_flat_0",
            "leaf_text": f"[FLAT]\n{full_text}",
            "parent_id": f"{doc_prefix}sec_root",
            "parent_text": full_text,
            "breadcrumb": "Flat",
            "document_id": doc_id or "unknown"
        }
        return [flat_chunk], full_text

    # --- 1. Detect Section Boundaries (Stateful AST Builder) ---
    matches = list(HEADING_REGEX.finditer(full_text))
    sections = []

    # Capture Preamble
    if matches and matches[0].start() > 0:
        preamble_text = full_text[:matches[0].start()].strip()
        if PREAMBLE_REGEX.search(preamble_text) or len(preamble_text) > 100:
            sections.append({"breadcrumb": "PREAMBLE", "text": preamble_text})

    # Iterate matches
    for i, match in enumerate(matches):
        h_type = match.group("type").upper()
        h_num = match.group("num")
        h_title = match.group("title").strip() if match.group("title") else ""
        breadcrumb = f"{h_type} {h_num}" + (f": {h_title}" if h_title else "")

        start_pos = match.start()
        end_pos = matches[i + 1].start() if (i + 1) < len(matches) else len(full_text)
        section_body = full_text[start_pos:end_pos].strip()
        sections.append({"breadcrumb": breadcrumb, "text": section_body})

    # If no sections found, treat whole doc as one section
    if not sections:
        sections.append({"breadcrumb": "DOCUMENT", "text": full_text})

    # --- 2. Chunk into Leaves with Inherited Breadcrumbs ---
    chunks = []
    chunk_counter = 0

    for sec_idx, section in enumerate(sections):
        parent_id = f"{doc_prefix}sec_{sec_idx}"
        parent_text = section["text"]
        breadcrumb = section["breadcrumb"]

        # If section fits, keep it intact
        if len(parent_text) <= max_leaf_chars:
            chunks.append({
                "leaf_id": f"{doc_prefix}chunk_{chunk_counter}",
                "leaf_text": f"[{breadcrumb}]\n{parent_text}",
                "parent_id": parent_id,
                "parent_text": parent_text,
                "breadcrumb": breadcrumb,
                "document_id": doc_id or "unknown"
            })
            chunk_counter += 1
            continue

        # --- 3. Sub-split large sections preserving subsection hierarchy ---
        sub_matches = list(SUBCLAUSE_REGEX.finditer(parent_text))

        if sub_matches:
            for j, sub in enumerate(sub_matches):
                sub_num = sub.group("num")
                s_start = sub.start()
                s_end = sub_matches[j + 1].start() if (j + 1) < len(sub_matches) else len(parent_text)
                sub_text = parent_text[s_start:s_end].strip()
                sub_breadcrumb = f"{breadcrumb} > {sub_num}"

                chunks.append({
                    "leaf_id": f"{doc_prefix}chunk_{chunk_counter}",
                    "leaf_text": f"[{sub_breadcrumb}]\n{sub_text}",
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "breadcrumb": sub_breadcrumb,
                    "document_id": doc_id or "unknown"
                })
                chunk_counter += 1
        else:
            # Fallback: split by paragraphs within the large section
            paras = parent_text.split("\n\n")
            current_buf = []
            current_len = 0

            for p in paras:
                if current_len + len(p) > max_leaf_chars and current_buf:
                    leaf_body = "\n\n".join(current_buf)
                    chunks.append({
                        "leaf_id": f"{doc_prefix}chunk_{chunk_counter}",
                        "leaf_text": f"[{breadcrumb}]\n{leaf_body}",
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "breadcrumb": breadcrumb,
                        "document_id": doc_id or "unknown"
                    })
                    chunk_counter += 1
                    current_buf = []
                    current_len = 0
                current_buf.append(p)
                current_len += len(p)

            if current_buf:
                leaf_body = "\n\n".join(current_buf)
                chunks.append({
                    "leaf_id": f"{doc_prefix}chunk_{chunk_counter}",
                    "leaf_text": f"[{breadcrumb}]\n{leaf_body}",
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "breadcrumb": breadcrumb,
                    "document_id": doc_id or "unknown"
                })
                chunk_counter += 1

    return chunks, full_text


def parse_and_chunk_contract(
    pdf_path: str,
    doc_id: Optional[str] = None,
    max_leaf_chars: int = 1200,
    min_text_length: int = 500
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts text from PDF, builds hierarchical breadcrumbs, and chunks into Small-to-Big format.
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text("text") for page in doc])
    doc.close()

    return parse_and_chunk_text(
        full_text=full_text,
        doc_id=doc_id,
        max_leaf_chars=max_leaf_chars,
        min_text_length=min_text_length
    )
