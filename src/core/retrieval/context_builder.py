"""
Retrieval Assembly (Small-to-Big Context Expansion)
Deduplicates parent sections from retrieved leaf chunks and fetches the full parent text
from the LocalDocStore so the LLM gets clean, comprehensive, unfragmented legal context.
"""
import logging
from typing import List, Dict, Any, Optional
from src.core.docstore import LocalDocStore

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Assembles LLM context using Small-to-Big retrieval.
    Expands retrieved leaf chunks to their full parent sections via LocalDocStore.
    """

    def __init__(self, docstore: Optional[LocalDocStore] = None):
        self.docstore = docstore or LocalDocStore()

    def assemble_context(self, hits: List[Dict[str, Any]]) -> str:
        """
        Deduplicates parent sections so the LLM gets clean, structured context.
        Falls back to leaf chunk text if parent_text is not found.
        """
        if not hits:
            return ""

        unique_parents: Dict[str, Dict[str, Any]] = {}
        fallback_contexts: List[str] = []

        for idx, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata", {}) or {}
            parent_id = metadata.get("parent_id")
            breadcrumb = metadata.get("breadcrumb", f"Section {idx}")
            chunk_text = hit.get("text", "")

            if parent_id:
                if parent_id not in unique_parents:
                    unique_parents[parent_id] = {
                        "breadcrumb": breadcrumb,
                        "hit_index": idx,
                        "leaf_text": chunk_text
                    }
            else:
                fallback_contexts.append(f"--- CONTEXT [{idx}]: {breadcrumb} ---\n{chunk_text}")

        # Fetch parent texts from DocStore
        full_context_blocks = []
        if unique_parents:
            parent_ids = list(unique_parents.keys())
            parent_texts = self.docstore.get_batch(parent_ids)

            for parent_id, info in unique_parents.items():
                p_text = parent_texts.get(parent_id)
                breadcrumb = info["breadcrumb"]
                idx = info["hit_index"]

                if p_text:
                    full_context_blocks.append(f"--- CONTEXT [{idx}]: {breadcrumb} ---\n{p_text}")
                else:
                    full_context_blocks.append(f"--- CONTEXT [{idx}]: {breadcrumb} ---\n{info['leaf_text']}")

        # --- FIX: Inject Global Context (Preamble & Definitions) ---
        global_context_blocks = []
        
        # Determine document id prefix from hits if present
        doc_id = None
        for h in hits:
            meta = h.get("metadata", {}) or {}
            if meta.get("document_id"):
                doc_id = meta.get("document_id")
                break

        sec_0_candidates = [f"{doc_id}_sec_0", "sec_0"] if doc_id else ["sec_0"]
        sec_1_candidates = [f"{doc_id}_sec_1", "sec_1"] if doc_id else ["sec_1"]

        # 1. Always try to add Preamble (sec_0) for context
        if not any("PREAMBLE" in info.get("breadcrumb", "") for info in unique_parents.values()):
            preamble_text = None
            for k in sec_0_candidates:
                preamble_text = self.docstore.get(k)
                if preamble_text:
                    break
            if preamble_text:
                global_context_blocks.append(f"--- GLOBAL CONTEXT (Preamble) ---\n{preamble_text}")
        
        # 2. Scan hits for defined terms (ALL CAPS) to force-inject Definitions
        # Regex to catch defined terms like "COMPANY" or "EFFECTIVE DATE"
        import re
        defined_term_pattern = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b')
        for hit in hits:
            text = hit.get("text", "")
            if defined_term_pattern.search(text):
                # Fetch Section 1 (usually Definitions) from DocStore
                def_text = None
                for k in sec_1_candidates:
                    def_text = self.docstore.get(k)
                    if def_text:
                        break
                if def_text and not any("DEFINITIONS" in block for block in global_context_blocks):
                    global_context_blocks.append(f"--- GLOBAL CONTEXT (Definitions) ---\n{def_text}")
                break  # Only inject once
        
        # Prepend global blocks to the main context
        full_context_blocks = global_context_blocks + full_context_blocks

        all_blocks = full_context_blocks + fallback_contexts
        return "\n\n".join(all_blocks)


def assemble_context(hits: List[Dict[str, Any]], docstore: Optional[LocalDocStore] = None) -> str:
    """Convenience functional interface for assemble_context."""
    builder = ContextBuilder(docstore=docstore)
    return builder.assemble_context(hits)

