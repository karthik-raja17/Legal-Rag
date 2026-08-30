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

        all_blocks = full_context_blocks + fallback_contexts
        return "\n\n".join(all_blocks)


def assemble_context(hits: List[Dict[str, Any]], docstore: Optional[LocalDocStore] = None) -> str:
    """Convenience functional interface for assemble_context."""
    builder = ContextBuilder(docstore=docstore)
    return builder.assemble_context(hits)

