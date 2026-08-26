"""
Reciprocal Rank Fusion (RRF) utilities for merging multiple ranked result lists.

RRF is a rank‑based fusion method that combines multiple retrieval systems by
assigning a score based on the reciprocal of the rank. It works well without
relying on calibrated scores.

References:
    - Cormack et al. (2009) "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"
"""

import logging
from typing import List, Dict, Any, Optional, Sequence

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: Sequence[List[Dict[str, Any]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
    normalize_weights: bool = True,
    top_n: Optional[int] = None,
    id_key: str = "id",
    merge_metadata_from: str = "first",  # "first" | "highest" | "all"
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Args:
        result_lists: A sequence of ranked result lists. Each list is a list of
                      dicts, each containing at least an `id_key` field.
                      Example: [ [{"id": "doc1"}, {"id": "doc2"}], ... ]
        k: RRF smoothing constant (typically 60). Larger values reduce the
           influence of rank differences.
        weights: Optional list of weights for each result list. If None, all are
                 equal. If provided, must have the same length as `result_lists`.
        normalize_weights: If True, weights are normalized to sum to 1.0.
                           If False, raw weights are used.
        top_n: If provided, return only the top N results.
        id_key: The dictionary key used to identify each item.
        merge_metadata_from: How to merge metadata for items that appear in
                             multiple lists:
                             - "first": use the item from the first list where
                               it appears.
                             - "highest": use the item from the list where it
                               got the best rank (lowest rank number).
                             - "all": merge by copying `item` from the first
                               occurrence, then update with fields from later
                               occurrences (non‑destructive).

    Returns:
        A fused list of dicts, each with an added `score` field representing the
        RRF score. Items are sorted by score descending.

    Raises:
        ValueError: If input validation fails (e.g., empty lists, missing keys).
    """
    if not result_lists:
        return []

    # Validate input
    if not all(isinstance(lst, list) for lst in result_lists):
        raise ValueError("All result_lists elements must be lists.")

    # Validate weights
    num_lists = len(result_lists)
    if weights is not None:
        if len(weights) != num_lists:
            raise ValueError("weights length must match number of result lists.")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative.")
    else:
        weights = [1.0] * num_lists

    # Normalize weights if requested
    if normalize_weights:
        total = sum(weights)
        if total == 0:
            raise ValueError("Sum of weights is zero; cannot normalize.")
        weights = [w / total for w in weights]

    # Build rank maps: list of dict {id: rank (1-indexed)}
    rank_maps = []
    metadata_maps = []  # list of dict {id: item}
    for results in result_lists:
        rank_map = {}
        meta_map = {}
        for rank, item in enumerate(results, start=1):
            # Ensure item has the ID key
            if id_key not in item:
                raise ValueError(f"Item missing '{id_key}' key: {item}")
            doc_id = item[id_key]
            rank_map[doc_id] = rank
            meta_map[doc_id] = item
        rank_maps.append(rank_map)
        metadata_maps.append(meta_map)

    # Collect all IDs
    all_ids = set()
    for rank_map in rank_maps:
        all_ids.update(rank_map.keys())

    if not all_ids:
        return []

    # Compute RRF scores
    fused_scores = {}
    for doc_id in all_ids:
        score = 0.0
        for idx, rank_map in enumerate(rank_maps):
            if doc_id in rank_map:
                score += weights[idx] / (k + rank_map[doc_id])
        fused_scores[doc_id] = score

    # Sort by score descending
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    # Merge metadata according to strategy
    def _get_merged_item(doc_id: str) -> Dict[str, Any]:
        if merge_metadata_from == "first":
            # Use the first list where the item appears
            for meta_map in metadata_maps:
                if doc_id in meta_map:
                    return meta_map[doc_id].copy()
        elif merge_metadata_from == "highest":
            # Find the list where this item has the highest rank (smallest rank number)
            best_idx = -1
            best_rank = float("inf")
            for idx, rank_map in enumerate(rank_maps):
                if doc_id in rank_map and rank_map[doc_id] < best_rank:
                    best_rank = rank_map[doc_id]
                    best_idx = idx
            if best_idx != -1:
                return metadata_maps[best_idx][doc_id].copy()
        elif merge_metadata_from == "all":
            # Start with first occurrence, then update with others
            merged = {}
            for meta_map in metadata_maps:
                if doc_id in meta_map:
                    # Update with new fields, but don't overwrite existing ones unless they are None
                    item = meta_map[doc_id]
                    for key, value in item.items():
                        if key not in merged or merged[key] is None:
                            merged[key] = value
            return merged
        else:
            raise ValueError(f"Invalid merge_metadata_from: {merge_metadata_from}")

        # Fallback (should not happen)
        return {id_key: doc_id}

    # Build final list
    final_results = []
    for doc_id in sorted_ids:
        item = _get_merged_item(doc_id)
        item["score"] = fused_scores[doc_id]
        final_results.append(item)

    # Apply top_n
    if top_n is not None and top_n > 0:
        final_results = final_results[:top_n]

    logger.debug(f"RRF fused {len(all_ids)} unique items from {num_lists} lists.")
    return final_results