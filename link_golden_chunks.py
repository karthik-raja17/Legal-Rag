#!/usr/bin/env python3
"""
link_golden_chunks.py

Links each entry in golden.jsonl to the chunk_id(s) in ChromaDB whose
text contains (or closely matches) the entry's answer_quote.

Strategy:
  1. ABSENT_DU_CONTRAT -> chunk_ids = [] (nothing to link, by design)
  2. Split answer_quote into paragraph fragments (it may contain \n\n-separated
     excerpts from 2 different articles, per the cross-article questions).
  3. For each fragment, look only at chunks belonging to that doc_id
     (via metadata filter -> much faster and avoids cross-document false positives).
  4. Try exact (whitespace-normalized) substring match first.
  5. If no exact match, fall back to fuzzy matching (difflib) with a
     configurable similarity threshold, using a sliding window over the
     chunk text so partial containment still scores well.
  6. Record match method + score for auditability; log entries that need
     manual review.

Usage:
  python link_golden_chunks.py \
      --golden golden.jsonl \
      --output golden_with_chunks.jsonl \
      --chroma-host 10.200.0.2 \
      --chroma-port 8000 \
      --collection legal_contracts \
      --unmatched-report unmatched_report.jsonl
"""

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import chromadb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_FRAGMENT_LEN = 20          # skip fragments shorter than this (too ambiguous)
FUZZY_THRESHOLD = 0.80         # difflib similarity ratio to accept a fuzzy match
FUZZY_WINDOW_SLACK = 100        # extra chars of slack around the fragment length
                                # when building the sliding window over chunk text


def normalize_ws(text: str) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces, strip."""
    return re.sub(r"\s+", " ", text).strip()


def split_fragments(answer_quote: str) -> list[str]:
    """
    Split answer_quote on blank-line boundaries (paragraph breaks).
    This is what separates the two article excerpts in cross-article
    Q&As, and also naturally isolates table-row style quotes.
    """
    raw_parts = re.split(r"\n\s*\n", answer_quote)
    fragments = [normalize_ws(p) for p in raw_parts if normalize_ws(p)]
    # Filter out fragments that are too short to match reliably on their own
    return [f for f in fragments if len(f) >= MIN_FRAGMENT_LEN] or (
        [normalize_ws(answer_quote)] if normalize_ws(answer_quote) else []
    )


@dataclass
class FragmentMatch:
    fragment: str
    chunk_id: Optional[str] = None
    method: Optional[str] = None      # "exact" | "fuzzy" | "unmatched"
    score: float = 0.0


def best_fuzzy_match(fragment: str, chunk_texts: dict[str, str]) -> FragmentMatch:
    """
    Slide a window roughly the size of `fragment` across each chunk's
    normalized text and score with difflib.SequenceMatcher.
    Returns the best match found across all chunks for this doc.
    """
    best = FragmentMatch(fragment=fragment, method="unmatched", score=0.0)
    frag_len = len(fragment)
    window = frag_len + FUZZY_WINDOW_SLACK

    for cid, norm_text in chunk_texts.items():
        if len(norm_text) <= window:
            candidates = [norm_text]
        else:
            step = max(1, window // 2)
            candidates = [
                norm_text[i:i + window] for i in range(0, len(norm_text) - window + 1, step)
            ]
            candidates.append(norm_text[-window:])

        for cand in candidates:
            ratio = difflib.SequenceMatcher(None, fragment, cand).ratio()
            if ratio > best.score:
                best = FragmentMatch(fragment=fragment, chunk_id=cid, method="fuzzy", score=ratio)

    return best


def match_fragment(fragment: str, chunk_texts: dict[str, str]) -> FragmentMatch:
    # 1. Exact (normalized) substring match
    for cid, norm_text in chunk_texts.items():
        if fragment in norm_text:
            return FragmentMatch(fragment=fragment, chunk_id=cid, method="exact", score=1.0)

    # 2. Fuzzy fallback
    fuzzy = best_fuzzy_match(fragment, chunk_texts)
    if fuzzy.score >= FUZZY_THRESHOLD:
        return fuzzy

    # 3. Give up
    return FragmentMatch(fragment=fragment, chunk_id=None, method="unmatched", score=fuzzy.score)


def get_chunks_for_doc(collection, doc_id: str) -> dict[str, str]:
    """
    Fetch all chunks for a given document_id, keyed by chunk_id,
    with whitespace-normalized text.

    NOTE: adjust the `where` key ("document_id") if your indexer stores
    it under a different metadata field name (e.g. "doc_id", "source").
    """
    results = collection.get(
        where={"document_id": doc_id},
        include=["documents", "metadatas"],
    )
    out = {}
    for cid, text in zip(results["ids"], results["documents"]):
        if text:
            out[cid] = normalize_ws(text)
    return out


def process_entry(entry: dict, collection, doc_chunk_cache: dict) -> dict:
    entry = dict(entry)  # shallow copy

    if entry.get("answer_quote") == "ABSENT_DU_CONTRAT":
        entry["chunk_ids"] = []
        entry["chunk_match_details"] = []
        return entry

    doc_id = entry["doc_id"]
    if doc_id not in doc_chunk_cache:
        doc_chunk_cache[doc_id] = get_chunks_for_doc(collection, doc_id)
    chunk_texts = doc_chunk_cache[doc_id]

    if not chunk_texts:
        entry["chunk_ids"] = []
        entry["chunk_match_details"] = [{
            "fragment": entry["answer_quote"][:80],
            "chunk_id": None,
            "method": "no_chunks_for_doc",
            "score": 0.0,
        }]
        return entry

    fragments = split_fragments(entry["answer_quote"])
    match_results = [match_fragment(frag, chunk_texts) for frag in fragments]

    chunk_ids = []
    for m in match_results:
        if m.chunk_id and m.chunk_id not in chunk_ids:
            chunk_ids.append(m.chunk_id)

    entry["chunk_ids"] = chunk_ids
    entry["chunk_match_details"] = [
        {
            "fragment": m.fragment[:120],
            "chunk_id": m.chunk_id,
            "method": m.method,
            "score": round(m.score, 3),
        }
        for m in match_results
    ]
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="Path to golden.jsonl")
    ap.add_argument("--output", required=True, help="Path to write golden_with_chunks.jsonl")
    ap.add_argument("--chroma-host", default="10.200.0.2")
    ap.add_argument("--chroma-port", type=int, default=8000)
    ap.add_argument("--collection", default="legal_contracts")
    ap.add_argument("--unmatched-report", default="unmatched_report.jsonl")
    ap.add_argument("--doc-id-field", default="document_id",
                     help="Metadata field name in Chroma that stores the document id")
    args = ap.parse_args()

    global get_chunks_for_doc
    if args.doc_id_field != "document_id":
        def get_chunks_for_doc(collection, doc_id, _field=args.doc_id_field):
            results = collection.get(
                where={_field: doc_id},
                include=["documents", "metadatas"],
            )
            out = {}
            for cid, text in zip(results["ids"], results["documents"]):
                if text:
                    out[cid] = normalize_ws(text)
            return out

    print(f"Connecting to ChromaDB at {args.chroma_host}:{args.chroma_port} ...", file=sys.stderr)
    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    collection = client.get_collection(args.collection)

    with open(args.golden, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(entries)} golden entries.", file=sys.stderr)

    doc_chunk_cache: dict[str, dict[str, str]] = {}
    processed = []
    unmatched_report = []

    for i, entry in enumerate(entries, 1):
        result = process_entry(entry, collection, doc_chunk_cache)
        processed.append(result)

        if result["answer_quote"] != "ABSENT_DU_CONTRAT":
            expected_fragments = len(split_fragments(result["answer_quote"]))
            found = len(result["chunk_ids"])
            if found < expected_fragments or any(
                d["method"] in ("unmatched", "no_chunks_for_doc") or
                (d["method"] == "fuzzy" and d["score"] < 0.97)
                for d in result["chunk_match_details"]
            ):
                unmatched_report.append({
                    "index": i,
                    "doc_id": result["doc_id"],
                    "question": result["question"],
                    "section_reference": result.get("section_reference"),
                    "chunk_match_details": result["chunk_match_details"],
                })

        if i % 10 == 0 or i == len(entries):
            print(f"  processed {i}/{len(entries)}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(args.unmatched_report, "w", encoding="utf-8") as f:
        for item in unmatched_report:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_absent = sum(1 for e in processed if e["answer_quote"] == "ABSENT_DU_CONTRAT")
    n_full = sum(
        1 for e in processed
        if e["answer_quote"] != "ABSENT_DU_CONTRAT" and e["chunk_ids"]
        and all(d["method"] in ("exact", "fuzzy") for d in e["chunk_match_details"])
    )
    print("\n--- Summary ---", file=sys.stderr)
    print(f"Total entries:        {len(processed)}", file=sys.stderr)
    print(f"ABSENT_DU_CONTRAT:    {n_absent} (chunk_ids=[] by design)", file=sys.stderr)
    print(f"Fully matched:        {n_full}", file=sys.stderr)
    print(f"Flagged for review:   {len(unmatched_report)}  -> see {args.unmatched_report}", file=sys.stderr)
    print(f"\nOutput written to:    {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()