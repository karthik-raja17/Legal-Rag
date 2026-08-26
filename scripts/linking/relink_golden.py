import json
import chromadb
from difflib import SequenceMatcher

# Connect to ChromaDB
client = chromadb.HttpClient(host='10.200.0.2', port=8000)
collection = client.get_collection('legal_contracts')

def match_chunk(quote, doc_id):
    # Get all chunks for this document
    results = collection.get(where={"document_id": doc_id}, limit=10000)
    if not results['ids']:
        return None, 0.0
    best_id = None
    best_score = 0.0
    for i, text in enumerate(results['documents']):
        score = SequenceMatcher(None, quote, text).ratio()
        if score > best_score:
            best_score = score
            best_id = results['ids'][i]
    if best_score >= 0.8:
        return best_id, best_score
    return None, best_score

with open('golden.jsonl', 'r') as f:
    entries = [json.loads(line) for line in f]

linked_entries = []
for entry in entries:
    if entry.get('answer_quote') == 'ABSENT_DU_CONTRAT':
        entry['chunk_ids'] = []
        entry['chunk_match_details'] = []
        linked_entries.append(entry)
        continue
    doc_id = entry.get('doc_id', '').replace('.json', '')
    quote = entry['answer_quote']
    # For multi‑paragraph answers, split into sentences (simplified)
    fragments = [s.strip() for s in quote.split('\n') if s.strip()]
    if not fragments:
        fragments = [quote]
    matched_ids = []
    details = []
    for frag in fragments:
        cid, score = match_chunk(frag, doc_id)
        if cid:
            matched_ids.append(cid)
            details.append({"fragment": frag, "chunk_id": cid, "method": "fuzzy", "score": score})
        else:
            details.append({"fragment": frag, "chunk_id": None, "method": "unmatched", "score": score})
    entry['chunk_ids'] = matched_ids
    entry['chunk_match_details'] = details
    linked_entries.append(entry)

with open('golden_with_chunks.jsonl', 'w') as f:
    for entry in linked_entries:
        f.write(json.dumps(entry) + '\n')

print("Re‑linked golden dataset. Entries with chunks:", sum(1 for e in linked_entries if e['chunk_ids']))