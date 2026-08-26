import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional

# ===================== CONFIGURATION =====================
import os
PARSER_URL = os.getenv("PARSER_URL", "https://your-parser-service-url/query")
GOLDEN_PATH = os.getenv("GOLDEN_PATH", "data/golden_with_chunks_bge.jsonl")
MAPPING_PATH = os.getenv("MAPPING_PATH", "data/doc_id_to_site.json")

# ===================== LOAD DATA =====================
def load_golden(path: str = GOLDEN_PATH) -> List[Dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def load_site_mapping(path: str = MAPPING_PATH) -> Dict[str, str]:
    with open(path, 'r') as f:
        return json.load(f)

# ===================== API CALL =====================
def query_parser(question: str, config: Dict[str, Any],
                 document_id: Optional[str] = None,
                 site_name: Optional[str] = None) -> Dict:
    payload = {
        "query": question,
        "generate": False,
        "hybrid": config.get("hybrid", True),
        "rerank": config.get("rerank", False),
        "expand": config.get("expand", False),
        "rewrite": config.get("rewrite", False),
        "auto_optimize": config.get("auto_optimize", False),
        "top_k": config.get("top_k", 10),
    }
    if document_id:
        payload["document_id"] = document_id
    elif site_name:
        payload["site_name"] = site_name

    payload = {k: v for k, v in payload.items() if v is not None}
    resp = requests.post(PARSER_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

# ===================== METRIC CALCULATIONS =====================
def dcg_at_k(relevances: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / np.log2(i + 1)
    return dcg

def ndcg_at_k(relevances: List[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal_rel = sorted(relevances, reverse=True)[:k]
    idcg = dcg_at_k(ideal_rel, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# ===================== EVALUATION ENGINE =====================
def evaluate_mode(mode: str, config: Dict[str, Any], gold_entries: List[Dict], site_mapping: Dict[str, str]) -> Dict:
    recalls = {k: [] for k in [1, 3, 5, 10]}
    precisions = {k: [] for k in [1, 3, 5, 10]}
    ndcgs = {k: [] for k in [1, 3, 5, 10]}
    reciprocal_ranks = []
    avg_precisions = []
    log = []

    for entry in gold_entries:
        gt_ids = set(entry.get("chunk_ids", []))
        if not gt_ids:
            continue

        question = entry["question"]
        doc_id = entry.get("doc_id")

        filter_doc_id = None
        filter_site_name = None

        if mode == "Doc":
            filter_doc_id = doc_id
        elif mode == "Site":
            if doc_id in site_mapping:
                filter_site_name = site_mapping[doc_id]
            else:
                print(f"⚠️ No site mapping found for doc_id: {doc_id}")
                filter_doc_id = doc_id

        try:
            result = query_parser(question, config,
                                  document_id=filter_doc_id,
                                  site_name=filter_site_name)
        except Exception as e:
            print(f"❌ Error: {question[:50]}... -> {e}")
            continue

        retrieved = [chunk["id"] for chunk in result.get("retrieved_chunks", [])]
        relevances = [1 if cid in gt_ids else 0 for cid in retrieved]

        rank = None
        for i, rel in enumerate(relevances, start=1):
            if rel == 1:
                rank = i
                break
        reciprocal_ranks.append(1 / rank if rank else 0)

        relevant_seen = 0
        ap = 0.0
        for i, rel in enumerate(relevances, start=1):
            if rel == 1:
                relevant_seen += 1
                ap += relevant_seen / i
        ap = ap / len(gt_ids) if gt_ids else 0.0
        avg_precisions.append(ap)

        for k in recalls.keys():
            hits = sum(relevances[:k])
            recalls[k].append(hits / len(gt_ids))
            precisions[k].append(hits / k)
            ndcgs[k].append(ndcg_at_k(relevances, k))

        log.append({
            "question": question[:100],
            "mode": mode,
            "doc_id": doc_id,
            "site_name": filter_site_name,
            "gt_count": len(gt_ids),
            "rank": rank,
            "ap": ap,
        })

    return {
        "mode": mode,
        "config": config.get("name", "Unknown"),
        "num_queries": len(avg_precisions),
        "MRR": np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
        "MAP": np.mean(avg_precisions) if avg_precisions else 0,
        "Recall@1": np.mean(recalls[1]) if recalls[1] else 0,
        "Recall@3": np.mean(recalls[3]) if recalls[3] else 0,
        "Recall@5": np.mean(recalls[5]) if recalls[5] else 0,
        "Recall@10": np.mean(recalls[10]) if recalls[10] else 0,
        "Precision@1": np.mean(precisions[1]) if precisions[1] else 0,
        "Precision@3": np.mean(precisions[3]) if precisions[3] else 0,
        "Precision@5": np.mean(precisions[5]) if precisions[5] else 0,
        "Precision@10": np.mean(precisions[10]) if precisions[10] else 0,
        "nDCG@1": np.mean(ndcgs[1]) if ndcgs[1] else 0,
        "nDCG@3": np.mean(ndcgs[3]) if ndcgs[3] else 0,
        "nDCG@5": np.mean(ndcgs[5]) if ndcgs[5] else 0,
        "nDCG@10": np.mean(ndcgs[10]) if ndcgs[10] else 0,
    }

def print_table(results: List[Dict], title: str):
    print(f"\n{title}")
    print("=" * 110)
    print(f"{'Config':<30} {'R@1':<7} {'R@5':<7} {'R@10':<7} {'P@1':<7} {'P@5':<7} {'P@10':<7} {'MRR':<7} {'MAP':<7} {'nDCG@5':<8}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['config']:<30} "
            f"{r['Recall@1']:.4f} {r['Recall@5']:.4f} {r['Recall@10']:.4f} "
            f"{r['Precision@1']:.4f} {r['Precision@5']:.4f} {r['Precision@10']:.4f} "
            f"{r['MRR']:.4f} {r['MAP']:.4f} {r['nDCG@5']:.4f}"
        )

# ===================== MAIN =====================
def main():
    gold = load_golden()
    valid = [e for e in gold if e.get("chunk_ids") and len(e["chunk_ids"]) > 0]
    print(f"✅ Loaded {len(gold)} entries, {len(valid)} with ground-truth.")

    site_mapping = load_site_mapping()
    print(f"✅ Loaded site mapping for {len(site_mapping)} documents.")

    configs = [
        ("Hybrid+Rerank", {"name": "Hybrid+Rerank", "hybrid": True, "rerank": True, "expand": False, "rewrite": False, "auto_optimize": False, "top_k": 10}),
        ("Hybrid+Rerank+Rewrite", {"name": "Hybrid+Rerank+Rewrite", "hybrid": True, "rerank": True, "expand": False, "rewrite": True, "auto_optimize": False, "top_k": 10}),
    ]

    for mode_name in ["Site", "Doc"]:
        print(f"\n🔍 Running {mode_name} mode...")
        mode_results = []
        for name, cfg in configs:
            print(f"  - {name}...")
            metrics = evaluate_mode(mode_name, cfg, valid, site_mapping)
            mode_results.append(metrics)

        with open(f"results_{mode_name.lower()}.json", "w") as f:
            json.dump(mode_results, f, indent=2)

        print_table(mode_results, f"📊 TABLE: {mode_name.upper()} SEARCH")

        # Compare the two
        hr = next((x for x in mode_results if x["config"] == "Hybrid+Rerank"), None)
        hrw = next((x for x in mode_results if x["config"] == "Hybrid+Rerank+Rewrite"), None)

        print("\n" + "=" * 110)
        print(f"🚀 PERFORMANCE COMPARISON ({mode_name} mode)")
        print("=" * 110)
        if hr and hrw:
            print(f"{'Metric':<15} {'Hybrid+Rerank':<18} {'+Rewrite':<18} {'Gain':<10}")
            print("-" * 65)
            for metric in ["MRR", "MAP", "Recall@1", "Recall@5", "Recall@10", "Precision@1", "Precision@5", "Precision@10", "nDCG@5"]:
                base = hr.get(metric, 0)
                rw = hrw.get(metric, 0)
                gain = rw - base
                print(f"{metric:<15} {base:<18.4f} {rw:<18.4f} {gain:+.4f}")

            print(f"\n💡 INTERPRETATION ({mode_name} mode):")
            print(f"  - MRR (Hybrid+Rerank):           {hr['MRR']:.4f}")
            print(f"  - MRR (+Rewrite):                {hrw['MRR']:.4f}")

if __name__ == "__main__":
    main()