"""
Split CUAD dataset into train and test contract splits.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "cuad" / "annotations"
raw_json_path = base_dir / "CUAD_v1.json"

if not raw_json_path.exists():
    print(f"File not found: {raw_json_path}")
    exit(1)

# 1. Load the raw JSON
with open(raw_json_path, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# 2. Group paragraphs by document title
doc_groups = defaultdict(list)
for doc in full_data["data"]:
    doc_title = doc["title"]
    for paragraph in doc["paragraphs"]:
        doc_groups[doc_title].append(paragraph)

# 3. Shuffle and split documents (80/20)
doc_titles = list(doc_groups.keys())
random.shuffle(doc_titles)

split_idx = int(0.8 * len(doc_titles))
train_docs = doc_titles[:split_idx]
test_docs = doc_titles[split_idx:]


# 4. Build the new splits
def build_split(doc_list):
    return {
        "data": [{"title": t, "paragraphs": doc_groups[t]} for t in doc_list]
    }


train_data = build_split(train_docs)
test_data = build_split(test_docs)

# 5. Save to disk
train_path = base_dir / "train_cuad.json"
test_path = base_dir / "test_cuad.json"

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

# 6. Print summary
train_qas = sum(
    len(p["qas"]) for d in train_data["data"] for p in d["paragraphs"]
)
test_qas = sum(
    len(p["qas"]) for d in test_data["data"] for p in d["paragraphs"]
)

print(f"Train Contracts: {len(train_docs)}")
print(f"Test Contracts:  {len(test_docs)}")
print(f"Train QA pairs:  {train_qas}")
print(f"Test QA pairs:   {test_qas}")
print(f"Saved splits to {base_dir}")
