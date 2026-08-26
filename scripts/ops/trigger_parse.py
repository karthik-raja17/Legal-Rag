import requests
import subprocess
import json

import os
BUCKET = os.getenv("GCS_BUCKET_NAME", "your-gcs-bucket-name")
PARSER_URL = os.getenv("PARSER_URL", "https://your-parser-service-url")
PREFIX = "pdfs/"

# List PDFs via gsutil
result = subprocess.run(
    ["gsutil", "ls", f"gs://{BUCKET}/{PREFIX}*.pdf"],
    capture_output=True, text=True
)
pdfs = [line.strip() for line in result.stdout.splitlines() if line.strip()]

for gcs_uri in pdfs:
    filename = gcs_uri.split('/')[-1]
    doc_id = filename.replace('.pdf', '')
    print(f"Triggering {doc_id} ...")
    data = {'document_id': doc_id, 'gcs_uri': gcs_uri}
    r = requests.post(f"{PARSER_URL}/parse_from_gcs", data=data)
    print(f"  → {r.status_code} {r.text[:50]}")