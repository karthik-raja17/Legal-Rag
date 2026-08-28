.PHONY: run test ingest ingest-cuad clean

# Run the FastAPI server locally
run:
	uvicorn src.app.main:app --host 0.0.0.0 --port 8080 --reload

# Run all local verification tests
test:
	python scripts/test_local_rag.py
	python scripts/test_fastapi_endpoints.py

# Ingest sample contract into FAISS index
ingest:
	python scripts/create_test_pdf.py
	python scripts/local_ingest.py --pdf data/sample_contract.pdf --doc-id msa_acme_01 --site "Delaware Headquarters"

# Ingest all CUAD contracts in data/cuad/pdfs
ingest-cuad:
	python scripts/local_ingest.py --dir data/cuad/pdfs/Part_I/Affiliate_Agreements --site "Affiliate Agreements"

# Clean caches and temporary build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
