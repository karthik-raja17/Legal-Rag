.PHONY: run test ingest clean

# Run the FastAPI server locally
run:
	uvicorn src.app.main:app --host 0.0.0.0 --port 8080 --reload

# Run all local verification tests
test:
	python scripts/test_local_rag.py
	python scripts/test_fastapi_endpoints.py

# Ingest test contract into FAISS index
ingest:
	python scripts/create_test_pdf.py
	python scripts/local_ingest.py --pdf data/test_contract.pdf --doc-id bail_lentilly_01 --site Lentilly

# Clean caches and temporary build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
