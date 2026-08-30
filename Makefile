.PHONY: run test ingest ingest-cuad eval-retrieval run-all clean

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
	python scripts/pipeline/ingest.py --pdf data/sample_contract.pdf
	python scripts/pipeline/index_documents.py

# Ingest all CUAD contracts across Part_I, Part_II, Part_III into storage & FAISS
ingest-cuad:
	python scripts/pipeline/ingest.py --cuad
	python scripts/pipeline/index_documents.py

# Run retrieval benchmark evaluation (Recall@1/5/10, Precision@1/5/10, MRR, nDCG, MAP)
eval-retrieval:
	python scripts/eval/evaluate_retrieval_metrics.py --mode hybrid --candidate-k 60 --top-k 10

# Master pipeline orchestrator (Ingest, Index, Benchmark & QA Demo)
run-all:
	python scripts/run_all.py --cuad --candidate-k 60 --top-k 10

# Clean caches and temporary build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
