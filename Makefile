.PHONY: build push deploy-parser deploy-indexer deploy-job test build-bge deploy-bge deploy-all

# ====== CONFIGURATION - Replace with your own values or export as env vars ======
# GCP_PROJECT_ID ?= your-gcp-project-id
# GCS_BUCKET_NAME ?= your-gcs-bucket-name
# EXCEL_FILE_ID ?= your-excel-file-id
# See .env.example for all required variables

REGISTRY ?= europe-west9-docker.pkg.dev/$(GCP_PROJECT_ID)/legal-rag-repo
SERVICE_ACCOUNT ?= legal-rag-sa@$(GCP_PROJECT_ID).iam.gserviceaccount.com

# Build and push the main parser/indexer image
build:
	# Build the image with the latest code (Dockerfile now in deploy/docker/)
	docker build -f deploy/docker/Dockerfile -t $(REGISTRY)/parser:latest .
	# Also tag it as indexer:latest for the indexer deployment
	docker tag $(REGISTRY)/parser:latest $(REGISTRY)/indexer:latest
	# Push both tags
	docker push $(REGISTRY)/parser:latest
	docker push $(REGISTRY)/indexer:latest

# Build and push the BGE embedder image (custom)
build-bge:
	docker build -f deploy/docker/Dockerfile.bge -t $(REGISTRY)/bge-embedder:latest .
	docker push $(REGISTRY)/bge-embedder:latest

# Deploy parser service (public) – production config with rewrite enabled
deploy-parser:
	gcloud run deploy legal-rag-parser \
		--image=$(REGISTRY)/parser:latest \
		--region=europe-west9 \
		--memory=8Gi \
		--cpu=2 \
		--cpu-boost \
		--concurrency=5 \
		--min-instances=0 \
		--max-instances=10 \
		--timeout=3600 \
		--service-account=$(SERVICE_ACCOUNT) \
		--network=default \
		--subnet=default \
		--vpc-egress=private-ranges-only \
		--set-env-vars "GCP_PROJECT_ID=$(GCP_PROJECT_ID)" \
		--set-env-vars "GCP_LOCATION=europe-west9" \
		--set-env-vars "DEDOC_SERVICE_URL=$(DEDOC_SERVICE_URL)" \
		--set-env-vars "CHROMA_HOST=$(CHROMA_HOST)" \
		--set-env-vars "CHROMA_PORT=8000" \
		--set-env-vars "CHROMA_COLLECTION=legal_contracts_bge" \
		--set-env-vars "BGE_EMBEDDER_URL=$(BGE_EMBEDDER_URL)" \
		--set-env-vars "GCS_BUCKET_NAME=$(GCS_BUCKET_NAME)" \
		--set-env-vars "FIRESTORE_COLLECTION=contract_state" \
		--set-env-vars "DOCUMENT_AI_PROCESSOR_ID=$(DOCUMENT_AI_PROCESSOR_ID)" \
		--set-env-vars "DOCUMENT_AI_LOCATION=eu" \
		--set-env-vars "VERTEX_AI_EMBEDDING_MODEL=text-multilingual-embedding-002" \
		--set-env-vars "VERTEX_AI_LLM_MODEL=gemini-2.5-flash" \
		--set-env-vars "MAX_CHUNK_TOKENS=512" \
		--set-env-vars "PARSER_CACHE_DIR=/tmp/parser_cache" \
		--set-env-vars "PUBSUB_TOPIC_ID=sync-requests" \
		--set-env-vars "HYBRID_RRF_K=60" \
		--set-env-vars "HYBRID_DENSE_WEIGHT=0.7" \
		--set-env-vars "HYBRID_BM25_WEIGHT=0.3" \
		--set-env-vars "HYBRID_TOP_K=5" \
		--set-env-vars "BM25_CACHE_TTL_SECONDS=86400" \
		--set-env-vars "HNSW_M=128" \
		--set-env-vars "HNSW_EF_CONSTRUCTION=1000" \
		--set-env-vars "HNSW_EF_SEARCH=2000" \
		--set-env-vars "ENABLE_QUERY_EXPANSION=false" \
		--set-env-vars "VERTEX_RERANKER_ENABLED=true" \
		--set-env-vars "VERTEX_RERANKER_LOCATION=global" \
		--set-env-vars "VERTEX_RERANKER_MODEL=semantic-ranker-512-004" \
		--set-env-vars "VERTEX_RERANKER_CANDIDATE_K=60" \
		--set-env-vars "VERTEX_RERANKER_TOP_N=5" \
		--set-env-vars "EMBEDDING_BATCH_SIZE=64" \
		--set-env-vars "FORCE_REBUILD=$$(date +%s)" \
		--set-env-vars "ENABLE_QUERY_REWRITING=true" \
		--set-env-vars "REWRITER_MODEL=gemini-2.5-flash" \
		--set-env-vars "REWRITER_TEMPERATURE=0.2" \
		--set-env-vars "REWRITER_CACHE_TTL_SECONDS=3600" \
		--set-env-vars "QUERY_ANALYZER_ENABLED=false" \
		--port=8080 \
		--allow-unauthenticated

# Deploy BGE embedder (private, only service account can invoke)
deploy-bge:
	gcloud run deploy legal-bge-embedder \
		--image=$(REGISTRY)/bge-embedder:latest \
		--region=europe-west9 \
		--cpu=4 \
		--memory=16Gi \
		--cpu-boost \
		--concurrency=16 \
		--min-instances=0 \
		--max-instances=3 \
		--port=8080 \
		--timeout=3600 \
		--service-account=$(SERVICE_ACCOUNT) \
		--no-allow-unauthenticated

# Deploy indexer (private, for Pub/Sub push)
deploy-indexer:
	gcloud run deploy legal-rag-indexer \
		--image=$(REGISTRY)/indexer:latest \
		--region=europe-west9 \
		--command="uvicorn" \
		--args="src.app.indexer_main:app,--host,0.0.0.0,--port,8080" \
		--set-env-vars "CHROMA_COLLECTION=legal_contracts_bge" \
		--set-env-vars "CHROMA_HOST=$(CHROMA_HOST)" \
		--set-env-vars "CHROMA_PORT=8000" \
		--set-env-vars "BGE_EMBEDDER_URL=$(BGE_EMBEDDER_URL)" \
		--set-env-vars "EMBEDDING_BATCH_SIZE=64" \
		--set-env-vars "GCP_PROJECT_ID=$(GCP_PROJECT_ID)" \
		--set-env-vars "GCP_LOCATION=europe-west9" \
		--set-env-vars "GCS_BUCKET_NAME=$(GCS_BUCKET_NAME)" \
		--set-env-vars "FIRESTORE_COLLECTION=contract_state" \
		--set-env-vars "MAX_CHUNK_TOKENS=512" \
		--set-env-vars "PUBSUB_TOPIC_ID=sync-requests" \
		--set-env-vars "PUBSUB_SUBSCRIPTION_ID=sync-subscription" \
		--set-env-vars "HNSW_M=128" \
		--set-env-vars "HNSW_EF_CONSTRUCTION=1000" \
		--set-env-vars "HNSW_EF_SEARCH=2000" \
		--set-env-vars "FORCE_REBUILD=$$(date +%s)" \
		--concurrency=1 \
		--max-instances=1 \
		--cpu=2 \
		--memory=4Gi \
		--cpu-boost \
		--no-allow-unauthenticated \
		--network=default \
		--subnet=default \
		--vpc-egress=private-ranges-only \
		--service-account=$(SERVICE_ACCOUNT)

# Deploy Dedoc service (public)
deploy-dedoc:
	gcloud run deploy dedoc-service \
		--image=$(REGISTRY)/dedoc:latest \
		--region=europe-west9 \
		--memory=4Gi \
		--cpu=2 \
		--concurrency=5 \
		--min-instances=0 \
		--max-instances=10 \
		--timeout=3600 \
		--allow-unauthenticated \
		--update-env-vars "DEDOC_TIMEOUT=3600"

# Deploy ingestion job (not active)
deploy-job:
	gcloud run jobs deploy legal-rag-ingestion \
		--image=$(REGISTRY)/parser:latest \
		--region=europe-west9 \
		--memory=2Gi \
		--cpu=1 \
		--task-timeout=600 \
		--service-account=$(SERVICE_ACCOUNT) \
		--command="python" \
		--args="-m,src.orchestrator.ingestor,--excel-file-id,$(EXCEL_FILE_ID),--concurrency,5" \
		--set-env-vars "PARSER_URL=$(PARSER_URL)" \
		--set-env-vars "GCS_BUCKET_NAME=$(GCS_BUCKET_NAME)"

# Execute the job (manual trigger)
run-job:
	gcloud run jobs execute legal-rag-ingestion --region=europe-west9

# Full deploy (all services)
deploy-all: build deploy-parser deploy-indexer deploy-bge deploy-dedoc

# Test locally (requires env vars)
test-local:
	EXCEL_FILE_ID=$(EXCEL_FILE_ID) \
	GOOGLE_APPLICATION_CREDENTIALS=sa-key.json \
	python -m src.orchestrator.ingestor --dry-run
