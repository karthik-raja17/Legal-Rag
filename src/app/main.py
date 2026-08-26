"""
FastAPI entrypoint for the Legal RAG Parser – production‑grade.

Exposes:
- /parse      : async upload + processing (Pub/Sub + GCS)
- /parse_from_gcs : parse a PDF already in GCS
- /status     : check indexing progress
- /query      : RAG question answering
- /health     : liveness + readiness probe
- /api/chat   : frontend proxy endpoint
- /           : frontend UI
- /dropdown-options : list of distinct site_names and document_ids for dropdowns
- /api/dropdown-options : same as above (for frontend)
- /api/documents/count : total number of unique documents
"""
import asyncio
import logging
import os
import tempfile
import functools
import uuid
from contextvars import ContextVar
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import firestore
from vertexai.generative_models import GenerativeModel

from src.core.indexer.chroma_client import ChromaClient
from src.core.parser.pdf_parser import PDFParser
from src.core.storage.gcs import GCSClient
from src.core.pubsub.publisher import Publisher
from src.config.settings import settings
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.reranker import VertexReranker
from src.core.retrieval.query_expander import QueryExpander
from src.core.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion
from src.core.retrieval.query_analyzer import QueryAnalyzer
from src.core.retrieval.query_rewriter import QueryRewriter
from src.core.embedding.bge_client import BGEEmbedderClient


class PublicBGEEmbedderClient(BGEEmbedderClient):
    def _get_headers(self) -> dict:
        return {"Content-Type": "application/json"}


# ----------------------------------------------------------------------------
# Logging with Request‑ID Context
# ----------------------------------------------------------------------------
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="unknown")

old_factory = logging.getLogRecordFactory()

def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.request_id = _request_id_ctx.get()
    return record

logging.setLogRecordFactory(record_factory)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s request_id=%(request_id)s %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Legal RAG Parser",
    description="Parses French legal contracts with OCR, structure, tables, and NLP.",
    version="2.0.0",
)

# ----------------------------------------------------------------------------
# Middleware – Request‑ID
# ----------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = _request_id_ctx.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        _request_id_ctx.reset(token)

# ----------------------------------------------------------------------------
# Lazy Singletons
# ----------------------------------------------------------------------------
_parser: Optional[PDFParser] = None
_gcs_client: Optional[GCSClient] = None
_publisher: Optional[Publisher] = None
_firestore_client: Optional[firestore.Client] = None
_chroma_client: Optional[ChromaClient] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_reranker: Optional[VertexReranker] = None
_query_expander: Optional[QueryExpander] = None
_query_analyzer: Optional[QueryAnalyzer] = None
_bge_client: Optional[BGEEmbedderClient] = None
_rewriter: Optional[QueryRewriter] = None


def get_bge_client() -> BGEEmbedderClient:
    global _bge_client
    if _bge_client is None:
        try:
            _bge_client = BGEEmbedderClient()
            logger.info("BGE embedder client initialized.")
        except Exception as e:
            logger.error(f"BGE client init failed: {e}", exc_info=True)
            raise HTTPException(503, "BGE embedder unavailable")
    return _bge_client


def get_rewriter() -> QueryRewriter:
    """Lazy singleton for query rewriter."""
    global _rewriter
    if _rewriter is None:
        try:
            _rewriter = QueryRewriter()
            logger.info("QueryRewriter initialized.")
        except Exception as e:
            logger.error(f"QueryRewriter init failed: {e}", exc_info=True)
            _rewriter = None
    return _rewriter


def get_query_expander() -> QueryExpander:
    global _query_expander
    if _query_expander is None:
        try:
            _query_expander = QueryExpander()
            logger.info("QueryExpander initialized.")
        except Exception as e:
            logger.error(f"QueryExpander init failed: {e}", exc_info=True)
            raise HTTPException(503, "Query expansion service unavailable")
    return _query_expander


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        try:
            chroma_client = get_chroma_client()
            _hybrid_retriever = HybridRetriever(chroma_client)
            logger.info("HybridRetriever initialized.")
        except Exception as e:
            logger.error(f"HybridRetriever init failed: {e}", exc_info=True)
            raise HTTPException(503, "Hybrid search service unavailable")
    return _hybrid_retriever


def get_parser() -> PDFParser:
    global _parser
    if _parser is None:
        try:
            _parser = PDFParser(
                use_ocr=True,
                use_dedoc=True,
                extract_tables=True,
                extract_figures=False,
                semantic_enrichment=True,
                language="fr",
                cache_dir=settings.PARSER_CACHE_DIR or None,
                dedoc_url=settings.DEDOC_SERVICE_URL,
            )
            logger.info("PDFParser initialized.")
        except Exception as e:
            logger.error(f"PDFParser init failed: {e}", exc_info=True)
            raise HTTPException(503, "Parser unavailable")
    return _parser


def get_gcs_client() -> GCSClient:
    global _gcs_client
    if _gcs_client is None:
        try:
            _gcs_client = GCSClient()
            logger.info("GCS client initialized.")
        except Exception as e:
            logger.error(f"GCS client init failed: {e}", exc_info=True)
            raise HTTPException(503, "Storage service unavailable")
    return _gcs_client


def get_publisher() -> Publisher:
    global _publisher
    if _publisher is None:
        try:
            _publisher = Publisher()
            logger.info("Pub/Sub publisher initialized.")
        except Exception as e:
            logger.error(f"Pub/Sub init failed: {e}", exc_info=True)
            raise HTTPException(503, "Messaging service unavailable")
    return _publisher


def get_firestore_client() -> firestore.Client:
    global _firestore_client
    if _firestore_client is None:
        try:
            _firestore_client = firestore.Client(project=settings.GCP_PROJECT_ID)
            logger.info("Firestore client initialized.")
        except Exception as e:
            logger.error(f"Firestore init failed: {e}", exc_info=True)
            raise HTTPException(503, "Database service unavailable")
    return _firestore_client


def get_chroma_client() -> ChromaClient:
    global _chroma_client
    if _chroma_client is None:
        try:
            _chroma_client = ChromaClient()
            logger.info("ChromaDB client initialized.")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}", exc_info=True)
            raise HTTPException(503, "Vector database unavailable")
    return _chroma_client


def get_reranker() -> VertexReranker:
    global _reranker
    if _reranker is None:
        try:
            _reranker = VertexReranker()
            logger.info("VertexReranker initialized.")
        except Exception as e:
            logger.error(f"VertexReranker init failed: {e}", exc_info=True)
            raise HTTPException(503, "Reranker service unavailable")
    return _reranker


# ----------------------------------------------------------------------------
# Core Parse Logic (shared between /parse and /parse_from_gcs)
# ----------------------------------------------------------------------------
async def parse_document_from_bytes(content: bytes, document_id: str, sync: bool = False) -> JSONResponse:
    """
    Core parsing logic – offloaded to a thread to avoid blocking the event loop.
    """
    try:
        parsed_result = await asyncio.to_thread(get_parser().parse, content, document_id)
    except Exception as e:
        logger.error(f"Parsing failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Parsing error: {str(e)}")

    if sync:
        return JSONResponse(content=parsed_result.to_dict())

    try:
        gcs = get_gcs_client()
        blob_name = f"parsed/{document_id}.json"
        gcs_uri = gcs.upload_json(parsed_result.to_dict(), blob_name)
        logger.info(f"Stored parsed JSON at {gcs_uri}")
    except Exception as e:
        logger.error(f"GCS upload failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(503, f"Storage error: {str(e)}")

    try:
        pub = get_publisher()
        msg_id = pub.publish(document_id, gcs_uri)
        logger.info(f"Published message {msg_id} for {document_id}")
    except Exception as e:
        logger.error(f"Pub/Sub publish failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(503, f"Messaging error: {str(e)}")

    return {
        "document_id": document_id,
        "status": "accepted",
        "gcs_uri": gcs_uri,
        "pubsub_message_id": msg_id,
        "message": "Document accepted for processing. Use /status to check indexing progress."
    }


# ----------------------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    status = {"status": "healthy", "project": settings.GCP_PROJECT_ID, "checks": {}}
    all_ok = True

    try:
        db = get_firestore_client()
        db.collection(settings.FIRESTORE_COLLECTION).document("_health_check").get(timeout=5)
        status["checks"]["firestore"] = "ok"
    except Exception as e:
        status["checks"]["firestore"] = f"error: {str(e)}"
        all_ok = False

    try:
        gcs = get_gcs_client()
        gcs.bucket.exists(timeout=5)
        status["checks"]["gcs"] = "ok"
    except Exception as e:
        status["checks"]["gcs"] = f"error: {str(e)}"
        all_ok = False

    try:
        chroma = get_chroma_client()
        chroma.client.heartbeat()
        status["checks"]["chromadb"] = "ok"
    except Exception as e:
        status["checks"]["chromadb"] = f"error: {str(e)}"
        all_ok = False

    if not all_ok:
        status["status"] = "degraded"
        raise HTTPException(status_code=503, detail=status)
    return status


# ----------------------------------------------------------------------------
# Status Endpoint
# ----------------------------------------------------------------------------
@app.get("/status")
async def get_status(document_id: str = Query(..., description="Document ID to check")):
    try:
        db = get_firestore_client()
        doc_ref = db.collection(settings.FIRESTORE_COLLECTION).document(document_id)
        doc = doc_ref.get(timeout=10)
        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(404, f"Document {document_id} not found in status index.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Firestore read failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(503, f"Database error: {str(e)}")


# ----------------------------------------------------------------------------
# Dropdown Options Endpoint (for frontend)
# ----------------------------------------------------------------------------
@app.get("/dropdown-options")
@app.get("/api/dropdown-options")
async def get_dropdown_options():
    """
    Return distinct site_names and document_ids for frontend dropdowns.
    """
    try:
        chroma = get_chroma_client()
        collection = chroma.client.get_collection(settings.CHROMA_COLLECTION)
        results = collection.get(include=["metadatas"], limit=100000)
        sites = set()
        documents = {}
        for meta in results.get("metadatas", []):
            doc_id = meta.get("document_id")
            site = meta.get("site_name")
            if doc_id:
                documents[doc_id] = {
                    "id": doc_id,
                    "label": doc_id,
                    "site_name": site or "Unknown"
                }
            if site and site not in ["SITE_NON_TROUVE", "unknown"]:
                sites.add(site)
        return {
            "sites": sorted(list(sites)),
            "documents": sorted(list(documents.values()), key=lambda x: x["label"])
        }
    except Exception as e:
        logger.error(f"Failed to fetch dropdown options: {e}")
        raise HTTPException(500, "Could not fetch dropdown options")


# ----------------------------------------------------------------------------
# Document Count Endpoint
# ----------------------------------------------------------------------------
@app.get("/api/documents/count")
async def get_documents_count():
    """
    Return the total number of unique documents in the collection.
    """
    try:
        chroma = get_chroma_client()
        collection = chroma.client.get_collection(settings.CHROMA_COLLECTION)
        results = collection.get(include=["metadatas"], limit=100000)
        doc_ids = set()
        for meta in results.get("metadatas", []):
            doc_id = meta.get("document_id")
            if doc_id:
                doc_ids.add(doc_id)
        return {"count": len(doc_ids)}
    except Exception as e:
        logger.error(f"Failed to fetch document count: {e}")
        raise HTTPException(500, "Could not fetch document count")


# ----------------------------------------------------------------------------
# Parse Endpoint – Upload PDF
# ----------------------------------------------------------------------------
@app.post("/parse", status_code=202)
async def parse_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form("unknown"),
    sync: bool = Query(False, description="If true, return JSON immediately (synchronous)."),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}", exc_info=True)
        raise HTTPException(400, "Could not read file")

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 20 MB)")

    result = await parse_document_from_bytes(content, document_id, sync)
    return result


# ----------------------------------------------------------------------------
# Parse Endpoint – PDF from GCS
# ----------------------------------------------------------------------------
@app.post("/parse_from_gcs", status_code=202)
async def parse_from_gcs(
    document_id: str = Form(...),
    gcs_uri: str = Form(...),
    sync: bool = Query(False),
):
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(400, "Invalid GCS URI")

    gcs = get_gcs_client()
    tmp_path = None

    try:
        blob_name = gcs_uri.replace(f"gs://{gcs.bucket_name}/", "")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            blob = gcs.bucket.blob(blob_name)
            blob.download_to_filename(tmp.name, timeout=60)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            content = f.read()

        return await parse_document_from_bytes(content, document_id, sync)

    except Exception as e:
        logger.error(f"Failed to read GCS file {gcs_uri}: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to read GCS file: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ----------------------------------------------------------------------------
# Dense Search Helper (pure dense retrieval using BGE)
# ----------------------------------------------------------------------------
async def dense_search(
    query: str,
    top_k: int,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Pure dense retrieval via ChromaDB using BGE embeddings.
    """
    loop = asyncio.get_running_loop()
    bge_client = get_bge_client()
    query_vector = await loop.run_in_executor(None, bge_client.embed_query, query)

    chroma = get_chroma_client()
    collection = chroma.client.get_collection(settings.CHROMA_COLLECTION)

    where_filter = filter_metadata if filter_metadata else None

    query_func = functools.partial(
        collection.query,
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances", "documents"],
    )
    raw_results = await loop.run_in_executor(None, query_func)

    candidates = []
    ids = raw_results.get("ids", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    documents = raw_results.get("documents", [[]])[0]

    for i in range(len(ids)):
        candidates.append({
            "id": ids[i],
            "text": documents[i] if i < len(documents) else "",
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": distances[i] if i < len(distances) else 1.0,
            "heading": metadatas[i].get("heading", "") if metadatas and i < len(metadatas) else "",
            "score": distances[i],
        })
    return candidates


async def expand_with_local_context(
    chroma_client: ChromaClient,
    final_chunks: List[Dict[str, Any]],
    top_k: int,
    window_size: int = 2
) -> List[Dict[str, Any]]:
    expanded = []
    seen_families = set()
    collection = chroma_client.client.get_collection(settings.CHROMA_COLLECTION)

    for chunk in final_chunks:
        metadata = chunk.get("metadata", {})
        parent_id = metadata.get("parent_section_id")
        is_part = metadata.get("is_part", False)
        part_num = metadata.get("part_number", 0)

        if is_part and parent_id and part_num > 0:
            family_key = f"{parent_id}_{part_num}"
            if family_key in seen_families:
                continue
            seen_families.add(family_key)

            start = max(1, part_num - window_size)
            end = part_num + window_size

            results = collection.get(
                where={
                    "parent_section_id": parent_id,
                    "part_number": {"$gte": start, "$lte": end}
                },
                include=["documents", "metadatas"]
            )

            if results.get("ids"):
                siblings = sorted(
                    zip(results["ids"], results["documents"], results["metadatas"]),
                    key=lambda x: x[2].get("part_number", 0)
                )
                combined_text = "\n\n".join([doc for _, doc, _ in siblings])
                merged_metadata = siblings[0][2]

                expanded.append({
                    "id": chunk["id"],
                    "text": combined_text,
                    "metadata": merged_metadata,
                    "score": chunk.get("score", 1.0),
                    "context_window": f"{start}-{end}"
                })
                continue

        expanded.append(chunk)

    return expanded[:top_k]


# ----------------------------------------------------------------------------
# Query Endpoint – RAG
# ----------------------------------------------------------------------------
@app.post("/query")
async def query_documents(
    query: str = Body(..., embed=True),
    document_id: Optional[str] = Body(None),
    document_ids: Optional[List[str]] = Body(None),
    site_name: Optional[str] = Body(None),
    top_k: Optional[int] = Body(None),
    generate: bool = Body(True),
    hybrid: bool = Body(True),
    rerank: Optional[bool] = Body(None),
    expand: Optional[bool] = Body(None),
    rewrite: Optional[bool] = Body(True),
    auto_optimize: bool = Body(True),
):
    try:
        # ---- Build filter_metadata ----
        filter_metadata = {}
        if document_ids:
            filter_metadata["document_id"] = {"$in": document_ids}
        elif document_id:
            filter_metadata["document_id"] = document_id
        elif site_name:
            filter_metadata["site_name"] = site_name
        # else: no filter (global search)

        # ---- Query Rewriting (if enabled) ----
        if settings.ENABLE_QUERY_REWRITING and rewrite:
            rewriter = get_rewriter()
            if rewriter:
                retrieval_query = await rewriter.rewrite(query)
                logger.info(f"Original query: '{query}' -> rewritten: '{retrieval_query}'")
            else:
                retrieval_query = query
                logger.warning("QueryRewriter unavailable, using original query.")
        else:
            retrieval_query = query

        # ---- Query analysis (auto_optimize) ----
        if auto_optimize:
            analyzer = QueryAnalyzer()
            analysis = analyzer.analyze(retrieval_query)
            logger.info(f"Query analysis: {analysis}")
            if top_k is None:
                top_k = analysis["suggested_top_k"]
            if expand is None:
                expand = analysis["suggested_expand"]
            if rerank is None:
                rerank = analysis["suggested_rerank"]
        else:
            if top_k is None:
                top_k = 5
            if expand is None:
                expand = False
            if rerank is None:
                rerank = True

        candidate_k = settings.VERTEX_RERANKER_CANDIDATE_K if rerank else top_k

        # ---- Query expansion (if enabled) ----
        queries_to_search = [retrieval_query]
        if expand:
            expander = get_query_expander()
            expanded = await expander.expand(retrieval_query)
            if expanded and len(expanded) > 1:
                queries_to_search = expanded
                logger.info(f"Expanded query into {len(queries_to_search)} variants")

        # ---- Retrieve for each variant ----
        all_result_sets = []
        for q in queries_to_search:
            if hybrid:
                retriever = get_hybrid_retriever()
                results = await retriever.hybrid_search(q, top_k=candidate_k, filter_metadata=filter_metadata)
            else:
                results = await dense_search(q, candidate_k, filter_metadata=filter_metadata)
            all_result_sets.append(results)

        # ---- Fuse variants ----
        if len(queries_to_search) > 1:
            fused_candidates = reciprocal_rank_fusion(
                all_result_sets,
                k=settings.HYBRID_RRF_K,
                weights=[1.0] * len(all_result_sets),
                merge_metadata_from="first"
            )
        else:
            fused_candidates = all_result_sets[0] if all_result_sets else []

        # ---- Rerank ----
        if rerank and fused_candidates:
            reranker = get_reranker()
            # Rerank uses the original query (not rewritten) for better cross‑encoder scoring
            final_chunks = await reranker.rerank(query, fused_candidates, top_n=top_k)
        else:
            final_chunks = fused_candidates[:top_k] if top_k else fused_candidates

        # ---- Local context expansion ----
        final_chunks = await expand_with_local_context(
            get_chroma_client(),
            final_chunks,
            top_k,
            window_size=2
        )

        # ---- Build response ----
        response = {
            "query": query,
            "retrieval_query": retrieval_query if retrieval_query != query else None,
            "document_id": document_id,
            "document_ids": document_ids,
            "site_name": site_name,
            "retrieved_chunks": final_chunks,
            "top_k": len(final_chunks),
            "hybrid": hybrid,
            "rerank": rerank,
            "expand": expand,
            "rewrite": rewrite,
        }
        if auto_optimize:
            response["analysis"] = analysis

        # ---- Generate answer with citations ----
        if generate and final_chunks:
            # Build citation map and context with numbered chunks
            citation_map = {}
            context_parts = []
            for idx, chunk in enumerate(final_chunks, start=1):
                key = str(idx)
                citation_map[key] = {
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": chunk.get("score"),
                    "rerank_score": chunk.get("rerank_score"),
                    "chunk_id": chunk.get("id"),
                }
                context_parts.append(f"[{key}] {chunk['text']}")

            context = "\n\n".join(context_parts)

            llm = GenerativeModel(settings.VERTEX_AI_LLM_MODEL)
            prompt = (
                f"Vous êtes un assistant juridique STRICT. Vous devez répondre UNIQUEMENT en utilisant le contexte fourni.\n\n"
                f"1. IDENTIFICATION: Si plusieurs clauses sont fournies, identifiez la clause qui répond PRÉCISÉMENT à la question. "
                f"Si une clause est générique (définition) et une autre est spécifique (pénalité), choisissez TOUJOURS la clause spécifique.\n"
                f"2. CITATION: Citez UNIQUEMENT la clause la plus pertinente. N'inventez pas de citations.\n"
                f"3. CHIFFRES EXACTS: Copiez les nombres, dates et montants TEXTUELLEMENT. "
                f"Si le contexte dit '500,000€', répondez '500,000€'. Si le contexte dit '30 jours', répondez '30 jours'. "
                f"Ne changez PAS la ponctuation (virgules, points) et ne supprimez PAS les symboles (€, $, %).\n"
                f"4. ABSENCE: Si la réponse n'est PAS explicitement mentionnée dans le contexte, répondez uniquement par : "
                f"'Le contrat ne contient pas d'information sur ce point.' N'inventez rien.\n\n"
                f"Contexte :\n{context}\n\nQuestion : {query}\n\nRéponse :"
            )
            response_obj = llm.generate_content(prompt, generation_config={"temperature": 0.2})
            answer = response_obj.text

            response["answer"] = answer
            response["citations"] = citation_map
        elif generate and not final_chunks:
            response["answer"] = "Aucun document pertinent trouvé pour répondre à cette question."
            response["citations"] = {}

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query error: {str(e)}")


# ----------------------------------------------------------------------------
# Frontend endpoints
# ----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("src/static/index.html")

@app.post("/api/chat")
async def api_chat(
    query: str = Body(..., embed=True),
    document_id: Optional[str] = Body(None),
    document_ids: Optional[List[str]] = Body(None),
    site_name: Optional[str] = Body(None),
    top_k: int = Body(5),
    generate: bool = Body(True),
    hybrid: bool = Body(True),
    rerank: bool = Body(True),
    expand: bool = Body(False),
    rewrite: bool = Body(True),
    auto_optimize: bool = Body(True)
):
    return await query_documents(
        query=query,
        document_id=document_id,
        document_ids=document_ids,
        site_name=site_name,
        top_k=top_k,
        generate=generate,
        hybrid=hybrid,
        rerank=rerank,
        expand=expand,
        rewrite=rewrite,
        auto_optimize=auto_optimize,
    )


# ----------------------------------------------------------------------------
# Graceful Shutdown
# ----------------------------------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global _parser, _gcs_client, _publisher, _firestore_client, _chroma_client, _bge_client, _rewriter
    for client in [_parser, _gcs_client, _publisher, _firestore_client, _chroma_client, _bge_client, _rewriter]:
        if client and hasattr(client, "close"):
            try:
                client.close()
                logger.info(f"Closed {client.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error closing {client.__class__.__name__}: {e}")
    logger.info("Shutdown complete.")