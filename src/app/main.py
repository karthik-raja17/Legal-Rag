"""
FastAPI entrypoint for the Local Legal RAG Engine.

Exposes:
- /parse               : upload PDF, parse, and index directly into FAISS
- /status              : check document status
- /query               : local hybrid RAG question answering with Ollama & citations
- /health              : local health probe (FAISS & Ollama)
- /dropdown-options    : list distinct sites and documents for UI
- /api/chat            : frontend chat proxy
- /api/documents/count : total unique indexed documents
- /                    : web UI
"""
import asyncio
import logging
import os
import uuid
from contextvars import ContextVar
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.config.settings import settings
from src.core.indexer.faiss_client import FAISSClient
from src.core.indexer.indexer import Indexer
from src.core.embedding.local_embedder import LocalEmbedder
from src.core.llm.ollama_client import OllamaClient
from src.core.storage.local_storage import LocalStorageClient
from src.core.parser.pdf_parser import PDFParser
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.reranker import LocalReranker
from src.core.retrieval.query_expander import QueryExpander
from src.core.retrieval.query_analyzer import QueryAnalyzer
from src.core.retrieval.query_rewriter import QueryRewriter
from src.core.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion

# ----------------------------------------------------------------------------
# Logging with Request-ID Context
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
    title="Legal RAG Engine (Local)",
    description="Local French legal contract RAG engine with FAISS, all-MiniLM-L6-v2, and Ollama.",
    version="3.0.0",
)


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
_faiss_client: Optional[FAISSClient] = None
_embedder: Optional[LocalEmbedder] = None
_ollama_client: Optional[OllamaClient] = None
_local_storage: Optional[LocalStorageClient] = None
_indexer: Optional[Indexer] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_rewriter: Optional[QueryRewriter] = None
_query_expander: Optional[QueryExpander] = None
_query_analyzer: Optional[QueryAnalyzer] = None
_reranker: Optional[LocalReranker] = None
_parser: Optional[PDFParser] = None


def get_faiss_client() -> FAISSClient:
    global _faiss_client
    if _faiss_client is None:
        _faiss_client = FAISSClient()
    return _faiss_client


def get_embedder() -> LocalEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder()
    return _embedder


def get_ollama_client() -> OllamaClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def get_local_storage() -> LocalStorageClient:
    global _local_storage
    if _local_storage is None:
        _local_storage = LocalStorageClient()
    return _local_storage


def get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer(vector_client=get_faiss_client(), embedder=get_embedder())
    return _indexer


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(
            vector_client=get_faiss_client(),
            embedder=get_embedder(),
        )
    return _hybrid_retriever


def get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter(ollama_client=get_ollama_client())
    return _rewriter


def get_query_expander() -> QueryExpander:
    global _query_expander
    if _query_expander is None:
        _query_expander = QueryExpander(ollama_client=get_ollama_client())
    return _query_expander


def get_query_analyzer() -> QueryAnalyzer:
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer(ollama_client=get_ollama_client())
    return _query_analyzer


def get_reranker() -> LocalReranker:
    global _reranker
    if _reranker is None:
        _reranker = LocalReranker()
    return _reranker


def get_parser() -> PDFParser:
    global _parser
    if _parser is None:
        _parser = PDFParser(
            use_ocr=False,  # default to digital extraction locally; fallback available
            use_dedoc=bool(settings.DEDOC_SERVICE_URL),
            extract_tables=True,
            extract_figures=False,
            semantic_enrichment=True,
            language="fr",
            cache_dir=settings.PARSER_CACHE_DIR,
            dedoc_url=settings.DEDOC_SERVICE_URL,
        )
    return _parser


# ----------------------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    status = {"status": "healthy", "checks": {}}
    all_ok = True

    # 1. FAISS check
    try:
        faiss_ok = get_faiss_client().health_check()
        status["checks"]["faiss"] = "ok" if faiss_ok else "uninitialized"
        if not faiss_ok:
            all_ok = False
    except Exception as e:
        status["checks"]["faiss"] = f"error: {str(e)}"
        all_ok = False

    # 2. Ollama check
    try:
        ollama_ok = get_ollama_client().health_check()
        status["checks"]["ollama"] = "ok" if ollama_ok else "unreachable (check 'ollama serve')"
        if not ollama_ok:
            status["checks"]["ollama_hint"] = f"Ensure Ollama is running on {settings.OLLAMA_BASE_URL}"
    except Exception as e:
        status["checks"]["ollama"] = f"error: {str(e)}"

    # 3. Local Storage check
    try:
        storage = get_local_storage()
        status["checks"]["local_storage"] = "ok"
    except Exception as e:
        status["checks"]["local_storage"] = f"error: {str(e)}"
        all_ok = False

    if not all_ok:
        status["status"] = "degraded"
    return status


# ----------------------------------------------------------------------------
# Status Endpoint
# ----------------------------------------------------------------------------
@app.get("/status")
async def get_status(document_id: str = Query(..., description="Document ID to check")):
    storage = get_local_storage()
    doc_status = storage.get_status(document_id)
    if not doc_status:
        raise HTTPException(404, f"Document {document_id} not found.")
    return doc_status


# ----------------------------------------------------------------------------
# Dropdown Options Endpoint (for frontend)
# ----------------------------------------------------------------------------
@app.get("/dropdown-options")
@app.get("/api/dropdown-options")
async def get_dropdown_options():
    """
    Return distinct site_names and document_ids from FAISS metadata for frontend dropdowns.
    """
    try:
        faiss_client = get_faiss_client()
        chunks = faiss_client.get_all_chunks(limit=100000, include=["metadatas"])
        metadatas = chunks.get("metadatas", [])

        sites = set()
        documents = {}
        for meta in metadatas:
            doc_id = meta.get("document_id")
            site = meta.get("site_name")
            if doc_id:
                documents[doc_id] = {
                    "id": doc_id,
                    "label": doc_id,
                    "site_name": site or "Unknown",
                }
            if site and site not in ["SITE_NON_TROUVE", "unknown"]:
                sites.add(site)

        return {
            "sites": sorted(list(sites)),
            "documents": sorted(list(documents.values()), key=lambda x: x["label"]),
        }
    except Exception as e:
        logger.error(f"Failed to fetch dropdown options: {e}")
        return {"sites": [], "documents": []}


# ----------------------------------------------------------------------------
# Document Count Endpoint
# ----------------------------------------------------------------------------
@app.get("/api/documents/count")
async def get_documents_count():
    """Return the total number of unique documents indexed."""
    try:
        faiss_client = get_faiss_client()
        chunks = faiss_client.get_all_chunks(limit=100000, include=["metadatas"])
        doc_ids = {meta.get("document_id") for meta in chunks.get("metadatas", []) if meta.get("document_id")}
        return {"count": len(doc_ids)}
    except Exception as e:
        logger.error(f"Failed to count documents: {e}")
        return {"count": 0}


# ----------------------------------------------------------------------------
# Parse Endpoint – Upload PDF and Index Directly
# ----------------------------------------------------------------------------
@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    site_name: Optional[str] = Form(None),
    sync: bool = Query(False, description="If true, return full parsed JSON immediately."),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    doc_id = document_id or os.path.splitext(file.filename)[0]

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file {file.filename}: {e}")
        raise HTTPException(400, "Could not read uploaded file")

    storage = get_local_storage()
    storage.save_pdf(doc_id, content)
    storage.update_status(doc_id, "parsing", "Parsing PDF structure...")

    try:
        parser = get_parser()
        parsed_doc = await asyncio.to_thread(parser.parse, content, doc_id)
        storage.save_parsed_json(doc_id, parsed_doc.to_dict())
    except Exception as e:
        logger.error(f"Parsing failed for {doc_id}: {e}", exc_info=True)
        storage.update_status(doc_id, "failed", f"Parsing error: {str(e)}")
        raise HTTPException(500, f"Parsing error: {str(e)}")

    if sync:
        return JSONResponse(content=parsed_doc.to_dict())

    # Index directly into FAISS
    storage.update_status(doc_id, "indexing", "Indexing chunks into FAISS...")
    try:
        indexer = get_indexer()
        chunk_count = await asyncio.to_thread(indexer.index_document, parsed_doc, doc_id, site_name)
        storage.update_status(doc_id, "indexed", "Successfully indexed into FAISS", chunk_count=chunk_count)
    except Exception as e:
        logger.error(f"Indexing failed for {doc_id}: {e}", exc_info=True)
        storage.update_status(doc_id, "failed", f"Indexing error: {str(e)}")
        raise HTTPException(500, f"Indexing error: {str(e)}")

    return {
        "document_id": doc_id,
        "status": "indexed",
        "chunk_count": chunk_count,
        "message": f"Successfully parsed and indexed {chunk_count} chunks into FAISS.",
    }


# ----------------------------------------------------------------------------
# Local Context Window Expansion
# ----------------------------------------------------------------------------
async def expand_with_local_context(
    faiss_client: FAISSClient,
    final_chunks: List[Dict[str, Any]],
    top_k: int,
    window_size: int = 2,
) -> List[Dict[str, Any]]:
    expanded = []
    seen_families = set()

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

            # Fetch matching siblings from FAISS
            all_chunks = faiss_client.get_all_chunks(limit=10000, include=["documents", "metadatas"])
            siblings = []
            for cid, doc, meta in zip(
                all_chunks.get("ids", []),
                all_chunks.get("documents", []),
                all_chunks.get("metadatas", []),
            ):
                if meta.get("parent_section_id") == parent_id and start <= meta.get("part_number", 0) <= end:
                    siblings.append((cid, doc, meta))

            if siblings:
                siblings.sort(key=lambda x: x[2].get("part_number", 0))
                combined_text = "\n\n".join([doc for _, doc, _ in siblings])
                merged_metadata = siblings[0][2]

                expanded.append({
                    "id": chunk["id"],
                    "text": combined_text,
                    "metadata": merged_metadata,
                    "score": chunk.get("score", 1.0),
                    "context_window": f"{start}-{end}",
                })
                continue

        expanded.append(chunk)

    return expanded[:top_k]


# ----------------------------------------------------------------------------
# Query Endpoint – Local RAG
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
        # 1. Build metadata filter
        filter_metadata = {}
        if document_ids:
            filter_metadata["document_id"] = {"$in": document_ids}
        elif document_id:
            filter_metadata["document_id"] = document_id
        elif site_name:
            filter_metadata["site_name"] = site_name

        # 2. Query Rewriting (via Ollama)
        retrieval_query = query
        if settings.ENABLE_QUERY_REWRITING and rewrite:
            rewriter = get_rewriter()
            retrieval_query = await rewriter.rewrite(query)

        # 3. Query analysis (auto_optimize)
        if auto_optimize:
            analyzer = get_query_analyzer()
            analysis = analyzer.analyze(retrieval_query)
            if top_k is None:
                top_k = analysis.get("suggested_top_k", 5)
            if expand is None:
                expand = analysis.get("suggested_expand", False)
            if rerank is None:
                rerank = analysis.get("suggested_rerank", False)
        else:
            top_k = top_k or 5
            expand = expand or False
            rerank = rerank or False

        candidate_k = max(top_k * 2, 10)

        # 4. Query expansion
        queries_to_search = [retrieval_query]
        if expand:
            expander = get_query_expander()
            expanded_queries = await expander.expand(retrieval_query)
            if expanded_queries and len(expanded_queries) > 1:
                queries_to_search = expanded_queries

        # 5. Retrieval for each query variant
        all_result_sets = []
        retriever = get_hybrid_retriever()
        for q in queries_to_search:
            if hybrid:
                results = await retriever.hybrid_search(q, top_k=candidate_k, filter_metadata=filter_metadata)
            else:
                results = await retriever._dense_search(q, top_k=candidate_k, filter_metadata=filter_metadata)
            all_result_sets.append(results)

        # 6. Fuse variants via RRF if multiple
        if len(all_result_sets) > 1:
            fused_candidates = reciprocal_rank_fusion(
                all_result_sets,
                k=settings.HYBRID_RRF_K,
                weights=[1.0] * len(all_result_sets),
                merge_metadata_from="first",
            )
        else:
            fused_candidates = all_result_sets[0] if all_result_sets else []

        # 7. Local Reranking
        if rerank and fused_candidates:
            reranker_inst = get_reranker()
            final_chunks = await reranker_inst.rerank(query, fused_candidates, top_n=top_k)
        else:
            final_chunks = fused_candidates[:top_k]

        # 8. Local context expansion
        final_chunks = await expand_with_local_context(get_faiss_client(), final_chunks, top_k)

        # 9. Build output response dict
        response: Dict[str, Any] = {
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

        # 10. Generate answer with Ollama
        if generate and final_chunks:
            citation_map = {}
            context_parts = []
            for idx, chunk in enumerate(final_chunks, start=1):
                key = str(idx)
                citation_map[key] = {
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": chunk.get("score"),
                    "chunk_id": chunk.get("id"),
                }
                context_parts.append(f"[{key}] {chunk['text']}")

            context_str = "\n\n".join(context_parts)

            system_prompt = (
                "Tu es un assistant juridique expert en analyse de contrats.\n"
                "Réponds précisément et directement à la question de l'utilisateur en exploitant les clauses du contexte ci-dessous.\n"
                "Règles :\n"
                "1. Cites systématiquement le numéro de la clause source entre crochets (ex: [1], [2]).\n"
                "2. Précise les chiffres exacts (montants en euros, loyers, redevances, pourcentages, dates).\n"
                "3. Si la réponse n'existe pas dans le contexte, indique que le contrat ne contient pas d'information sur ce point."
            )

            prompt = f"Contexte du contrat :\n{context_str}\n\nQuestion : {query}\n\nRéponse :"

            ollama = get_ollama_client()
            try:
                answer = await ollama.agenerate(
                    prompt=prompt,
                    system=system_prompt,
                    temperature=0.1,
                )
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                answer = f"Erreur lors de la génération locale Ollama ({e}). Veuillez vérifier que le serveur Ollama est démarré."

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
# Frontend UI endpoints
# ----------------------------------------------------------------------------
if os.path.exists("src/static"):
    app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/")
async def root():
    if os.path.exists("src/static/index.html"):
        return FileResponse("src/static/index.html")
    return {"message": "Legal RAG Engine API is running. Access /docs for API documentation."}


@app.post("/api/chat")
async def api_chat(
    query: str = Body(..., embed=True),
    document_id: Optional[str] = Body(None),
    document_ids: Optional[List[str]] = Body(None),
    site_name: Optional[str] = Body(None),
    top_k: int = Body(5),
    generate: bool = Body(True),
    hybrid: bool = Body(True),
    rerank: bool = Body(False),
    expand: bool = Body(False),
    rewrite: bool = Body(True),
    auto_optimize: bool = Body(True),
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
    global _faiss_client
    if _faiss_client is not None:
        try:
            _faiss_client.close()
        except Exception as e:
            logger.warning(f"Error closing FAISS client: {e}")