"""
Semantic Search & Caching System — FastAPI Application

This is the main entry point. It orchestrates all services:
1. VectorDB: FAISS + sentence-transformers for semantic search
2. ClusteringService: GMM fuzzy clustering for soft assignments
3. SemanticCache: Cluster-aware cache for O(N/k) lookup

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import Dict
import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from services.vector_db import VectorDB
from services.clustering import ClusteringService
from services.cache import SemanticCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
vector_db = VectorDB()
clustering = ClusteringService()
cache = SemanticCache(threshold=0.85)


# --- Lifespan: Initialize all services on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all ML models and build indices during startup.
    This ensures the FAISS index, GMM model, and cache are all
    in memory before any requests are served.
    """
    logger.info("=" * 60)
    logger.info("STARTING: Semantic Search & Caching System")
    logger.info("=" * 60)

    start_time = time.time()

    # Phase 1: Load embedding model and build FAISS index
    vector_db.initialize()

    # Phase 2: Train GMM clustering on document embeddings
    clustering.fit(
        embeddings=vector_db.embeddings,
        targets=vector_db.targets,
        target_names=vector_db.target_names
    )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("READY in %.1f seconds", elapsed)
    logger.info("  Documents: %d", len(vector_db.documents))
    logger.info("  Dimensions: %d", vector_db.dimension)
    logger.info("  Clusters (k): %d", clustering.optimal_k)
    logger.info("  Cache threshold: %.2f", cache.threshold)
    logger.info("=" * 60)

    yield  # App is running

    logger.info("Shutting down Semantic Search & Caching System")


# --- FastAPI app ---
app = FastAPI(
    title="Semantic Search & Caching System",
    description="FAISS vector search with GMM clustering and cluster-aware semantic cache",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS so that Vercel frontend can call the Railway backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    """Request body for POST /query"""
    query: str


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the dashboard UI."""
    cluster_info = clustering.get_cluster_info()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cluster_info": cluster_info[:10],
        "optimal_k": clustering.optimal_k,
        "total_documents": len(vector_db.documents),
        "dimension": vector_db.dimension,
        "threshold": cache.threshold,
    })


@app.post("/query")
async def query_endpoint(req: QueryRequest) -> Dict:
    """
    POST /query — The core pipeline:

    1. Embed the query using sentence-transformers
    2. Predict cluster distribution via GMM
    3. Check the semantic cache (within the dominant cluster only)
    4. On MISS: query FAISS for top-k results, store in cache
    5. Return results with cache status and cluster info
    """
    start_time = time.perf_counter()

    # Step 1: Embed the query
    query_embedding = vector_db.embed_query(req.query)

    # Step 2: Predict cluster
    cluster_info = clustering.predict_cluster(query_embedding)
    dominant_cluster = cluster_info["dominant_cluster"]

    # Step 3: Check semantic cache (search dominant + neighboring clusters)
    # We search the top clusters from GMM to handle boundary cases where a
    # rephrased query might be assigned to a neighboring cluster
    search_clusters = [c["cluster_id"] for c in cluster_info["top_clusters"]]
    if dominant_cluster not in search_clusters:
        search_clusters.insert(0, dominant_cluster)
    cache_result = cache.lookup(query_embedding, search_clusters)

    if cache_result is not None:
        # CACHE HIT
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "query": req.query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity_score"],
            "result": cache_result["result"],
            "dominant_cluster": dominant_cluster,
            "cluster_info": cluster_info,
            "response_time_ms": elapsed_ms
        }
    else:
        # CACHE MISS — Query FAISS
        search_results = vector_db.search(req.query, top_k=5)
        result_data = search_results[0] if search_results else {"text": "No results found", "score": 0.0}

        # Store in cache for future queries
        cache.store(query_embedding, req.query, result_data, dominant_cluster)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "query": req.query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result_data,
            "dominant_cluster": dominant_cluster,
            "cluster_info": cluster_info,
            "all_results": search_results[:3],
            "response_time_ms": elapsed_ms
        }


@app.get("/cache/stats")
async def cache_stats() -> Dict:
    """
    GET /cache/stats — Return current cache statistics.

    Response includes total entries, hit/miss counts, hit rate,
    and per-cluster entry distribution.
    """
    stats = cache.get_stats()
    return stats


@app.delete("/cache")
async def clear_cache() -> Dict:
    """
    DELETE /cache — Flush the cache and reset all stats.

    Returns the pre-clear statistics for confirmation.
    """
    pre_stats = cache.clear()
    return {
        "message": "Cache cleared successfully",
        "previous_stats": pre_stats
    }


@app.get("/clusters")
async def get_clusters() -> Dict:
    """Get detailed cluster information and analysis."""
    return {
        "cluster_info": clustering.get_cluster_info(),
        "analysis": clustering.get_analysis_summary(),
        "boundary_cases": clustering.get_boundary_cases(max_cases=5)
    }


@app.get("/cache/entries")
async def get_cache_entries() -> Dict:
    """Get recent cache entries for the UI inspector."""
    return {
        "entries": cache.get_recent_entries(limit=10),
        "stats": cache.get_stats()
    }
