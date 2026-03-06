# Semantic Search & Caching System

A production-grade semantic search system built with FAISS, GMM fuzzy clustering, and a custom cluster-aware semantic cache, served via FastAPI.

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim dense vectors for semantic representation |
| **Vector Store** | FAISS (`IndexFlatIP`) | In-memory similarity search with cosine similarity |
| **Clustering** | Gaussian Mixture Model (scikit-learn) | Soft/fuzzy cluster assignments with probability distributions |
| **Cache** | Custom Python (no Redis) | Cluster-aware semantic cache with O(N/k) lookup |
| **API** | FastAPI + Uvicorn | Async REST API with interactive dashboard |
| **Dataset** | 20 Newsgroups (~18,800 docs) | Classic text classification corpus |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --host 127.0.0.1 --port 8000
```

The first startup will:
- Download the `all-MiniLM-L6-v2` model (~80MB)
- Download the 20 Newsgroups dataset
- Embed all documents (2-5 minutes on CPU)
- Train the GMM clustering model
- Launch the API server on http://127.0.0.1:8000

## API Endpoints

### `POST /query`
```json
{
    "query": "government policy social welfare"
}
```
Response:
```json
{
    "query": "government policy social welfare",
    "cache_hit": false,
    "matched_query": null,
    "similarity_score": null,
    "result": { "text": "...", "score": 0.72, "category": "talk.politics.misc" },
    "dominant_cluster": 7,
    "cluster_info": { "dominant_cluster": 7, "max_probability": 0.85, "top_clusters": [...] }
}
```

### `GET /cache/stats`
```json
{
    "total_entries": 42,
    "hit_count": 17,
    "miss_count": 25,
    "hit_rate": 0.405
}
```

### `DELETE /cache`
Flushes the cache and resets all statistics.

## Docker

```bash
docker-compose up --build
```

## Design Decisions

### Why FAISS with IndexFlatIP?
Inner product on L2-normalized vectors equals cosine similarity. This avoids the overhead of separate normalization during search while maintaining exact similarity scores.

### Why GMM over K-Means?
K-Means forces hard cluster assignments. A document about "gun legislation" belongs to both politics and firearms with varying degrees. GMM's `predict_proba()` provides this probabilistic membership.

### Why custom cache instead of Redis?
The problem statement explicitly requires a cache built from first principles. Our cache uses GMM cluster structure to partition entries, reducing lookup from O(N) to O(N/k).

### Cache Threshold (0.90)
- **0.98**: Near-identical rephrases only. High precision, low hit rate.
- **0.90** (default): Semantically equivalent queries. Balanced.
- **0.85**: Loosely related queries. High recall, risks off-topic results.
