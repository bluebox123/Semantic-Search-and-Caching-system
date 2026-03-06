# Project Execution Plan: Semantic Search & Caching System

## 1. Architecture & Technology Stack

To ensure the system is lightweight, fast, and easy to run, we will use the following stack:

Web Framework: FastAPI (Asynchronous, fast, and self-documenting).

Vector Store: FAISS (Facebook AI Similarity Search). It runs locally, requires no external database servers, and is incredibly fast for in-memory vector similarity search.

Embeddings: sentence-transformers/all-MiniLM-L6-v2. It is lightweight, fast to run on a CPU, and produces excellent semantic representations.

Clustering: Gaussian Mixture Models (GMM) via scikit-learn. Unlike K-Means, GMM provides probability distributions for cluster membership (soft clustering), satisfying the core requirement.

## 2. Phase-by-Phase Implementation Guide

### Phase 1: Environment Setup

Keep it clean and standard.

Create a virtual environment: python -m venv venv

Activate it: source venv/bin/activate (or venv\Scripts\activate on Windows)

Create a requirements.txt with:

Plaintext
fastapi
uvicorn
scikit-learn
sentence-transformers
faiss-cpu
numpy
pandas

### Phase 2: Embedding & Vector Database (Part 1)

The 20 Newsgroups dataset is notorious for metadata noise (headers, email addresses, routing info).

Data Cleaning: Use sklearn.datasets.fetch_20newsgroups(remove=('headers', 'footers', 'quotes')). Justification: Removing this metadata forces the embedding model to focus purely on the semantic content of the text, rather than clustering based on the email domain of the sender.

Vectorization: Pass the cleaned text through the all-MiniLM-L6-v2 model to get 384-dimensional dense vectors.

Vector Store: Initialize a faiss.IndexFlatIP (Inner Product, which is equivalent to Cosine Similarity if vectors are normalized). Add all document embeddings to this index.

### Phase 3: Fuzzy Clustering (Part 2)

We must avoid hard assignments and justify the number of clusters.

Algorithm Choice: Use Gaussian Mixture Models (GMM). Use gmm.predict_proba(embeddings) to get a distribution array for each document (e.g., 80% Politics, 15% Religion, 5% Misc).

Determining 'k' (Number of Clusters): Do not guess. Write a script to calculate the Bayesian Information Criterion (BIC) or Silhouette Scores for k ranging from 10 to 30. Choose the k where the BIC score minimizes or the "elbow" occurs. Justification: This proves you used a mathematical heuristic to find the optimal semantic groupings, not just the dataset's arbitrary 20 labels.

Boundary Analysis: Write a brief analysis in your comments or a Jupyter Notebook showing documents where the highest cluster probability is low (e.g., max probability < 40%). These are your boundary cases.

### Phase 4: The Semantic Cache (Part 3)

This is the core engineering challenge. We are building an intelligent, cluster-aware cache from scratch.

Data Structure: Create a custom Python class. Internally, use a dictionary mapping Cluster IDs to lists of cached queries: {cluster_id: [(query_embedding, result_text), ...]}.

The Logic (Using Clusters for Efficiency):

When a new query arrives, embed it.

Predict its dominant cluster using the trained GMM.

Crucial Step: Only search for cache hits within the cached items of that specific dominant cluster. Justification: As the cache scales to millions of entries, doing a brute-force cosine similarity against the entire cache becomes a bottleneck. Grouping cache entries by cluster reduces the search space drastically (O(N/k) instead of O(N)).

The Tunable Parameter (Similarity Threshold): Define a CACHE_THRESHOLD (e.g., 0.90). If the cosine similarity between the incoming query embedding and a cached embedding in the same cluster is > 0.90, it's a hit. Exploration: In your documentation, explain that a higher threshold (0.98) prioritizes precision (fewer false hits, but lower cache hit rate), while a lower threshold (0.85) saves compute but risks returning slightly off-topic cached answers.

### Phase 5: FastAPI Service (Part 4)

Structure the app using clean routing.

State Management: Load the FAISS index, the GMM model, and initialize your custom Cache class during FastAPI's lifespan startup event so they stay in memory.

Endpoints:

POST /query: Implement the flow: Embed -> Predict Cluster -> Check Cache -> (If Miss) Query FAISS -> Update Cache -> Return JSON.

GET /cache/stats: Return lengths of dictionaries and tracking variables (hits / (hits + misses)).

DELETE /cache: Simply re-instantiate your Cache class to clear it.

### Phase 6: Dockerisation (Bonus)

A Dockerfile makes this undeniably professional.

Use a slim Python base image.

Pre-download the HuggingFace model during the Docker build process so the container doesn't have to download it every time it starts.

Expose port 8000.

Set the entrypoint to:

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"].

## How to Make This "Resume-Worthy"

Object-Oriented Design: Don't put everything in one main.py file. Create a services/ directory with vector_db.py, clustering.py, and cache.py.

Type Hinting: Use strict Python type hints (List, Dict, Optional, Pydantic models) everywhere.

Docstrings: Every function should have a brief explanation of why it does what it does, fulfilling the assignment's requirement to justify your choices.
