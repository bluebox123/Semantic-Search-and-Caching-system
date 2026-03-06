"""
Vector Database Service — FAISS + Sentence-Transformers

Design Decisions:
- Model: all-MiniLM-L6-v2 (384-dim). Chosen because it offers excellent
  semantic quality while remaining lightweight enough to run on CPU without
  GPU dependencies. This is critical for a portable, reproducible submission.
- Index: faiss.IndexFlatIP (Inner Product). After L2-normalizing all vectors,
  inner product is mathematically equivalent to cosine similarity but avoids
  the per-query normalization overhead of IndexFlatL2 + post-processing.
- Data Cleaning: We strip headers, footers, and quotes from 20 Newsgroups
  to force the embedding model to encode pure semantic content rather than
  clustering on sender email domains or routing metadata.
- Filtering: Documents shorter than 20 characters are discarded — they carry
  insufficient semantic signal and would only pollute cluster assignments.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
import logging

logger = logging.getLogger(__name__)


class VectorDB:
    """FAISS-backed vector store for the 20 Newsgroups corpus."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name: str = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[str] = []
        self.target_names: List[str] = []
        self.targets: np.ndarray = np.array([])
        self.embeddings: Optional[np.ndarray] = None
        self.dimension: int = 384  # all-MiniLM-L6-v2 output dimension

    def initialize(self) -> None:
        """Load dataset, embed documents, and build FAISS index."""
        logger.info("Loading sentence-transformer model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

        logger.info("Fetching 20 Newsgroups dataset...")
        # Justification: Removing headers, footers, and quotes ensures
        # the model learns from the actual content, not metadata noise
        # like email addresses, routing info, or quoted reply chains.
        dataset = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes')
        )
        self.target_names = list(dataset.target_names)

        # Filter out empty or very short documents (< 20 chars).
        # These carry no meaningful semantic content and would only
        # degrade cluster quality and embedding usefulness.
        raw_docs = dataset.data
        raw_targets = dataset.target
        valid_indices = [
            i for i, doc in enumerate(raw_docs)
            if len(doc.strip()) >= 20
        ]
        self.documents = [raw_docs[i].strip() for i in valid_indices]
        self.targets = raw_targets[valid_indices]

        logger.info(
            "Kept %d / %d documents after filtering short/empty posts.",
            len(self.documents), len(raw_docs)
        )

        # Embed all documents
        logger.info("Embedding %d documents (this may take a few minutes)...", len(self.documents))
        self.embeddings = self.model.encode(
            self.documents,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True  # L2-normalize so IP == cosine similarity
        )
        self.embeddings = self.embeddings.astype(np.float32)

        # Build FAISS index (Inner Product on normalized vectors = cosine sim)
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

        logger.info(
            "FAISS index built: %d vectors of dimension %d",
            self.index.ntotal, self.dimension
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a normalized vector."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)
        return embedding[0]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search: embed the query and retrieve top-k similar documents.

        Returns a list of dicts with keys: index, score, text, category.
        """
        query_embedding = self.embed_query(query)
        query_embedding_2d = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding_2d, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "index": int(idx),
                "score": float(score),
                "text": self.documents[idx][:500],  # truncate for response
                "category": self.target_names[self.targets[idx]]
            })
        return results

    def get_document(self, idx: int) -> Dict:
        """Retrieve a single document by index."""
        return {
            "index": idx,
            "text": self.documents[idx],
            "category": self.target_names[self.targets[idx]]
        }

    def get_stats(self) -> Dict:
        """Return vector DB statistics."""
        return {
            "total_documents": len(self.documents),
            "vector_dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0,
            "model": self.model_name,
            "categories": self.target_names
        }
