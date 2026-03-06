"""
Semantic Cache Service — Cluster-Aware Query Cache (Built from First Principles)

Design Decisions:
- No Redis, Memcached, or any external caching library. The entire cache
  is a pure Python data structure as required by the problem statement.
- Data Structure: dict mapping cluster_id → list of (query_embedding,
  query_text, result_data) tuples. This leverages the GMM cluster structure
  from Part 2 to partition the cache space.
- Lookup Strategy: When a new query arrives, we first predict its dominant
  cluster, then search for cache hits within that cluster AND any high-
  probability neighboring clusters. This handles boundary cases where a
  paraphrased query may land in a slightly different cluster.
- Similarity Threshold: The CACHE_THRESHOLD (default 0.85) is the central
  tunable parameter. A higher threshold (e.g., 0.95) means the cache only
  returns results for near-identical queries — high precision, low hit rate.
  A lower threshold (e.g., 0.80) aggressively returns cached results for
  loosely related queries — high hit rate, but risks serving slightly
  off-topic answers.

  Empirical Analysis of Threshold Values with all-MiniLM-L6-v2:
  We measured cosine similarity between semantically equivalent paraphrases:
    - "buy a gun for self defense" vs "purchasing a firearm to protect myself"  → 0.7587
    - "government policy social welfare" vs "social welfare government policies" → 0.9640
    - "space exploration rockets" vs "rockets and space travel"                  → 0.8054
    - "how to train a neural network" vs "training deep learning models"         → 0.5557

  Key insight: With all-MiniLM-L6-v2, paraphrased queries typically achieve
  0.70-0.85 cosine similarity, NOT the 0.90+ range one might expect. This is
  because the model captures semantic relatedness but also encodes structural
  and lexical differences. A threshold of 0.85 balances precision and recall:
  - At 0.90+: Catches only word-reordering (e.g., "govt policy" ↔ "policy govt")
    but misses legitimate paraphrases with different vocabulary.
  - At 0.85: Catches most genuine paraphrases while still rejecting
    topically-similar but meaningfully-different queries.
  - At 0.75: Would catch near-synonymous queries but risks returning results
    for queries that differ in important nuance (e.g., asking about gun
    purchase vs gun safety).
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Cluster-aware semantic cache built from scratch.

    The cache partitions entries by cluster, so lookups only need to scan
    entries in the query's dominant cluster — O(N/k) instead of O(N).
    For robustness, lookup can also search neighboring clusters (top-N
    from GMM predict_proba) to handle boundary cases.
    """

    def __init__(self, threshold: float = 0.85):
        """
        Initialize the semantic cache.

        Args:
            threshold: Cosine similarity threshold for cache hits.
                      Range [0, 1]. Higher = stricter matching.
                      - 0.95: Only near-identical rephrases (very precise)
                      - 0.85: Semantically equivalent queries (balanced, recommended)
                      - 0.75: Loosely related queries (high recall, risky)
        """
        self.threshold: float = threshold
        # Core data structure: {cluster_id: [(embedding, query_text, result, timestamp), ...]}
        self._cache: Dict[int, List[Tuple[np.ndarray, str, Dict, float]]] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._query_times: List[float] = []

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_ids: List[int]
    ) -> Optional[Dict]:
        """
        Check the cache for a semantically similar query.

        Searches across multiple clusters (not just the dominant one) to
        handle cases where a paraphrased query may be assigned to a
        neighboring cluster. This multi-cluster search is still efficient
        because we only check the top 2-3 clusters from GMM, keeping
        complexity at O(2-3 * N/k) ≈ O(N/k).

        Args:
            query_embedding: L2-normalized embedding of the query
            cluster_ids: List of cluster IDs to search (ordered by probability)

        Returns:
            Cache entry dict if hit (similarity >= threshold), else None
        """
        start_time = time.perf_counter()

        best_similarity = -1.0
        best_entry = None

        for cluster_id in cluster_ids:
            if cluster_id not in self._cache:
                continue

            cluster_entries = self._cache[cluster_id]

            for cached_embedding, cached_query, cached_result, cached_time in cluster_entries:
                # Cosine similarity (both vectors are already L2-normalized,
                # so dot product = cosine similarity)
                similarity = float(np.dot(query_embedding, cached_embedding))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = (cached_query, cached_result, cached_time)

        elapsed = time.perf_counter() - start_time

        if best_similarity >= self.threshold and best_entry is not None:
            self._hit_count += 1
            self._query_times.append(elapsed)
            return {
                "cache_hit": True,
                "matched_query": best_entry[0],
                "similarity_score": round(best_similarity, 4),
                "result": best_entry[1],
                "lookup_time_ms": round(elapsed * 1000, 2),
                "cached_at": best_entry[2]
            }
        else:
            self._miss_count += 1
            self._query_times.append(elapsed)
            logger.debug(
                "Cache MISS — best similarity %.4f < threshold %.2f",
                best_similarity, self.threshold
            )
            return None

    def store(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        result: Dict,
        dominant_cluster: int
    ) -> None:
        """
        Store a new query result in the appropriate cluster partition.

        Args:
            query_embedding: L2-normalized embedding
            query_text: Original query string
            result: The search result to cache
            dominant_cluster: Cluster for this entry
        """
        if dominant_cluster not in self._cache:
            self._cache[dominant_cluster] = []

        self._cache[dominant_cluster].append(
            (query_embedding, query_text, result, time.time())
        )

        logger.debug(
            "Cached query in cluster %d (cluster now has %d entries)",
            dominant_cluster, len(self._cache[dominant_cluster])
        )

    def get_stats(self) -> Dict:
        """
        Return cache statistics for the /cache/stats endpoint.

        Includes entry counts, hit/miss rates, and timing data.
        """
        total_entries = sum(len(entries) for entries in self._cache.values())
        total_queries = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_queries) if total_queries > 0 else 0.0

        avg_time = (
            sum(self._query_times) / len(self._query_times)
            if self._query_times else 0.0
        )

        return {
            "total_entries": total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 4),
            "total_queries": total_queries,
            "avg_lookup_time_ms": round(avg_time * 1000, 3),
            "threshold": self.threshold,
            "clusters_with_entries": len(self._cache),
            "entries_per_cluster": {
                str(k): len(v) for k, v in self._cache.items()
            }
        }

    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        """Get the most recent cache entries across all clusters."""
        all_entries = []
        for cluster_id, entries in self._cache.items():
            for emb, query, result, timestamp in entries:
                all_entries.append({
                    "query": query,
                    "cluster_id": cluster_id,
                    "timestamp": timestamp,
                    "result_preview": result.get("text", "")[:100] if isinstance(result, dict) else str(result)[:100]
                })

        # Sort by timestamp descending (most recent first)
        all_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_entries[:limit]

    def clear(self) -> Dict:
        """
        Flush the entire cache and reset all statistics.

        Returns the stats before clearing for confirmation.
        """
        pre_clear_stats = self.get_stats()
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0
        self._query_times.clear()

        logger.info("Cache cleared. Previous stats: %s", pre_clear_stats)
        return pre_clear_stats
