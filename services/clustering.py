"""
Clustering Service — Gaussian Mixture Model (GMM) Fuzzy Clustering

Design Decisions:
- Algorithm: GMM instead of K-Means. GMM provides soft/fuzzy cluster
  assignments via predict_proba(), giving a probability distribution per
  document rather than a hard label. This is essential because documents
  like "gun legislation" genuinely belong to multiple categories (politics,
  firearms) with different degrees of membership.
- Determining k: We evaluate BIC (Bayesian Information Criterion) scores
  for k in [10, 30]. Lower BIC indicates a better model fit while penalizing
  excessive complexity. We also compute silhouette scores as a secondary
  validation metric. The optimal k is where BIC is minimized.
- Covariance: 'diag' covariance type balances expressiveness with
  computational feasibility on 384-dimensional data. Full covariance
  would require O(d²) parameters per component, which is prohibitive.
- Boundary Analysis: Documents with max cluster probability < 0.40 are
  flagged as boundary cases — these are semantically ambiguous documents
  that sit between clusters and are often the most interesting.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class ClusteringService:
    """GMM-based fuzzy clustering for document embeddings."""

    def __init__(self):
        self.gmm: Optional[GaussianMixture] = None
        self.optimal_k: int = 0
        self.cluster_probabilities: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.bic_scores: Dict[int, float] = {}
        self.silhouette_scores: Dict[int, float] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.target_names: List[str] = []
        self.targets: np.ndarray = np.array([])

    def find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_range: Tuple[int, int] = (10, 25)
    ) -> int:
        """
        Determine the optimal number of clusters using BIC scores.

        BIC (Bayesian Information Criterion) penalizes model complexity
        while rewarding goodness of fit. Lower BIC = better model.
        We test k values in [k_min, k_max] and pick the minimizer.

        We also compute silhouette scores as a secondary signal, but
        BIC is the primary criterion because silhouette can be unreliable
        in high-dimensional spaces.
        """
        k_min, k_max = k_range
        logger.info("Evaluating BIC scores for k in [%d, %d]...", k_min, k_max)

        best_bic = float('inf')
        best_k = k_min

        for k in range(k_min, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='diag',  # diag is tractable for 384-dim
                max_iter=150,
                n_init=3,
                random_state=42
            )
            gmm.fit(embeddings)

            bic = gmm.bic(embeddings)
            self.bic_scores[k] = bic

            if bic < best_bic:
                best_bic = bic
                best_k = k

            logger.info("  k=%d → BIC=%.1f", k, bic)

        # Compute silhouette score for the best k as secondary validation
        logger.info("Computing silhouette score for optimal k=%d...", best_k)
        gmm_best = GaussianMixture(
            n_components=best_k,
            covariance_type='diag',
            max_iter=150,
            n_init=3,
            random_state=42
        )
        labels = gmm_best.fit_predict(embeddings)

        # Use a sample for silhouette to avoid memory issues
        sample_size = min(5000, len(embeddings))
        indices = np.random.RandomState(42).choice(
            len(embeddings), sample_size, replace=False
        )
        sil_score = silhouette_score(
            embeddings[indices], labels[indices], metric='cosine'
        )
        self.silhouette_scores[best_k] = sil_score

        logger.info(
            "Optimal k=%d (BIC=%.1f, Silhouette=%.4f)",
            best_k, best_bic, sil_score
        )
        return best_k

    def fit(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        target_names: List[str]
    ) -> None:
        """
        Train the GMM model on document embeddings.

        Steps:
        1. Find optimal k via BIC analysis
        2. Fit the final GMM with optimal k
        3. Compute soft cluster assignments (probability distributions)
        """
        self.embeddings = embeddings
        self.targets = targets
        self.target_names = target_names

        # Step 1: Find optimal k
        self.optimal_k = self.find_optimal_k(embeddings)

        # Step 2: Fit the final model
        logger.info("Training final GMM with k=%d...", self.optimal_k)
        self.gmm = GaussianMixture(
            n_components=self.optimal_k,
            covariance_type='diag',
            max_iter=200,
            n_init=5,
            random_state=42
        )
        self.gmm.fit(embeddings)

        # Step 3: Soft cluster assignments
        self.cluster_probabilities = self.gmm.predict_proba(embeddings)
        self.cluster_labels = self.gmm.predict(embeddings)

        logger.info(
            "Clustering complete: %d documents → %d clusters",
            len(embeddings), self.optimal_k
        )

    def predict_cluster(self, query_embedding: np.ndarray) -> Dict:
        """
        Predict the cluster distribution for a query embedding.

        Returns dict with:
        - dominant_cluster: int (cluster with highest probability)
        - probabilities: list of (cluster_id, probability) sorted desc
        - is_boundary: bool (max prob < 0.40)
        """
        probs = self.gmm.predict_proba(query_embedding.reshape(1, -1))[0]
        sorted_indices = np.argsort(probs)[::-1]

        dominant_cluster = int(sorted_indices[0])
        max_prob = float(probs[dominant_cluster])

        top_clusters = [
            {"cluster_id": int(idx), "probability": float(probs[idx])}
            for idx in sorted_indices[:5]
            if probs[idx] > 0.01  # only include clusters with >1% membership
        ]

        return {
            "dominant_cluster": dominant_cluster,
            "max_probability": max_prob,
            "top_clusters": top_clusters,
            "is_boundary": max_prob < 0.40
        }

    def get_cluster_info(self) -> List[Dict]:
        """
        Get information about each cluster: size, dominant categories,
        and representative label based on most common original category.
        """
        cluster_info = []
        for k in range(self.optimal_k):
            # Documents belonging primarily to this cluster
            mask = self.cluster_labels == k
            cluster_size = int(mask.sum())

            # Find dominant original categories in this cluster
            cluster_targets = self.targets[mask]
            if len(cluster_targets) > 0:
                unique, counts = np.unique(cluster_targets, return_counts=True)
                sorted_idx = np.argsort(counts)[::-1]
                top_categories = [
                    {
                        "name": self.target_names[unique[i]],
                        "count": int(counts[i]),
                        "fraction": float(counts[i] / cluster_size)
                    }
                    for i in sorted_idx[:3]
                ]
            else:
                top_categories = []

            # Average membership probability for this cluster
            avg_prob = float(self.cluster_probabilities[:, k].mean())

            cluster_info.append({
                "cluster_id": k,
                "size": cluster_size,
                "avg_membership_prob": avg_prob,
                "top_categories": top_categories,
                "label": top_categories[0]["name"] if top_categories else "unknown"
            })

        return sorted(cluster_info, key=lambda x: x["size"], reverse=True)

    def get_boundary_cases(self, threshold: float = 0.40, max_cases: int = 10) -> List[Dict]:
        """
        Identify boundary documents where the model is genuinely uncertain.

        These are documents where the maximum cluster probability is below
        the threshold, meaning no single cluster strongly claims them.
        These often represent cross-topic documents (e.g., a post about
        the religious implications of space exploration).
        """
        max_probs = self.cluster_probabilities.max(axis=1)
        boundary_mask = max_probs < threshold
        boundary_indices = np.where(boundary_mask)[0]

        # Sort by max probability ascending (most uncertain first)
        sorted_indices = boundary_indices[np.argsort(max_probs[boundary_indices])]
        cases = []

        for idx in sorted_indices[:max_cases]:
            probs = self.cluster_probabilities[idx]
            top_clusters_idx = np.argsort(probs)[::-1][:3]

            cases.append({
                "doc_index": int(idx),
                "max_probability": float(max_probs[idx]),
                "top_clusters": [
                    {
                        "cluster_id": int(c),
                        "probability": float(probs[c]),
                        "label": self._get_cluster_label(c)
                    }
                    for c in top_clusters_idx
                ],
                "original_category": self.target_names[self.targets[idx]],
            })

        return cases

    def _get_cluster_label(self, cluster_id: int) -> str:
        """Get a short label for a cluster based on dominant category."""
        mask = self.cluster_labels == cluster_id
        if mask.sum() == 0:
            return "unknown"
        cluster_targets = self.targets[mask]
        unique, counts = np.unique(cluster_targets, return_counts=True)
        return self.target_names[unique[np.argmax(counts)]]

    def get_analysis_summary(self) -> Dict:
        """Return a summary of the clustering analysis."""
        max_probs = self.cluster_probabilities.max(axis=1)
        boundary_count = int((max_probs < 0.40).sum())

        return {
            "optimal_k": self.optimal_k,
            "bic_scores": self.bic_scores,
            "silhouette_scores": self.silhouette_scores,
            "total_documents": len(self.cluster_labels),
            "boundary_cases_count": boundary_count,
            "boundary_fraction": float(boundary_count / len(self.cluster_labels)),
            "avg_max_probability": float(max_probs.mean()),
        }
