from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
    assignments: pd.DataFrame
    top_terms: dict[int, list[str]]
    silhouette_score: float
    k: int


class TopicClusterer:
    """Run k-means across a k range and return best clustering outputs."""

    def __init__(self, k_range: tuple[int, int] = (2, 8), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state

    def run(
        self,
        feature_matrix: csr_matrix,
        article_ids: list[str],
        outlets: list[str],
        feature_names: list[str],
    ) -> ClusteringResult:
        dense_matrix = feature_matrix.toarray()
        best_model = None
        best_score = -1.0
        best_k = self.k_range[0]

        for k in range(self.k_range[0], self.k_range[1] + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(dense_matrix)
            score = float(silhouette_score(dense_matrix, labels))
            if score > best_score:
                best_model, best_score, best_k = model, score, k

        labels = best_model.labels_
        assignments = pd.DataFrame(
            {"article_id": article_ids, "news_outlet": outlets, "cluster": labels}
        )

        top_terms: dict[int, list[str]] = {}
        for cluster_idx, center in enumerate(best_model.cluster_centers_):
            top_indices = np.argsort(center)[-10:][::-1]
            top_terms[cluster_idx] = [feature_names[i] for i in top_indices]

        return ClusteringResult(assignments, top_terms, best_score, best_k)
