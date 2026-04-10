from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TopicClusteringResult:
    clustered_titles: pd.DataFrame
    top_terms_by_cluster: dict[int, list[str]]


class TopicClusterer:
    def __init__(
        self,
        n_clusters: int = 6,
        random_state: int = 22,
        top_terms_per_cluster: int = 10,
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.top_terms_per_cluster = top_terms_per_cluster
        self.vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.75)

    def cluster_titles(self, prepared_titles_df: pd.DataFrame) -> TopicClusteringResult:
        required_columns = {"title", "processed_title"}
        if not required_columns.issubset(prepared_titles_df.columns):
            raise ValueError(
                f"prepared_titles_df must contain columns {sorted(required_columns)}. "
                f"Found: {list(prepared_titles_df.columns)}"
            )

        working_df = prepared_titles_df.copy()
        if working_df.empty:
            raise ValueError("No titles remain after preprocessing.")

        cluster_count = min(self.n_clusters, len(working_df))
        tfidf_matrix = self.vectorizer.fit_transform(working_df["processed_title"])
        model = KMeans(n_clusters=cluster_count, random_state=self.random_state, n_init=10)
        working_df["cluster"] = model.fit_predict(tfidf_matrix)

        return TopicClusteringResult(
            clustered_titles=working_df,
            top_terms_by_cluster=self.extract_top_terms(model, self.vectorizer, self.top_terms_per_cluster),
        )

    @staticmethod
    def extract_top_terms(
        model: KMeans,
        vectorizer: TfidfVectorizer,
        limit: int,
    ) -> dict[int, list[str]]:
        feature_names = vectorizer.get_feature_names_out()
        top_terms_by_cluster = {}

        for cluster_id, centroid in enumerate(model.cluster_centers_):
            top_indices = centroid.argsort()[-limit:][::-1]
            top_terms_by_cluster[cluster_id] = [feature_names[index] for index in top_indices]

        return top_terms_by_cluster
