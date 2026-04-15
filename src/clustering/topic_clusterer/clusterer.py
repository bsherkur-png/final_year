from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        required_columns = {"title", "processed_title", "publish_date"}
        missing_columns = sorted(required_columns - set(prepared_titles_df.columns))
        if missing_columns:
            raise ValueError(
                f"prepared_titles_df must contain columns {sorted(required_columns)}. "
                f"Missing: {missing_columns}. Found: {list(prepared_titles_df.columns)}"
            )

        working_df = prepared_titles_df.copy()
        if working_df.empty:
            raise ValueError("No titles remain after preprocessing.")

        working_df["publish_date"] = pd.to_datetime(working_df["publish_date"])
        working_df = working_df.sort_values("publish_date").reset_index(drop=True)
        tfidf_matrix = self.vectorizer.fit_transform(working_df["processed_title"])

        # Clustering is now heuristic and time-window based rather than fixed-cluster KMeans.
        similarity_threshold = 0.65
        time_window = pd.Timedelta(hours=72)
        cluster_ids: list[int] = []
        next_cluster_id = 0
        window_start = 0

        for current_index in range(len(working_df)):
            current_publish_date = working_df.at[current_index, "publish_date"]

            while (
                window_start < current_index
                and working_df.at[window_start, "publish_date"] < current_publish_date - time_window
            ):
                window_start += 1

            candidate_positions = list(range(window_start, current_index))
            if not candidate_positions:
                cluster_ids.append(next_cluster_id)
                next_cluster_id += 1
                continue

            similarities = cosine_similarity(
                tfidf_matrix[current_index],
                tfidf_matrix[candidate_positions],
            ).ravel()
            best_match_offset = similarities.argmax()
            best_similarity = similarities[best_match_offset]

            if best_similarity >= similarity_threshold:
                best_match_index = candidate_positions[best_match_offset]
                cluster_ids.append(cluster_ids[best_match_index])
            else:
                cluster_ids.append(next_cluster_id)
                next_cluster_id += 1

        working_df["cluster_id"] = cluster_ids

        return TopicClusteringResult(
            clustered_titles=working_df,
            top_terms_by_cluster=self.extract_top_terms(
                working_df,
                tfidf_matrix,
                self.vectorizer,
                self.top_terms_per_cluster,
            ),
        )

    @staticmethod
    def extract_top_terms(
        clustered_titles: pd.DataFrame,
        tfidf_matrix,
        vectorizer: TfidfVectorizer,
        limit: int,
    ) -> dict[int, list[str]]:
        feature_names = vectorizer.get_feature_names_out()
        top_terms_by_cluster: dict[int, list[str]] = {}

        for cluster_id in sorted(clustered_titles["cluster_id"].unique()):
            cluster_rows = clustered_titles.index[clustered_titles["cluster_id"] == cluster_id]
            mean_scores = tfidf_matrix[cluster_rows].mean(axis=0).A1
            top_indices = mean_scores.argsort()[-limit:][::-1]
            top_terms_by_cluster[int(cluster_id)] = [
                feature_names[index]
                for index in top_indices
                if mean_scores[index] > 0
            ]

        return top_terms_by_cluster
