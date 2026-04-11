from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.clustering.topic_clusterer.clusterer import TopicClusterer
from src.clustering.topic_clusterer.label_rules import (
    build_cluster_labels,
    is_target_cluster_candidate,
    is_title_in_scope,
)
from src.preprocessing.article_preprocessor import ArticlePreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "news_meta_data.csv",
    PROJECT_ROOT / "data" / "intermediate" / "sample_extracted.csv",
    PROJECT_ROOT / "data" / "intermediate" / "news_meta_data_sample_1pct.csv",
)
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT /"data" / "intermediate" / "clustered_news_topics.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT /"data" / "intermediate" / "cluster_topic_summary.csv"



@dataclass
class TopicFilterResult:
    clustered_titles: pd.DataFrame
    summary: pd.DataFrame
    top_terms_by_cluster: dict[int, list[str]]
    topic_labels: dict[int, str]


class TopicFilterService:
    def __init__(
        self,
        clusterer: Optional[TopicClusterer] = None,
        preprocessor: Optional[ArticlePreprocessor] = None,
    ):
        self.clusterer = clusterer or TopicClusterer()
        self.preprocessor = preprocessor or ArticlePreprocessor.from_spacy_model()

    @staticmethod
    def resolve_input_path(input_candidates: Iterable[Path] = DEFAULT_INPUT_CANDIDATES) -> Path:
        required_columns = {"id", "title"}

        for path in input_candidates:
            if not path.exists():
                continue

            sample = pd.read_csv(path, nrows=0)
            if required_columns.issubset(sample.columns):
                return path

        raise FileNotFoundError(
            "Could not find an input CSV with the required 'id' and 'title' columns."
        )

    @staticmethod
    def load_news_titles(input_path: Path) -> pd.DataFrame:
        return pd.read_csv(input_path)

    def run(self, titles_df: pd.DataFrame) -> TopicFilterResult:
        prepared_titles = self.preprocessor.prepare_titles_for_clustering(titles_df)
        prepared_titles = prepared_titles.loc[
            prepared_titles.apply(
                lambda row: (
                    is_title_in_scope(row["title"], row["processed_title"])
                    and is_target_cluster_candidate(row["title"], row["processed_title"])
                ),
                axis=1,
            )
        ].reset_index(drop=True)
        if prepared_titles.empty:
            raise ValueError("No titles matched the final clustering cues after scope filtering.")
        clustering_result = self.clusterer.cluster_titles(prepared_titles)
        topic_labels = {
            cluster_id: str(label).strip()
            for cluster_id, label in build_cluster_labels(clustering_result.top_terms_by_cluster).items()
        }

        clustered_titles = clustering_result.clustered_titles.copy()
        clustered_titles["topic_label"] = clustered_titles["cluster_id"].map(topic_labels)
        source_id_column = "id" if "id" in titles_df.columns else "article_id"
        target_id_column = "id" if "id" in clustered_titles.columns else "article_id"
        metadata_columns = [source_id_column] + [
            column
            for column in ("url", "media_name", "news_agent", "news agent", "source")
            if column in titles_df.columns and column != source_id_column
        ]
        if len(metadata_columns) > 1:
            clustered_titles = clustered_titles.merge(
                titles_df.loc[:, metadata_columns].drop_duplicates(subset=[source_id_column]),
                left_on=target_id_column,
                right_on=source_id_column,
                how="left",
            )
            if source_id_column != target_id_column and source_id_column in clustered_titles.columns:
                clustered_titles = clustered_titles.drop(columns=[source_id_column])

        summary = self.build_summary(
            clustered_titles,
            clustering_result.top_terms_by_cluster,
            topic_labels,
        )

        clustered_titles = clustered_titles.loc[
            ~clustered_titles["topic_label"].fillna("").str.lower().str.startswith("topic")
        ].reset_index(drop=True)

        return TopicFilterResult(
            clustered_titles=clustered_titles,
            summary=summary,
            top_terms_by_cluster=clustering_result.top_terms_by_cluster,
            topic_labels=topic_labels,
        )

    @staticmethod
    def build_summary(
        clustered_titles: pd.DataFrame,
        top_terms_by_cluster: dict[int, list[str]],
        topic_labels: dict[int, str],
    ) -> pd.DataFrame:
        summary_rows = []

        for cluster_id in sorted(top_terms_by_cluster):
            cluster_rows = clustered_titles.loc[clustered_titles["cluster_id"] == cluster_id]
            cluster_titles = cluster_rows["title"].head(5).tolist()
            summary_rows.append(
                {
                    "cluster": cluster_id,
                    "topic_label": topic_labels[cluster_id],
                    "top_terms": ", ".join(top_terms_by_cluster[cluster_id]),
                    "sample_titles": " | ".join(cluster_titles),
                }
            )

        return pd.DataFrame(summary_rows)

    @staticmethod
    def save_results(
        result: TopicFilterResult,
        results_output: Path = DEFAULT_RESULTS_OUTPUT,
        summary_output: Path = DEFAULT_SUMMARY_OUTPUT,

    ) -> None:
        results_output.parent.mkdir(parents=True, exist_ok=True)
        result.clustered_titles.to_csv(results_output, index=False)
        result.summary.to_csv(summary_output, index=False)





def main() -> None:
    service = TopicFilterService()
    input_path = service.resolve_input_path()
    titles_df = service.load_news_titles(input_path)
    result = service.run(titles_df)
    service.save_results(result)

    print(f"Loaded: {input_path}")
    print(f"Titles clustered: {len(result.clustered_titles)}")
    print(f"Saved results to: {DEFAULT_RESULTS_OUTPUT}")
    print(f"Saved summary to: {DEFAULT_SUMMARY_OUTPUT}")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
