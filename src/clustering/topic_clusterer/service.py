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
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "src" / "ingestion" / "data" / "clustered_news_topics.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "src" / "ingestion" / "data" / "cluster_topic_summary.csv"
DEFAULT_TOPIC_DATASETS_DIR = PROJECT_ROOT / "src" / "ingestion" / "data" / "topic_datasets"


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
        clustered_titles["topic_label"] = clustered_titles["cluster"].map(topic_labels)
        metadata_columns = ["id"] + [
            column
            for column in ("url", "media_name", "news_agent", "news agent", "source")
            if column in titles_df.columns
        ]
        if len(metadata_columns) > 1:
            clustered_titles = clustered_titles.merge(
                titles_df.loc[:, metadata_columns].drop_duplicates(subset=["id"]),
                on="id",
                how="left",
            )

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
            cluster_rows = clustered_titles.loc[clustered_titles["cluster"] == cluster_id]
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
        topic_datasets_dir: Path = DEFAULT_TOPIC_DATASETS_DIR,
    ) -> None:
        results_output.parent.mkdir(parents=True, exist_ok=True)
        result.clustered_titles.to_csv(results_output, index=False)
        result.summary.to_csv(summary_output, index=False)
        TopicFilterService.save_topic_datasets(result.clustered_titles, topic_datasets_dir)

    @staticmethod
    def save_topic_datasets(clustered_titles: pd.DataFrame, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        for topic_label, topic_rows in clustered_titles.groupby("topic_label", sort=True):
            cleaned_label = str(topic_label).strip()
            if not cleaned_label:
                continue

            file_name = f"{TopicFilterService.slugify_label(cleaned_label)}.csv"
            topic_rows.to_csv(output_dir / file_name, index=False)

    @staticmethod
    def slugify_label(label: str) -> str:
        slug = label.strip().lower().replace("/", " ").replace("\\", " ")
        slug = "".join(character if character.isalnum() else "_" for character in slug)
        slug = "_".join(part for part in slug.split("_") if part)
        return slug or "topic_dataset"


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
