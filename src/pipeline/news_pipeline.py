from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.clustering.topic_clusterer.service import TopicFilterService
from src.extraction.scraper import Extractor
from src.preprocessing.article_preprocessor import ArticlePreprocessor
from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "news_meta_data.csv"
DEFAULT_INGESTION_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "master_articles.csv"
DEFAULT_CLUSTER_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "clustered_news_topics.csv"
DEFAULT_CLUSTER_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "cluster_topic_summary.csv"
DEFAULT_EXTRACTION_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "articles_with_bodies.csv"
DEFAULT_PREPROCESS_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "preprocessed_articles.csv"
DEFAULT_RAW_SENTIMENT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "raw_sentiment_articles.csv"


class NewsPipeline:
    def __init__(
        self,
        source: str | Path = DEFAULT_SOURCE,
        ingestion_output: str | Path = DEFAULT_INGESTION_OUTPUT,
        cluster_output: str | Path = DEFAULT_CLUSTER_OUTPUT,
        cluster_summary_output: str | Path = DEFAULT_CLUSTER_SUMMARY_OUTPUT,
        extraction_output: str | Path = DEFAULT_EXTRACTION_OUTPUT,
        preprocess_output: str | Path = DEFAULT_PREPROCESS_OUTPUT,
        raw_sentiment_output: str | Path = DEFAULT_RAW_SENTIMENT_OUTPUT,
        extractor=None,
        cluster_service=None,
        preprocessor=None,
        lexicon_scorer=None,
    ):
        self.source_path = Path(source)
        self.ingestion_output_path = Path(ingestion_output)
        self.cluster_output_path = Path(cluster_output)
        self.cluster_summary_output_path = Path(cluster_summary_output)
        self.extraction_output_path = Path(extraction_output)
        self.preprocess_output_path = Path(preprocess_output)
        self.raw_sentiment_output_path = Path(raw_sentiment_output)

        self.extractor = extractor
        self.cluster_service = cluster_service
        self.preprocessor = preprocessor
        self.lexicon_scorer = lexicon_scorer

    def _get_extractor(self):
        if self.extractor is None:
            self.extractor = Extractor()
        return self.extractor

    def _get_cluster_service(self):
        if self.cluster_service is None:
            self.cluster_service = TopicFilterService()
        return self.cluster_service

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = ArticlePreprocessor.from_spacy_model()
        return self.preprocessor

    def _get_lexicon_scorer(self):
        if self.lexicon_scorer is None:
            self.lexicon_scorer = LexiconScorer()
        return self.lexicon_scorer

    @staticmethod
    def _resolve_body_column(df: pd.DataFrame) -> str:
        for column in ("original_body_text", "text"):
            if column in df.columns:
                return column
        raise ValueError(
            "Missing article body text column. Expected one of: original_body_text, text"
        )

    @staticmethod
    def _ensure_article_id(df: pd.DataFrame) -> pd.DataFrame:
        if "article_id" in df.columns:
            return df
        if "id" in df.columns:
            return df.rename(columns={"id": "article_id"})
        return df

    @staticmethod
    def _write_csv(df: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def run_ingestion(self) -> pd.DataFrame:
        ingested_df = build_master_csv(
            input_file=self.source_path,
            output_file=self.ingestion_output_path,
        )
        return ingested_df

    def run_clustering(self) -> pd.DataFrame:
        ingested_df = pd.read_csv(self.ingestion_output_path)
        clustering_result = self._get_cluster_service().run(ingested_df)
        self._write_csv(clustering_result.clustered_titles, self.cluster_output_path)
        self._write_csv(clustering_result.summary, self.cluster_summary_output_path)
        return clustering_result.clustered_titles

    def run_extraction(self) -> pd.DataFrame:
        clustered_df = pd.read_csv(self.cluster_output_path)
        clustered_df = self._ensure_article_id(clustered_df)
        extracted_df = self._get_extractor().extract(clustered_df)
        self._write_csv(extracted_df, self.extraction_output_path)
        return extracted_df

    def run_preprocessing(self) -> pd.DataFrame:
        extracted_df = pd.read_csv(self.extraction_output_path)
        extracted_df = self._ensure_article_id(extracted_df)
        body_column = self._resolve_body_column(extracted_df)

        if body_column != "original_body_text" and "original_body_text" not in extracted_df.columns:
            extracted_df = extracted_df.copy()
            extracted_df["original_body_text"] = extracted_df[body_column]
            body_column = "original_body_text"

        preprocessed_df = self._get_preprocessor().preprocess_article_dataframe(
            extracted_df,
            body_column=body_column,
        )
        self._write_csv(preprocessed_df, self.preprocess_output_path)
        return preprocessed_df

    def run_raw_sentiment(self) -> pd.DataFrame:
        preprocessed_df = pd.read_csv(self.preprocess_output_path)
        preprocessed_df = self._ensure_article_id(preprocessed_df)
        scored_df = self._get_lexicon_scorer().score_dataframe(preprocessed_df)
        final_columns = [
            "article_id",
            "cluster_id",
            "publish_date",
            "media_name",
            "vader_score",
            "sentiwordnet_score",
            "nrc_score",
        ]
        missing_columns = [column for column in final_columns if column not in scored_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required final sentiment columns: {missing_columns}")

        final_df = scored_df.loc[:, final_columns]
        self._write_csv(final_df, self.raw_sentiment_output_path)
        return final_df

    def run(self) -> pd.DataFrame:
        self.run_ingestion()
        self.run_clustering()
        self.run_extraction()
        self.run_preprocessing()
        return self.run_raw_sentiment()


def run_news_pipeline(source):
    pipeline = NewsPipeline(source=source)
    return pipeline.run()
