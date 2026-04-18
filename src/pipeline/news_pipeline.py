from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.comparison.outlet_comparator import OutletComparator
from src.extraction.scraper import Extractor
from src.preprocessing.article_preprocessor import ArticlePreprocessor, ShamimaBegumFilter
from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "news_meta_data.csv"
DEFAULT_INGESTION_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "master_articles.csv"
DEFAULT_CLUSTER_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "clustered_news_topics.csv"
DEFAULT_CLUSTER_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "cluster_topic_summary.csv"
DEFAULT_EXTRACTION_RAW_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "articles_with_bodies_raw.csv"
DEFAULT_EXTRACTION_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "articles_with_bodies.csv"
DEFAULT_PREPROCESS_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "preprocessed_articles.csv"
DEFAULT_RAW_SENTIMENT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "raw_sentiment_articles.csv"
DEFAULT_OUTLET_COMPARISON_OUTPUT = (
    PROJECT_ROOT / "data" / "intermediate" / "outlet_comparison_summary.csv"
)


class NewsPipeline:
    def __init__(
        self,
        source: str | Path = DEFAULT_SOURCE,
        ingestion_output: str | Path = DEFAULT_INGESTION_OUTPUT,
        cluster_output: str | Path = DEFAULT_CLUSTER_OUTPUT,
        cluster_summary_output: str | Path = DEFAULT_CLUSTER_SUMMARY_OUTPUT,
        extraction_output: str | Path = DEFAULT_EXTRACTION_OUTPUT,
        extraction_raw_output: str | Path | None = DEFAULT_EXTRACTION_RAW_OUTPUT,
        preprocess_output: str | Path = DEFAULT_PREPROCESS_OUTPUT,
        raw_sentiment_output: str | Path = DEFAULT_RAW_SENTIMENT_OUTPUT,
        outlet_comparison_output: str | Path = DEFAULT_OUTLET_COMPARISON_OUTPUT,
        extractor=None,
        cluster_service=None,
        preprocessor=None,
        lexicon_scorer=None,
        outlet_comparator=None,
    ):
        self.source_path = Path(source)
        self.ingestion_output_path = Path(ingestion_output)
        self.cluster_output_path = Path(cluster_output)
        self.cluster_summary_output_path = Path(cluster_summary_output)
        self.extraction_output_path = Path(extraction_output)
        if extraction_raw_output is None:
            extraction_output_path = Path(extraction_output)
            self.extraction_raw_output_path = extraction_output_path.with_name(
                f"{extraction_output_path.stem}_raw{extraction_output_path.suffix}"
            )
        else:
            self.extraction_raw_output_path = Path(extraction_raw_output)
        self.preprocess_output_path = Path(preprocess_output)
        self.raw_sentiment_output_path = Path(raw_sentiment_output)
        self.outlet_comparison_output_path = Path(outlet_comparison_output)

        self.extractor = extractor
        self.cluster_service = cluster_service
        self.preprocessor = preprocessor
        self.lexicon_scorer = lexicon_scorer
        self.outlet_comparator = outlet_comparator

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
        if self.cluster_service is None:
            from src.clustering.topic_clusterer.service import TopicFilterService

            self.cluster_service = TopicFilterService()
        clustering_result = self.cluster_service.run(ingested_df)
        self._write_csv(clustering_result.clustered_titles, self.cluster_output_path)
        self._write_csv(clustering_result.summary, self.cluster_summary_output_path)
        return clustering_result.clustered_titles

    def run_extraction(self) -> pd.DataFrame:
        master_df = pd.read_csv(self.ingestion_output_path)
        master_df = self._ensure_article_id(master_df)

        if self.extractor is None:
            self.extractor = Extractor()
        extracted_df = self.extractor.extract(master_df)

        self._write_csv(extracted_df, self.extraction_raw_output_path)
        return extracted_df

    def run_filtering(self) -> pd.DataFrame:
        extracted_df = pd.read_csv(self.extraction_raw_output_path)
        extracted_df = self._ensure_article_id(extracted_df)

        if self.preprocessor is None:
            self.preprocessor = ArticlePreprocessor.from_spacy_model()
        filtered_df = ShamimaBegumFilter(self.preprocessor).filter_articles(extracted_df)

        self._write_csv(filtered_df, self.extraction_output_path)
        return filtered_df

    def run_preprocessing(self) -> pd.DataFrame:
        extracted_df = pd.read_csv(self.extraction_output_path)
        extracted_df = self._ensure_article_id(extracted_df)
        body_column = self._resolve_body_column(extracted_df)

        if self.preprocessor is None:
            self.preprocessor = ArticlePreprocessor.from_spacy_model()
        preprocessed_df = self.preprocessor.preprocess_dataframe(
            extracted_df,
            body_column=body_column,
        )

        self._write_csv(preprocessed_df, self.preprocess_output_path)
        return preprocessed_df

    def run_raw_sentiment(self) -> pd.DataFrame:
        preprocessed_df = pd.read_csv(self.preprocess_output_path)
        preprocessed_df = self._ensure_article_id(preprocessed_df)

        if self.lexicon_scorer is None:
            self.lexicon_scorer = LexiconScorer()
        scored_df = self.lexicon_scorer.score_dataframe(preprocessed_df)

        final_columns = [
            "article_id",
            "news_outlet",
            "title",
            "date_link",
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

    def run_outlet_comparison(self) -> pd.DataFrame:
        sentiment_df = pd.read_csv(self.raw_sentiment_output_path)
        sentiment_df = self._ensure_article_id(sentiment_df)

        if self.outlet_comparator is None:
            self.outlet_comparator = OutletComparator()
        summary_df = self.outlet_comparator.summarize_outlets(
            sentiment_df,
            polarity_column="vader_score",
        )

        self._write_csv(summary_df, self.outlet_comparison_output_path)
        return summary_df

    def run(self) -> pd.DataFrame:
        # Active execution path starts from the existing master CSV and skips clustering.
        self.run_extraction()
        self.run_filtering()
        self.run_preprocessing()
        sentiment_df = self.run_raw_sentiment()
        self.run_outlet_comparison()
        return sentiment_df

def run_news_pipeline(source):
    pipeline = NewsPipeline(ingestion_output=source)
    return pipeline.run()
