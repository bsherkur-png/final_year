from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.comparison.outlet_comparator import OutletComparator
from src.extraction.scraper import Extractor
from src.pipeline.stage_services import (
    ExtractionStageService,
    FilteringStageService,
    OutletComparisonStageService,
    PreprocessingStageService,
    SentimentStageService,
)
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
        extraction_stage_service=None,
        filtering_stage_service=None,
        preprocessing_stage_service=None,
        sentiment_stage_service=None,
        outlet_comparison_stage_service=None,
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
        self.extraction_stage_service = extraction_stage_service
        self.filtering_stage_service = filtering_stage_service
        self.preprocessing_stage_service = preprocessing_stage_service
        self.sentiment_stage_service = sentiment_stage_service
        self.outlet_comparison_stage_service = outlet_comparison_stage_service

    def _get_extractor(self):
        if self.extractor is None:
            self.extractor = Extractor()
        return self.extractor

    def _get_cluster_service(self):
        if self.cluster_service is None:
            from src.clustering.topic_clusterer.service import TopicFilterService

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

    def _get_outlet_comparator(self):
        if self.outlet_comparator is None:
            self.outlet_comparator = OutletComparator()
        return self.outlet_comparator

    def _get_extraction_stage_service(self):
        if self.extraction_stage_service is None:
            self.extraction_stage_service = ExtractionStageService(self._get_extractor())
        return self.extraction_stage_service

    def _get_filtering_stage_service(self):
        if self.filtering_stage_service is None:
            self.filtering_stage_service = FilteringStageService(
                ShamimaBegumFilter(self._get_preprocessor())
            )
        return self.filtering_stage_service

    def _get_preprocessing_stage_service(self):
        if self.preprocessing_stage_service is None:
            self.preprocessing_stage_service = PreprocessingStageService(self._get_preprocessor())
        return self.preprocessing_stage_service

    def _get_sentiment_stage_service(self):
        if self.sentiment_stage_service is None:
            self.sentiment_stage_service = SentimentStageService(self._get_lexicon_scorer())
        return self.sentiment_stage_service

    def _get_outlet_comparison_stage_service(self):
        if self.outlet_comparison_stage_service is None:
            self.outlet_comparison_stage_service = OutletComparisonStageService(
                self._get_outlet_comparator(),
                polarity_column="vader_score",
            )
        return self.outlet_comparison_stage_service

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
        return self._get_extraction_stage_service().run(
            self.ingestion_output_path,
            self.extraction_raw_output_path,
        )

    def run_filtering(self) -> pd.DataFrame:
        return self._get_filtering_stage_service().run(
            self.extraction_raw_output_path,
            self.extraction_output_path,
        )

    def run_preprocessing(self) -> pd.DataFrame:
        return self._get_preprocessing_stage_service().run(
            self.extraction_output_path,
            self.preprocess_output_path,
        )

    def run_raw_sentiment(self) -> pd.DataFrame:
        return self._get_sentiment_stage_service().run(
            self.preprocess_output_path,
            self.raw_sentiment_output_path,
        )

    def run_outlet_comparison(self) -> pd.DataFrame:
        return self._get_outlet_comparison_stage_service().run(
            self.raw_sentiment_output_path,
            self.outlet_comparison_output_path,
        )

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
