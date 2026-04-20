from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.bias.feature_builder import FeatureBuilder
from src.bias.topic_clusterer import TopicClusterer, ClusteringResult
from src.comparison.outlet_comparator import summarize_outlets
from src.extraction.web_extractor import WebExtractor
from src.pipeline.config import PipelineConfig
from src.preprocessing.filters import filter_shamima_mentions
from src.preprocessing.spacy_processor import SpacyProcessor, ProcessedArticle
from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer, SentimentScores


class NewsPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()

    @staticmethod
    def _write_csv(df: pd.DataFrame, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)

    @staticmethod
    def _resolve_body_column(df: pd.DataFrame) -> str:
        for candidate in ("body", "original_body_text", "text", "content"):
            if candidate in df.columns:
                return candidate
        raise ValueError(f"No body column found. Available: {list(df.columns)}")

    def _process_with_spacy(self, df: pd.DataFrame) -> list[ProcessedArticle]:
        body_column = self._resolve_body_column(df)
        processor = SpacyProcessor()
        return processor.process_dataframe(df, body_column=body_column)

    def _score_sentiment(
        self,
        articles: list[ProcessedArticle],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        scorer = LexiconScorer()
        scored = df.copy()

        scores_by_id: dict[str, SentimentScores] = {}
        for article in articles:
            scores_by_id[article.article_id] = scorer.score_article(article)

        scored["vader_score"] = scored["article_id"].map(
            lambda aid: scores_by_id[aid].vader
        )
        scored["sentiwordnet_score"] = scored["article_id"].map(
            lambda aid: scores_by_id[aid].sentiwordnet
        )
        scored["nrc_score"] = scored["article_id"].map(
            lambda aid: scores_by_id[aid].nrc
        )
        scored["nrc_anger"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_anger)
        scored["nrc_fear"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_fear)
        scored["nrc_trust"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_trust)
        scored["nrc_joy"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_joy)
        scored["nrc_disgust"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_disgust)
        scored["nrc_surprise"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_surprise)
        scored["nrc_anticipation"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_anticipation)
        scored["nrc_sadness"] = scored["article_id"].map(lambda aid: scores_by_id[aid].nrc_sadness)

        return scored

    def run_ingestion(self) -> pd.DataFrame:
        ingested_df = build_master_csv(
            input_file=self.config.source_path,
            output_file=self.config.ingestion_output,
        )
        return ingested_df

    def run_extraction(self) -> pd.DataFrame:
        master_df = pd.read_csv(self.config.ingestion_output)

        extracted_df = WebExtractor().extract(master_df)

        self._write_csv(extracted_df, self.config.extraction_raw_output)
        return extracted_df

    def run_filtering(self) -> pd.DataFrame:
        extracted_df = pd.read_csv(self.config.extraction_raw_output)
        filtered_df = filter_shamima_mentions(
            extracted_df,
            min_mentions=2,
            text_columns=("title", "body"),
        )

        self._write_csv(filtered_df, self.config.extraction_output)
        return filtered_df

    def run_preprocessing(self, df: pd.DataFrame) -> list[ProcessedArticle]:
        articles = self._process_with_spacy(df)

        # CSV checkpoint — write a flat version for debugging/inspection
        rows = []
        for a in articles:
            rows.append(
                {
                    "article_id": a.article_id,
                    "vader_text": a.vader_text,
                    "lemmas": " ".join(a.lemmas),
                }
            )
        checkpoint_df = pd.DataFrame(rows)
        # Merge metadata from the input df so the checkpoint is self-contained
        meta_cols = [c for c in ("news_outlet", "title", "date_link") if c in df.columns]
        if meta_cols:
            checkpoint_df = checkpoint_df.merge(
                df[["article_id"] + meta_cols], on="article_id", how="left"
            )
        self._write_csv(checkpoint_df, self.config.preprocess_output)

        return articles

    def run_raw_sentiment(
        self,
        articles: list[ProcessedArticle],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        scored_df = self._score_sentiment(articles, df)

        final_columns = [
            "article_id",
            "news_outlet",
            "title",
            "date_link",
            "vader_score",
            "sentiwordnet_score",
            "nrc_score",
            "nrc_anger",
            "nrc_fear",
            "nrc_trust",
            "nrc_joy",
            "nrc_disgust",
            "nrc_surprise",
            "nrc_anticipation",
            "nrc_sadness",
        ]
        missing_columns = [column for column in final_columns if column not in scored_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required final sentiment columns: {missing_columns}")

        final_df = scored_df.loc[:, final_columns]
        self._write_csv(final_df, self.config.raw_sentiment_output)
        return final_df

    def run_outlet_comparison(self) -> pd.DataFrame:
        sentiment_df = pd.read_csv(self.config.raw_sentiment_output)

        summary_df = summarize_outlets(
            sentiment_df,
            polarity_column="vader_score",
        )

        self._write_csv(summary_df, self.config.outlet_comparison_output)
        return summary_df

    def run_clustering(
        self,
        articles: list[ProcessedArticle],
        df: pd.DataFrame,
    ) -> ClusteringResult:
        if "news_outlet" not in df.columns:
            raise ValueError("Missing required column: news_outlet")

        builder = FeatureBuilder()
        feature_matrix = builder.build(articles)
        feature_names = builder.feature_names

        article_ids = [a.article_id for a in articles]
        id_to_outlet = dict(zip(df["article_id"], df["news_outlet"]))
        outlets = [id_to_outlet[article_id] for article_id in article_ids]
        result = TopicClusterer().run(
            feature_matrix, article_ids, outlets, feature_names
        )
        self._write_csv(result.assignments, self.config.cluster_assignments_output)
        top_terms_df = pd.DataFrame(
            [
                {"cluster": cluster, "terms": ", ".join(terms)}
                for cluster, terms in result.top_terms.items()
            ]
        )
        self._write_csv(top_terms_df, self.config.cluster_top_terms_output)

        return result

    def run(self) -> pd.DataFrame:
        self.run_ingestion()
        self.run_extraction()
        filtered_df = self.run_filtering()
        filtered_df = filtered_df.groupby("news_outlet").filter(lambda g: len(g) >= 6)

        # spaCy processes once — result shared by sentiment + clustering
        articles = self.run_preprocessing(filtered_df)

        sentiment_df = self.run_raw_sentiment(articles, filtered_df)
        self.run_clustering(articles, filtered_df)
        self.run_outlet_comparison()
        return sentiment_df
