from dataclasses import asdict
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


def _write_csv(df: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


class NewsPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()

    def _score_sentiment(
        self,
        articles: list[ProcessedArticle],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        scorer = LexiconScorer()

        scores_by_id: dict[str, SentimentScores] = {}
        for article in articles:
            scores_by_id[article.article_id] = scorer.score_article(article)

        scores_rows = {aid: asdict(scores) for aid, scores in scores_by_id.items()}
        scores_df = pd.DataFrame.from_dict(scores_rows, orient="index")
        scores_df.index.name = "article_id"

        scored = df.set_index("article_id").join(scores_df).reset_index()
        scored = scored.rename(
            columns={
                "vader": "vader_score",
                "nrc": "nrc_score",
            }
        )

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

        _write_csv(extracted_df, self.config.extraction_raw_output)
        return extracted_df

    def run_filtering(self) -> pd.DataFrame:
        extracted_df = pd.read_csv(self.config.extraction_raw_output)
        filtered_df = filter_shamima_mentions(
            extracted_df,
            min_mentions=2,
            text_columns=("title", "body"),
        )

        _write_csv(filtered_df, self.config.extraction_output)
        return filtered_df

    def run_preprocessing(self, df: pd.DataFrame) -> list[ProcessedArticle]:
        articles = SpacyProcessor().process_dataframe(df)

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
        _write_csv(checkpoint_df, self.config.preprocess_output)

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
        _write_csv(final_df, self.config.raw_sentiment_output)
        return final_df

    def run_scaled_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score VADER and NRC polarity, then compute a composite mean."""
        from scipy.stats import zscore

        scaled_df = df.copy()
        scaled_df[["vader_z", "nrc_z"]] = scaled_df[["vader_score", "nrc_score"]].apply(zscore)
        scaled_df["composite_score"] = scaled_df[["vader_z", "nrc_z"]].mean(axis=1)

        _write_csv(scaled_df, self.config.scaled_sentiment_output)
        return scaled_df

    def run_outlet_comparison(self) -> pd.DataFrame:
        sentiment_df = pd.read_csv(self.config.scaled_sentiment_output)

        summary_df = summarize_outlets(
            sentiment_df,
            polarity_column="composite_score",
        )

        _write_csv(summary_df, self.config.outlet_comparison_output)
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
        _write_csv(result.assignments, self.config.cluster_assignments_output)
        top_terms_df = pd.DataFrame(
            [
                {"cluster": cluster, "terms": ", ".join(terms)}
                for cluster, terms in result.top_terms.items()
            ]
        )
        _write_csv(top_terms_df, self.config.cluster_top_terms_output)

        return result

    def run(self) -> pd.DataFrame:
        self.run_ingestion()
        self.run_extraction()
        filtered_df = self.run_filtering()
        filtered_df = filtered_df.groupby("news_outlet").filter(lambda g: len(g) >= 6)

        # spaCy processes once — result shared by sentiment + clustering
        articles = self.run_preprocessing(filtered_df)

        raw_df = self.run_raw_sentiment(articles, filtered_df)
        scaled_df = self.run_scaled_sentiment(raw_df)
        self.run_clustering(articles, filtered_df)
        self.run_outlet_comparison()
        return scaled_df
