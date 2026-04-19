from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.comparison.outlet_comparator import summarize_outlets
from src.extraction.web_extractor import WebExtractor
from src.preprocessing.article_preprocessor import filter_shamima_mentions
from src.preprocessing.spacy_processor import SpacyProcessor, ProcessedArticle
from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer, SentimentScores
from src.bias.feature_builder import FeatureBuilder
from src.bias.topic_clusterer import TopicClusterer, ClusteringResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "news_meta_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_INGESTION_OUTPUT = DEFAULT_OUTPUT_DIR / "master_articles.csv"


class NewsPipeline:
    def __init__(
        self,
        source: str | Path = DEFAULT_SOURCE,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        ingestion_output: str | Path | None = None,
    ):
        self.source_path = Path(source)
        self.output_dir = Path(output_dir)
        self.ingestion_output_path = (
            Path(ingestion_output) if ingestion_output else self.output_dir / "master_articles.csv"
        )
        self.extraction_raw_output_path = self.output_dir / "articles_with_bodies_raw.csv"
        self.extraction_output_path = self.output_dir / "articles_with_bodies.csv"
        self.preprocess_output_path = self.output_dir / "preprocessed_articles.csv"
        self.raw_sentiment_output_path = self.output_dir / "raw_sentiment_articles.csv"
        self.outlet_comparison_output_path = self.output_dir / "outlet_comparison_summary.csv"
        self.cluster_assignments_output_path = self.output_dir / "cluster_assignments.csv"
        self.cluster_top_terms_output_path = self.output_dir / "cluster_top_terms.csv"

    @staticmethod
    def _write_csv(df: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    @staticmethod
    def _ensure_article_id(df: pd.DataFrame) -> pd.DataFrame:
        import hashlib

        if "article_id" in df.columns:
            return df

        url_col = next((c for c in ("date_link", "link", "url") if c in df.columns), None)
        if url_col is None:
            raise ValueError("Cannot derive article_id: no URL column found.")

        output_df = df.copy()
        output_df["article_id"] = output_df[url_col].apply(
            lambda value: hashlib.sha256(str(value).encode()).hexdigest()
        )
        return output_df

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

        return scored

    def run_ingestion(self) -> pd.DataFrame:
        ingested_df = build_master_csv(
            input_file=self.source_path,
            output_file=self.ingestion_output_path,
        )
        return ingested_df

    def run_extraction(self) -> pd.DataFrame:
        master_df = pd.read_csv(self.ingestion_output_path)
        master_df = self._ensure_article_id(master_df)

        extracted_df = WebExtractor().extract(master_df)

        self._write_csv(extracted_df, self.extraction_raw_output_path)
        return extracted_df

    def run_filtering(self) -> pd.DataFrame:
        extracted_df = pd.read_csv(self.extraction_raw_output_path)
        extracted_df = self._ensure_article_id(extracted_df)
        filtered_df = filter_shamima_mentions(
            extracted_df,
            min_mentions=2,
            text_columns=("title", "body"),
        )

        self._write_csv(filtered_df, self.extraction_output_path)
        return filtered_df

    def run_preprocessing(self, df: pd.DataFrame) -> list[ProcessedArticle]:
        articles = self._process_with_spacy(df)

        # CSV checkpoint — write a flat version for debugging/inspection
        rows = []
        for a in articles:
            rows.append(
                {
                    "article_id": a.article_id,
                    "minimal_body_text": a.minimal_text,
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
        self._write_csv(checkpoint_df, self.preprocess_output_path)

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

        summary_df = summarize_outlets(
            sentiment_df,
            polarity_column="vader_score",
        )

        self._write_csv(summary_df, self.outlet_comparison_output_path)
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
        self._write_csv(result.assignments, self.cluster_assignments_output_path)
        top_terms_df = pd.DataFrame(
            [
                {"cluster": cluster, "terms": ", ".join(terms)}
                for cluster, terms in result.top_terms.items()
            ]
        )
        self._write_csv(top_terms_df, self.cluster_top_terms_output_path)

        return result

    def run(self) -> pd.DataFrame:
        self.run_ingestion()
        self.run_extraction()
        filtered_df = self.run_filtering()

        # spaCy processes once — result shared by sentiment + clustering
        articles = self.run_preprocessing(filtered_df)

        sentiment_df = self.run_raw_sentiment(articles, filtered_df)
        self.run_clustering(articles, filtered_df)
        self.run_outlet_comparison()
        return sentiment_df
