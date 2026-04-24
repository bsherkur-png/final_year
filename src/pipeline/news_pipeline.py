from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.bias.feature_builder import FeatureBuilder
from src.bias.topic_clusterer import TopicClusterer, ClusteringResult
from src.comparison.outlet_comparator import summarize_outlets
from src.comparison.statistical_tests import (
    kruskal_wallis,
    dunns_posthoc,
    effect_sizes,
    wilcoxon_signed_rank,
)
from src.extraction.web_extractor import WebExtractor
from src.pipeline.config import PipelineConfig
from src.preprocessing.filters import filter_shamima_mentions, filter_short_articles, filter_opinion_pieces
from src.preprocessing.spacy_processor import SpacyProcessor, ProcessedArticle
from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer
from src.sentiment.zeroshot_scorer import ZeroshotScorer
from src.sentiment.scaling import scale_sentiment


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
        lexicon_df = scorer.score_all(articles)

        scored = df.set_index("article_id").join(lexicon_df).reset_index()
        scored = scored.rename(
            columns={
                "vader": "vader_score",
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
        filtered_df = filter_short_articles(filtered_df, min_words=250)
        filtered_df = filter_opinion_pieces(filtered_df)

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
            "vader_score",
            "zeroshot_score",
        ]
        optional = {"zeroshot_score"}
        missing_columns = [
            column for column in final_columns
            if column not in scored_df.columns and column not in optional
        ]
        if missing_columns:
            raise ValueError(f"Missing required final sentiment columns: {missing_columns}")
        final_columns = [c for c in final_columns if c in scored_df.columns]

        final_df = scored_df.loc[:, final_columns]
        _write_csv(final_df, self.config.raw_sentiment_output)
        return final_df

    def run_zeroshot_sentiment(
        self,
        articles: list[ProcessedArticle],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run zero-shot classification and merge scores into the raw sentiment CSV."""
        zeroshot = ZeroshotScorer()
        zeroshot_df = zeroshot.score_all(articles)

        if not self.config.raw_sentiment_output.exists():
            raise ValueError(
                "Raw sentiment CSV not found. Run lexicon sentiment first."
            )

        raw_df = pd.read_csv(self.config.raw_sentiment_output)
        raw_df = raw_df.set_index("article_id").join(
            zeroshot_df.rename(columns={"zeroshot": "zeroshot_score"})
        ).reset_index()

        _write_csv(raw_df, self.config.raw_sentiment_output)
        return raw_df

    def run_scaled_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score VADER and NRC polarity, then compute a composite mean."""
        polarity_cols = ["vader_score"]
        if "zeroshot_score" in df.columns:
            polarity_cols.append("zeroshot_score")
        scaled_df = scale_sentiment(df, polarity_columns=polarity_cols)

        checkpoint_columns = [
            c for c in scaled_df.columns
            if c not in ("title", "date_link", "vader_score", "zeroshot_score")
        ]
        _write_csv(scaled_df[checkpoint_columns], self.config.scaled_sentiment_output)
        return scaled_df

    def run_outlet_comparison(self) -> pd.DataFrame:
        sentiment_df = pd.read_csv(self.config.scaled_sentiment_output)

        summary_df = summarize_outlets(
            sentiment_df,
            polarity_column="zeroshot_z",
        )

        _write_csv(summary_df, self.config.outlet_comparison_output)
        return summary_df

    def run_statistical_tests(self) -> dict:
        """Run Kruskal-Wallis on zeroshot_z, Wilcoxon on vader_z vs zeroshot_z.

        Dunn's post-hoc is only run if Kruskal-Wallis is significant
        (p < 0.05).
        """
        sentiment_df = pd.read_csv(self.config.scaled_sentiment_output)

        kw = kruskal_wallis(sentiment_df)
        es = effect_sizes(kw["H"], kw["n"], kw["k"])

        kw_row = {**kw, **es}
        kw_df = pd.DataFrame([kw_row])
        _write_csv(kw_df, self.config.kruskal_wallis_output)

        if kw["p"] < 0.05:
            dunn_df = dunns_posthoc(sentiment_df)
            _write_csv(dunn_df, self.config.dunns_posthoc_output)

        wc = wilcoxon_signed_rank(sentiment_df)
        wc_df = pd.DataFrame([wc])
        _write_csv(wc_df, self.config.wilcoxon_output)

        return {**kw_row, "wilcoxon": wc}

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
