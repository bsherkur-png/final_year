from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.comparison.outlet_comparator import OutletComparator
from src.extraction.web_extractor import WebExtractor
from src.preprocessing.article_preprocessor import ArticlePreprocessor, ShamimaBegumFilter
from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer


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
        filtered_df = ShamimaBegumFilter(min_mentions=2).filter_articles(
            extracted_df,
            text_columns=("title", "body"),
        )

        self._write_csv(filtered_df, self.extraction_output_path)
        return filtered_df

    def run_preprocessing(self) -> pd.DataFrame:
        extracted_df = pd.read_csv(self.extraction_output_path)
        extracted_df = self._ensure_article_id(extracted_df)
        body_column = self._resolve_body_column(extracted_df)

        preprocessed_df = ArticlePreprocessor.from_spacy_model().preprocess_dataframe(
            extracted_df,
            body_column=body_column,
        )

        self._write_csv(preprocessed_df, self.preprocess_output_path)
        return preprocessed_df

    def run_raw_sentiment(self) -> pd.DataFrame:
        preprocessed_df = pd.read_csv(self.preprocess_output_path)
        preprocessed_df = self._ensure_article_id(preprocessed_df)

        scored_df = LexiconScorer().score_dataframe(preprocessed_df)

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

        summary_df = OutletComparator().summarize_outlets(
            sentiment_df,
            polarity_column="vader_score",
        )

        self._write_csv(summary_df, self.outlet_comparison_output_path)
        return summary_df

    def run(self) -> pd.DataFrame:
        # Active execution path starts from the existing master CSV.
        self.run_extraction()
        self.run_filtering()
        self.run_preprocessing()
        sentiment_df = self.run_raw_sentiment()
        self.run_outlet_comparison()
        return sentiment_df
