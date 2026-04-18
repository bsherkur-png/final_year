from pathlib import Path

import pandas as pd


def _ensure_article_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "article_id" in dataframe.columns:
        return dataframe
    if "id" in dataframe.columns:
        return dataframe.rename(columns={"id": "article_id"})
    return dataframe


def _resolve_body_column(dataframe: pd.DataFrame) -> str:
    for column in ("body", "original_body_text", "text"):
        if column in dataframe.columns:
            return column
    raise ValueError(
        "Missing article body text column. Expected one of: body, original_body_text, text"
    )


def _write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


class ExtractionStageService:
    def __init__(self, extractor):
        self.extractor = extractor

    def run(self, ingestion_output_path: Path, extraction_raw_output_path: Path) -> pd.DataFrame:
        master_df = pd.read_csv(ingestion_output_path)
        master_df = _ensure_article_id(master_df)
        extracted_df = self.extractor.extract(master_df)
        _write_csv(extracted_df, extraction_raw_output_path)
        return extracted_df


class FilteringStageService:
    def __init__(self, article_filter):
        self.article_filter = article_filter

    def run(self, extraction_raw_output_path: Path, extraction_output_path: Path) -> pd.DataFrame:
        extracted_df = pd.read_csv(extraction_raw_output_path)
        extracted_df = _ensure_article_id(extracted_df)
        filtered_df = self.article_filter.filter_articles(extracted_df)
        _write_csv(filtered_df, extraction_output_path)
        return filtered_df


class PreprocessingStageService:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def run(self, extraction_output_path: Path, preprocess_output_path: Path) -> pd.DataFrame:
        extracted_df = pd.read_csv(extraction_output_path)
        extracted_df = _ensure_article_id(extracted_df)
        body_column = _resolve_body_column(extracted_df)
        preprocessed_df = self.preprocessor.preprocess_dataframe(
            extracted_df,
            body_column=body_column,
        )
        _write_csv(preprocessed_df, preprocess_output_path)
        return preprocessed_df


class SentimentStageService:
    FINAL_COLUMNS = [
        "article_id",
        "news_outlet",
        "title",
        "date_link",
        "vader_score",
        "sentiwordnet_score",
        "nrc_score",
    ]

    def __init__(self, lexicon_scorer):
        self.lexicon_scorer = lexicon_scorer

    def run(self, preprocess_output_path: Path, raw_sentiment_output_path: Path) -> pd.DataFrame:
        preprocessed_df = pd.read_csv(preprocess_output_path)
        preprocessed_df = _ensure_article_id(preprocessed_df)
        scored_df = self.lexicon_scorer.score_dataframe(preprocessed_df)
        missing_columns = [column for column in self.FINAL_COLUMNS if column not in scored_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required final sentiment columns: {missing_columns}")

        final_df = scored_df.loc[:, self.FINAL_COLUMNS]
        _write_csv(final_df, raw_sentiment_output_path)
        return final_df


class OutletComparisonStageService:
    def __init__(self, outlet_comparator, polarity_column: str = "vader_score"):
        self.outlet_comparator = outlet_comparator
        self.polarity_column = polarity_column

    def run(self, raw_sentiment_output_path: Path, outlet_comparison_output_path: Path) -> pd.DataFrame:
        sentiment_df = pd.read_csv(raw_sentiment_output_path)
        sentiment_df = _ensure_article_id(sentiment_df)
        summary_df = self.outlet_comparator.summarize_outlets(
            sentiment_df,
            polarity_column=self.polarity_column,
        )
        _write_csv(summary_df, outlet_comparison_output_path)
        return summary_df

