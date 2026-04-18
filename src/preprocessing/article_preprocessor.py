import re

import pandas as pd


class ArticlePreprocessor:
    DEFAULT_SPACY_MODEL = "en_core_web_sm"
    DEFAULT_SPACY_DISABLE = ["ner"]

    def __init__(self, nlp=None):
        self.nlp = nlp

    @classmethod
    def from_spacy_model(cls, model_name: str = DEFAULT_SPACY_MODEL, disable=None):
        nlp = cls._load_spacy_model(model_name=model_name, disable=disable)
        return cls(nlp=nlp)

    def _ensure_nlp(self):
        if self.nlp is None:
            self.nlp = self._load_spacy_model()
        return self.nlp

    @classmethod
    def _load_spacy_model(cls, model_name: str = DEFAULT_SPACY_MODEL, disable=None):
        import spacy

        try:
            return spacy.load(model_name, disable=disable or cls.DEFAULT_SPACY_DISABLE)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model_name}' is required. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from exc

    def preprocess_dataframe(
        self,
        dataframe: pd.DataFrame,
        body_column: str = "original_body_text",
    ) -> pd.DataFrame:
        if body_column not in dataframe.columns:
            raise ValueError(f"Missing required column: {body_column}")

        preprocessed_df = dataframe.copy()
        preprocessed_df["original_body_text"] = preprocessed_df[body_column].apply(
            lambda body: "" if pd.isna(body) else str(body)
        )
        preprocessed_df["minimal_body_text"] = preprocessed_df["original_body_text"].str.strip()

        nlp = self._ensure_nlp()
        preprocessed_df["fully_preprocessed_body_text"] = preprocessed_df["minimal_body_text"].apply(
            lambda text: " ".join(token.text.lower() for token in nlp(text) if not token.is_space)
        )
        return preprocessed_df


class ShamimaBegumFilter:
    def __init__(self, preprocessor: ArticlePreprocessor | None = None, min_mentions: int = 2):
        self.preprocessor = preprocessor
        self.min_mentions = min_mentions
        self.pattern = re.compile(r"\bshamima\s+begum\b", re.IGNORECASE)

    def _count_mentions(self, *texts: str) -> int:
        merged_text = " ".join(
            "" if text is None or pd.isna(text) else str(text)
            for text in texts
        )
        return len(self.pattern.findall(merged_text))

    def filter_articles(
        self,
        articles: pd.DataFrame,
        text_columns: tuple[str, ...] = ("title", "body"),
    ) -> pd.DataFrame:
        missing_columns = [column for column in text_columns if column not in articles.columns]
        if missing_columns:
            raise ValueError(
                f"DataFrame must contain columns {missing_columns}. "
                f"Available: {list(articles.columns)}"
            )

        mention_counts = articles.apply(
            lambda row: self._count_mentions(*(row[column] for column in text_columns)),
            axis=1,
        )
        return articles.loc[mention_counts >= self.min_mentions].reset_index(drop=True)


