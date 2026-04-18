import re

import pandas as pd


class ArticlePreprocessor:
    def __init__(self, nlp=None):
        self.nlp = nlp

    def _ensure_nlp(self):
        if self.nlp is None:
            import spacy

            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        return self.nlp

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


_SHAMIMA_PATTERN = re.compile(r"\bshamima\s+begum\b", re.IGNORECASE)


def filter_shamima_mentions(
    df: pd.DataFrame,
    min_mentions: int = 2,
    text_columns: tuple[str, ...] = ("title", "body"),
) -> pd.DataFrame:
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns {missing}. Available: {list(df.columns)}")

    def count(row):
        combined = " ".join("" if pd.isna(row[c]) else str(row[c]) for c in text_columns)
        return len(_SHAMIMA_PATTERN.findall(combined))

    counts = df.apply(count, axis=1)
    return df.loc[counts >= min_mentions].reset_index(drop=True)


