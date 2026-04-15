import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from nltk.tokenize import RegexpTokenizer

try:
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
except ImportError:
    SPACY_STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
    }


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ArticlePreprocessor:
    DEFAULT_SPACY_MODEL = "en_core_web_sm"
    DEFAULT_SPACY_DISABLE = ["ner"]
    DEFAULT_TITLE_NEWS_WORDS = frozenset({
        "announce", "announces", "breaking", "exclusive", "latest", "live", "news",
        "opinion", "podcast", "report", "reports", "reveal", "reveals", "say", "update",
    })
    DEFAULT_TITLE_POLITICIAN_NAMES = frozenset({
        "badenoch", "biden", "farage", "johnson", "macron", "obama",  "putin",
        "starmer", "sunak", "trudeau", "trump", "keir","rishi",
    })

    def __init__(self, nlp=None, title_stopwords=None, politician_names=None):
        self.nlp = nlp
        self.fallback_tokenizer = RegexpTokenizer(r"\w+")
        self.title_stopwords = set(SPACY_STOP_WORDS)
        self.title_stopwords.update(title_stopwords or self.DEFAULT_TITLE_NEWS_WORDS)
        self.title_stopwords.update(politician_names or self.DEFAULT_TITLE_POLITICIAN_NAMES)

    @classmethod
    def from_spacy_model(cls, model_name: str = DEFAULT_SPACY_MODEL, disable=None):
        nlp = cls._load_spacy_model(model_name=model_name, disable=disable)
        return cls(nlp=nlp)

    def preprocess_title(self, title: str) -> str:
        normalized = self._normalize_text(title)
        if not normalized:
            return ""

        if self.nlp is None:
            tokens = [
                token for token in self.fallback_tokenizer.tokenize(normalized)
                if token not in self.title_stopwords
            ]
            return " ".join(tokens)

        doc = self.nlp(normalized)
        return " ".join(self._extract_title_tokens(doc))

    def preprocess_body(self, body: str) -> str:
        return " ".join(self.tokenize_body(body))

    def minimal_preprocess_body(self, body: str) -> str:
        if body is None or pd.isna(body):
            return ""
        return str(body).strip()

    def preprocess_titles(self, titles: Iterable[str]) -> list[str]:
        if self.nlp is None:
            return [self.preprocess_title(title) for title in titles]

        title_texts = [self._normalize_text(title) for title in titles]
        processed_titles = []

        for doc in self.nlp.pipe(title_texts, batch_size=64):
            processed_titles.append(" ".join(self._extract_title_tokens(doc)))

        return processed_titles

    def preprocess_bodies(self, bodies: Iterable[str]) -> list[str]:
        body_texts = [self._normalize_text(body) for body in bodies]
        nlp = self._ensure_nlp()
        processed_bodies = []

        for doc in nlp.pipe(body_texts, batch_size=32):
            processed_bodies.append(" ".join(self._extract_body_tokens(doc)))

        return processed_bodies

    def preprocess_article_dataframe(
        self,
        df: pd.DataFrame,
        body_column: str = "original_body_text",
    ) -> pd.DataFrame:
        """Return a copy of df with minimal and fully preprocessed body-text columns added."""
        if body_column not in df.columns:
            raise ValueError(
                f"DataFrame must contain body column '{body_column}'. Found: {list(df.columns)}"
            )

        output_df = df.copy()
        minimal_body_texts = [self.minimal_preprocess_body(body) for body in output_df[body_column]]
        output_df["minimal_body_text"] = minimal_body_texts
        output_df["fully_preprocessed_body_text"] = self.preprocess_bodies(minimal_body_texts)
        return output_df

    def preprocess_body_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
        id_column: str = "id",
        body_column: str = "text",
    ) -> pd.DataFrame:
        df = pd.read_csv(input_path)
        required_columns = {id_column, body_column}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain columns {sorted(required_columns)}. "
                f"Found: {list(df.columns)}"
            )

        output_df = (
            df.loc[:, [id_column, body_column]]
            .rename(columns={id_column: "article_id", body_column: "original_body_text"})
        )
        output_df["original_body_text"] = output_df["original_body_text"].apply(
            lambda body: "" if pd.isna(body) else str(body)
        )
        output_df = self.preprocess_article_dataframe(output_df)
        output_df = output_df.loc[
            :,
            [
                "article_id",
                "original_body_text",
                "minimal_body_text",
                "fully_preprocessed_body_text",
            ],
        ]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        return output_df

    def prepare_titles_for_clustering(
        self,
        df: pd.DataFrame,
        title_column: str = "title",
        id_column: str = "article_id",
    ) -> pd.DataFrame:
        source_id_column = id_column
        if id_column == "article_id" and source_id_column not in df.columns and "id" in df.columns:
            source_id_column = "id"

        required_columns = {source_id_column, title_column}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain columns {sorted(required_columns)}. "
                f"Found: {list(df.columns)}"
            )

        titles_df = (
            df.loc[:, [source_id_column, title_column]]
            .rename(columns={source_id_column: "article_id", title_column: "title"})
            .dropna(subset=["title"])
            .assign(title=lambda frame: frame["title"].astype(str).str.strip())
        )
        titles_df = titles_df.loc[titles_df["title"] != ""].reset_index(drop=True)
        titles_df["processed_title"] = self.preprocess_titles(titles_df["title"])
        titles_df = titles_df.loc[titles_df["processed_title"].str.strip() != ""].reset_index(drop=True)
        return titles_df.loc[:, ["article_id", "title", "processed_title"]]

    def tokenize_body(self, body: str) -> list[str]:
        normalized = self._normalize_text(body)
        if not normalized:
            return []

        doc = self._ensure_nlp()(normalized)
        return self._extract_body_tokens(doc)


    @staticmethod
    def _extract_lemmas(doc) -> list[str]:
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and token.lemma_.strip()
        ]

    def _extract_title_tokens(self, doc) -> list[str]:
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_.strip()
            and token.lemma_.lower() not in self.title_stopwords
        ]

    @staticmethod
    def _extract_body_tokens(doc) -> list[str]:
        return ArticlePreprocessor._extract_lemmas(doc)

    @staticmethod
    def _normalize_text(text: str) -> str:
        if text is None:
            return ""

        return re.sub(r"\s+", " ", str(text)).strip().lower()

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
                f"spaCy model '{model_name}' is required for body preprocessing. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from exc


def main():
    input_path = PROJECT_ROOT / "data" / "intermediate" / "sample_body_urls.csv"
    output_path = PROJECT_ROOT / "data" / "intermediate" / "sample_body_urls_preprocessed.csv"

    preprocessor = ArticlePreprocessor.from_spacy_model()
    processed_df = preprocessor.preprocess_body_csv(input_path, output_path)
    print(f"Saved {len(processed_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
