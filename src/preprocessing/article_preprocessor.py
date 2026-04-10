import re
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

    def prepare_titles_for_clustering(
        self,
        df: pd.DataFrame,
        title_column: str = "title",
        id_column: str = "id",
    ) -> pd.DataFrame:
        required_columns = {id_column, title_column}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain columns {sorted(required_columns)}. "
                f"Found: {list(df.columns)}"
            )

        titles_df = (
            df.loc[:, [id_column, title_column]]
            .dropna(subset=[title_column])
            .assign(**{title_column: lambda frame: frame[title_column].astype(str).str.strip()})
        )
        titles_df = titles_df.loc[titles_df[title_column] != ""].drop_duplicates(subset=[title_column]).reset_index(drop=True)
        titles_df["processed_title"] = self.preprocess_titles(titles_df[title_column])
        titles_df = titles_df.loc[titles_df["processed_title"].str.strip() != ""].reset_index(drop=True)
        return titles_df

    def tokenize_body(self, body: str) -> list[str]:
        normalized = self._normalize_text(body)
        if not normalized:
            return []

        doc = self._ensure_nlp()(normalized)
        return self._extract_body_tokens(doc)

    def _preprocess_text(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return ""

        if self.nlp is None:
            return " ".join(self.fallback_tokenizer.tokenize(normalized))

        doc = self.nlp(normalized)
        return " ".join(self._extract_lemmas(doc))

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
