
from dataclasses import dataclass
import re
from typing import Sequence

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


class ShamimaBegumFilter:
    def __init__(self, preprocessor: ArticlePreprocessor):
        self.preprocessor = preprocessor
        self.pattern = re.compile(r"\bshamima\s+begum\b")

    def _normalize_text(self, text: str) -> str:
        if text is None or pd.isna(text):
            return ""

        doc = self.preprocessor._ensure_nlp()(str(text))
        return " ".join(token.text.lower() for token in doc if not token.is_space)

    def _count_mentions(self, text: str) -> int:
        normalized_text = self._normalize_text(text)
        return len(self.pattern.findall(normalized_text))

    def filter_articles(self, articles: pd.DataFrame) -> pd.DataFrame:
        if "body" not in articles.columns:
            raise ValueError("DataFrame must contain column 'body'.")

        mention_counts = articles["body"].apply(self._count_mentions)
        return articles.loc[mention_counts >= 2].reset_index(drop=True)


