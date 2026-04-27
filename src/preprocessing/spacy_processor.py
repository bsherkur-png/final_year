import re
from dataclasses import dataclass

import pandas as pd
import spacy.tokens

from src.preprocessing.text_cleaner import strip_boilerplate

CHUNK_SIZE_SENTENCES = 4


@dataclass
class ProcessedArticle:
    """Shared processed representation for a single article."""

    article_id: str
    raw_text: str
    cleaned_text: str
    doc: spacy.tokens.Doc

    @property
    def lemmas(self) -> list[str]:
        """Normalised lemmas excluding stopwords, punctuation, and non-alpha tokens."""
        return [
            token.lemma_.lower()
            for token in self.doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]

    @property
    def nrc_tokens(self) -> list[str]:
        """Lowercased lemmas for NRC scoring. Stop words and non-alpha removed."""
        return self.lemmas

    @property
    def chunks(self) -> list[str]:
        """Non-overlapping 4-sentence windows of the cleaned article text."""
        sentences: list[str] = []
        for sent in self.doc.sents:
            sentence_text = re.sub(r"\s+", " ", sent.text).strip()
            if sentence_text and sum(1 for t in sent if t.is_alpha) >= 4:
                sentences.append(sentence_text)
        return [
            " ".join(sentences[i:i + CHUNK_SIZE_SENTENCES])
            for i in range(0, len(sentences), CHUNK_SIZE_SENTENCES)
        ]


class SpacyProcessor:
    """Wrapper around spaCy that emits ProcessedArticle objects."""

    def __init__(self, nlp=None):
        """Initialize with an injected spaCy model or load en_core_web_sm."""
        if nlp is not None:
            self._nlp = nlp
        else:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def _resolve_body_column(df: pd.DataFrame) -> str:
        """Identify the body text column from known candidates."""
        for candidate in ("body", "original_body_text", "text", "content"):
            if candidate in df.columns:
                return candidate
        raise ValueError(f"No body column found. Available: {list(df.columns)}")

    def process(self, article_id: str, raw_text: str) -> ProcessedArticle:
        """Process one article body into a ProcessedArticle."""
        text = (
            ""
            if raw_text is None or (isinstance(raw_text, float) and str(raw_text) == "nan")
            else str(raw_text)
        )
        cleaned = re.sub(r"\s+", " ", strip_boilerplate(text)).strip()
        return ProcessedArticle(
            article_id=article_id,
            raw_text=text,
            cleaned_text=cleaned,
            doc=self._nlp(text),
        )

    def process_dataframe(
        self, df: pd.DataFrame, body_column: str | None = None
    ) -> list[ProcessedArticle]:
        """Process a dataframe into a list of ProcessedArticle objects."""
        if "article_id" not in df.columns:
            raise ValueError("Missing required column: article_id")

        if body_column is None:
            body_column = self._resolve_body_column(df)

        if body_column not in df.columns:
            raise ValueError(f"Missing required column: {body_column}")

        return [
            self.process(row["article_id"], row[body_column])
            for _, row in df.iterrows()
        ]
