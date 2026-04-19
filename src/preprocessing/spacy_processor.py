from dataclasses import dataclass

import pandas as pd
import spacy.tokens


@dataclass
class ProcessedArticle:
    """Shared processed representation for a single article."""

    article_id: str
    raw_text: str
    doc: spacy.tokens.Doc

    @property
    def minimal_text(self) -> str:
        """Return minimally cleaned text for VADER scoring."""
        return self.raw_text.strip()

    @property
    def lemmas(self) -> list[str]:
        """Return normalized lemmas excluding stopwords, punctuation, and non-alpha tokens."""
        return [
            token.lemma_.lower()
            for token in self.doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]

    @property
    def tokens(self) -> list[str]:
        """Return lowercased token text excluding punctuation only."""
        return [token.text.lower() for token in self.doc if not token.is_punct]


class SpacyProcessor:
    """Wrapper around spaCy that emits ProcessedArticle objects."""

    def __init__(self, nlp=None):
        """Initialize with an injected spaCy model or load en_core_web_sm."""
        if nlp is not None:
            self._nlp = nlp
        else:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")

    def process(self, article_id: str, raw_text: str) -> ProcessedArticle:
        """Process one article body into a ProcessedArticle."""
        text = (
            ""
            if raw_text is None or (isinstance(raw_text, float) and str(raw_text) == "nan")
            else str(raw_text)
        )
        return ProcessedArticle(
            article_id=article_id,
            raw_text=text,
            doc=self._nlp(text),
        )

    def process_dataframe(self, df: pd.DataFrame, body_column: str = "body") -> list[ProcessedArticle]:
        """Process a dataframe into a list of ProcessedArticle objects."""
        if "article_id" not in df.columns:
            raise ValueError("Missing required column: article_id")
        if body_column not in df.columns:
            raise ValueError(f"Missing required column: {body_column}")

        return [
            self.process(row["article_id"], row[body_column])
            for _, row in df.iterrows()
        ]
