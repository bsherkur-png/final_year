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
    def vader_text(self) -> str:
        """Whitespace-normalised raw text for VADER. No other preprocessing."""
        import re

        return re.sub(r"\s+", " ", self.raw_text).strip()

    @property
    def lemmas(self) -> list[str]:
        """Return normalized lemmas excluding stopwords, punctuation, and non-alpha tokens."""
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
    def sentiwordnet_tokens(self) -> list[tuple[str, str]]:
        """Lemma + WordNet POS pairs for SentiWordNet scoring.

        Returns (lemma, wn_pos) tuples where wn_pos is one of
        'n', 'v', 'a', 'r' (noun, verb, adj, adv). Tokens whose
        spaCy POS does not map to a WordNet POS are skipped.
        Stop words, punctuation, and non-alpha tokens are excluded.
        """
        _SPACY_TO_WN = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

        result = []
        for token in self.doc:
            if token.is_stop or token.is_punct or not token.is_alpha:
                continue
            wn_pos = _SPACY_TO_WN.get(token.pos_)
            if wn_pos is None:
                continue
            result.append((token.lemma_.lower(), wn_pos))
        return result

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
