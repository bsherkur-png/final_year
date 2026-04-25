import re
from dataclasses import dataclass

import pandas as pd
import spacy.tokens


# Boilerplate patterns identified by manual corpus inspection.
# Each tuple is (pattern_type, compiled_regex).
# "head" patterns are removed from the start of text.
# "tail" patterns truncate text at the match position.
# "inline" patterns are removed wherever they appear.
_BOILERPLATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Daily Mail: "By NAME Published: ... View comments"
    ("head", re.compile(
        r"^By\s+[A-Z\s,]+(?:FOR\s+[A-Z\s]+)?\s*"
        r"Published:\s*[\d:,\s]+\w+\s+\d{4}\s*"
        r"(?:\|\s*Updated:\s*[\d:,\s]+\w+\s+\d{4}\s*)?"
        r"[\d.]+[k]?\s*View\s+comments\s*",
        re.IGNORECASE,
    )),
    # The Independent: subheadline + "Removed from bookmarks ... Privacy notice"
    ("head", re.compile(
        r"^.{0,300}?Removed\s+from\s+bookmarks\s+"
        r"I\s+would\s+like\s+to\s+be\s+emailed\s+about\s+offers.*?"
        r"Read\s+our\s+Privacy\s+notice\s*",
        re.IGNORECASE | re.DOTALL,
    )),
    # Daily Mail: comment section block
    ("tail", re.compile(
        r"\s*Share\s+what\s+you\s+think\s+The\s+comments\s+below\b",
        re.IGNORECASE,
    )),
    # The Independent: engagement CTA
    ("tail", re.compile(
        r"\s*Join\s+thought-provoking\s+conversations,\s+follow\s+other\s+"
        r"Independent\s+readers\b",
        re.IGNORECASE,
    )),
    # BBC: copyright footer (English and non-English variants)
    ("tail", re.compile(
        r"\s*(?:Copyright|©)\s+20\d{2}\s+BBC\b",
        re.IGNORECASE,
    )),
    # BBC: social/email CTA
    ("tail", re.compile(
        r"\s*Follow\s+BBC\s+\w+\s+on\s+Facebook\b",
        re.IGNORECASE,
    )),
    # The Guardian: letters page CTA
    ("tail", re.compile(
        r"\s*Join\s+the\s+debate\s+.{0,10}email\s+guardian\.letters\b",
        re.IGNORECASE,
    )),
    # The Mirror: TV scheduling
    ("tail", re.compile(
        r"\s*\*This\s+Morning\s+airs\b",
        re.IGNORECASE,
    )),
    # The Mirror: contact CTA
    ("tail", re.compile(
        r"\s*Do\s+you\s+have\s+a\s+story\s+to\s+sell\?\s+Get\s+in\s+touch\b",
        re.IGNORECASE,
    )),
    # The Mirror: inline image captions — "(Image: SOURCE)"
    ("inline", re.compile(
        r"\(Image:\s*[^)]+\)",
        re.IGNORECASE,
    )),
]


@dataclass
class ProcessedArticle:
    """Shared processed representation for a single article."""

    article_id: str
    raw_text: str
    doc: spacy.tokens.Doc

    @property
    def vader_text(self) -> str:
        """Whitespace-normalised raw text for VADER. No other preprocessing."""
        return re.sub(r"\s+", " ", self.raw_text).strip()

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        """Remove non-editorial boilerplate identified by corpus inspection.

        Applies head patterns (strip from start), tail patterns
        (truncate at match), and inline patterns (remove all
        occurrences). Order: head → inline → tail.
        """
        for pattern_type, pattern in _BOILERPLATE_PATTERNS:
            if pattern_type == "head":
                text = pattern.sub("", text, count=1)

        for pattern_type, pattern in _BOILERPLATE_PATTERNS:
            if pattern_type == "inline":
                text = pattern.sub("", text)

        for pattern_type, pattern in _BOILERPLATE_PATTERNS:
            if pattern_type == "tail":
                match = pattern.search(text)
                if match:
                    text = text[:match.start()]
                    break

        return text

    @property
    def cleaned_text(self) -> str:
        """Boilerplate-stripped, whitespace-normalised text.

        Used by both VADER and zero-shot scorers to ensure identical
        input for fair model comparison. Preserves capitalisation,
        punctuation, and sentence structure.
        """
        stripped = self._strip_boilerplate(self.raw_text)
        return re.sub(r"\s+", " ", stripped).strip()

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
        return ProcessedArticle(
            article_id=article_id,
            raw_text=text,
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
