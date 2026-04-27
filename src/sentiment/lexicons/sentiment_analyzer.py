from __future__ import annotations

from src.preprocessing.spacy_processor import ProcessedArticle


class LexiconScorer:
    """Calculate raw lexicon scores for one article."""

    def __init__(self):
        from nltk.sentiment import SentimentIntensityAnalyzer

        self.vader = SentimentIntensityAnalyzer()

    def score_vader(self, text: str) -> float:
        """Return VADER compound score for pre-normalised text."""
        if not text:
            return 0.0

        scores = self.vader.polarity_scores(text)
        return float(scores["compound"])

    def score_vader_chunks(self, chunks: list[str]) -> float:
        """Mean of VADER compound scores across non-empty chunks.

        Returns 0.0 if the list is empty or contains only empty strings.
        """
        valid_chunks = [c for c in chunks if c.strip()]
        if not valid_chunks:
            return 0.0

        scores = [self.vader.polarity_scores(c)["compound"] for c in valid_chunks]
        return float(sum(scores) / len(scores))

    def score_all(self, articles: list[ProcessedArticle]) -> pd.DataFrame:
        """Score all articles and return a DataFrame indexed by article_id."""
        import pandas as pd

        rows = {
            article.article_id: self.score_vader_chunks(article.chunks)
            for article in articles
        }
        scores_df = pd.DataFrame.from_dict(rows, orient="index", columns=["vader"])
        scores_df.index.name = "article_id"
        return scores_df
