from __future__ import annotations

from dataclasses import dataclass, fields

from src.preprocessing.spacy_processor import ProcessedArticle

@dataclass
class NrcScores:
    positive: float
    negative: float
    anger: float
    fear: float
    trust: float
    joy: float
    disgust: float
    surprise: float
    anticipation: float
    sadness: float


@dataclass
class SentimentScores:
    """Scores from lexicon-based sentiment tools for one article."""

    vader: float
    nrc: float
    nrc_anger: float
    nrc_fear: float
    nrc_trust: float
    nrc_joy: float
    nrc_disgust: float
    nrc_surprise: float
    nrc_anticipation: float
    nrc_sadness: float





class LexiconScorer:
    """Calculate raw lexicon scores for one article."""

    def __init__(self):
        from nrclex import NRCLex
        from nltk.sentiment import SentimentIntensityAnalyzer

        self.vader = SentimentIntensityAnalyzer()
        self._nrclex = NRCLex()

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

    def score_nrc(self, tokens: list[str]) -> NrcScores:
        """Return NRC emotion proportions normalised by token count."""
        if not tokens:
            return NrcScores(**{f.name: 0.0 for f in fields(NrcScores)})

        n = len(tokens)
        self._nrclex.load_token_list(tokens)
        e = self._nrclex.raw_emotion_scores
        return NrcScores(
            positive=float(e.get("positive", 0)) / n,
            negative=float(e.get("negative", 0)) / n,
            anger=float(e.get("anger", 0)) / n,
            fear=float(e.get("fear", 0)) / n,
            trust=float(e.get("trust", 0)) / n,
            joy=float(e.get("joy", 0)) / n,
            disgust=float(e.get("disgust", 0)) / n,
            surprise=float(e.get("surprise", 0)) / n,
            anticipation=float(e.get("anticipation", 0)) / n,
            sadness=float(e.get("sadness", 0)) / n,
        )

    def score_article(self, article: ProcessedArticle) -> SentimentScores:
        """Return the three raw lexicon scores for one article."""
        nrc = self.score_nrc(article.lemmas)
        return SentimentScores(
            vader=self.score_vader_chunks(article.chunks),
            nrc=nrc.positive - nrc.negative,
            nrc_anger=nrc.anger,
            nrc_fear=nrc.fear,
            nrc_trust=nrc.trust,
            nrc_joy=nrc.joy,
            nrc_disgust=nrc.disgust,
            nrc_surprise=nrc.surprise,
            nrc_anticipation=nrc.anticipation,
            nrc_sadness=nrc.sadness,

        )

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
