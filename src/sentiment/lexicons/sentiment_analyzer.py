from dataclasses import dataclass
import re

import pandas as pd

from src.preprocessing.spacy_processor import ProcessedArticle


@dataclass
class SentimentScores:
    """Scores from all three lexicon-based sentiment tools for one article."""

    vader: float
    sentiwordnet: float
    nrc: float


class LexiconScorer:
    """Calculate raw lexicon scores for one article."""

    def __init__(self, nlp=None):
        from nltk.sentiment import SentimentIntensityAnalyzer
        from nltk.corpus import sentiwordnet as swn
        from nltk.corpus import wordnet as wn

        self.vader = SentimentIntensityAnalyzer()
        self.wordnet = wn
        self.sentiwordnet = swn
        self._nlp = nlp

    def score_vader(self, text: str) -> float:
        """Return a raw VADER score for minimally preprocessed text."""
        normalized_text = self._normalise_text(text)
        if not normalized_text:
            return 0.0

        scores = self.vader.polarity_scores(normalized_text)
        return float(scores["compound"])

    def score_sentiwordnet(self, tokens: list[str]) -> float:
        """Return the average SentiWordNet score for matched tokens."""
        if not tokens:
            return 0.0

        scores = []

        for token in tokens:
            token_score = self._lookup_sentiwordnet_score(token)
            if token_score is not None:
                scores.append(token_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def score_nrc(self, tokens: list[str]) -> float:
        """Return NRC raw sentiment as positive token count minus negative token count."""
        from nrclex import NRCLex

        if not tokens:
            return 0.0

        nrclex = NRCLex()
        nrclex.load_token_list(tokens)
        emotion_scores = nrclex.raw_emotion_scores
        positive_count = emotion_scores.get("positive", 0)
        negative_count = emotion_scores.get("negative", 0)

        return float(positive_count - negative_count)

    def score_article(self, article: ProcessedArticle) -> SentimentScores:
        """Return the three raw lexicon scores for one article."""
        return SentimentScores(
            vader=self.score_vader(article.minimal_text),
            sentiwordnet=self.score_sentiwordnet(article.lemmas),
            nrc=self.score_nrc(article.lemmas),
        )

    def score_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return the input dataframe with lexicon score columns appended."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("score_dataframe expects a pandas DataFrame.")

        self._validate_dataframe_columns(dataframe)
        nlp = self._ensure_nlp()
        scored_dataframe = dataframe.copy()

        for idx, row in scored_dataframe.iterrows():
            article = ProcessedArticle(
                article_id=row["article_id"],
                raw_text=row["minimal_body_text"],
                doc=nlp(str(row.get("minimal_body_text", ""))),
            )
            scores = self.score_article(article)
            scored_dataframe.at[idx, "vader_score"] = scores.vader
            scored_dataframe.at[idx, "sentiwordnet_score"] = scores.sentiwordnet
            scored_dataframe.at[idx, "nrc_score"] = scores.nrc

        return scored_dataframe

    def _ensure_nlp(self):
        """Load and cache the spaCy pipeline for DataFrame compatibility scoring."""
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")

        return self._nlp

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Clean extra whitespace before scoring."""
        if text is None:
            return ""

        return re.sub(r"\s+", " ", str(text)).strip()

    def _lookup_sentiwordnet_score(self, token: str) -> float | None:
        """Return a simple SentiWordNet score for one token."""
        try:
            synsets = self.wordnet.synsets(token)
        except LookupError:
            raise

        if not synsets:
            return None

        senti_synset = self.sentiwordnet.senti_synset(synsets[0].name())
        return float(senti_synset.pos_score() - senti_synset.neg_score())

    @staticmethod
    def _validate_dataframe_columns(dataframe) -> None:
        """Check that the input data has the columns needed for scoring."""
        required_columns = {
            "article_id",
            "minimal_body_text",
            "fully_preprocessed_body_text",
        }
        missing_columns = sorted(required_columns - set(dataframe.columns))

        if missing_columns:
            raise ValueError(
                f"Input data is missing required columns: {missing_columns}"
            )
