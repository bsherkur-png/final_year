import re
from pathlib import Path


class LexiconScorer:
    """Calculate raw lexicon scores for one article."""

    def __init__(self):
        """Load the resources needed for VADER, SentiWordNet, and NRC."""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer

            self.vader = SentimentIntensityAnalyzer()
        except ImportError as exc:
            raise ImportError(
                "NLTK is required for VADER scoring."
            ) from exc
        except LookupError as exc:
            raise LookupError(
                "The NLTK VADER lexicon is required. "
                "Install it with: python -m nltk.downloader vader_lexicon"
            ) from exc

        try:
            from nltk.corpus import sentiwordnet as swn
            from nltk.corpus import wordnet as wn

            wn.synsets("good")
            swn.senti_synset("good.a.01")
            self.wordnet = wn
            self.sentiwordnet = swn
        except ImportError as exc:
            raise ImportError("NLTK is required for SentiWordNet scoring.") from exc
        except LookupError as exc:
            raise LookupError(
                "The NLTK wordnet and sentiwordnet resources are required. "
                "Install them with: python -m nltk.downloader wordnet sentiwordnet"
            ) from exc

        try:
            from nrclex import NRCLex

            self.nrclex = NRCLex()
        except ImportError as exc:
            raise ImportError(
                "The nrclex package is required for NRC scoring."
            ) from exc

    def score_vader(self, text: str) -> float:
        """Return a raw VADER score for minimally preprocessed text."""
        normalized_text = self._normalise_text(text)
        if not normalized_text:
            return 0.0

        scores = self.vader.polarity_scores(normalized_text)
        return float(scores["compound"])

    def score_sentiwordnet(self, text: str) -> float:
        """Return the average SentiWordNet score for matched tokens."""
        tokens = self._tokenize(text)
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

    def score_nrc(self, text: str) -> float:
        """Return NRC raw sentiment as positive token count minus negative token count."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0

        self.nrclex.load_token_list(tokens)
        emotion_scores = self.nrclex.raw_emotion_scores
        positive_count = emotion_scores.get("positive", 0)
        negative_count = emotion_scores.get("negative", 0)

        return float(positive_count - negative_count)

    def score_article(self, minimal_text: str, processed_text: str) -> dict:
        """Return the three raw lexicon scores for one article."""
        return {
            "vader_score": self.score_vader(minimal_text),
            "sentiwordnet_score": self.score_sentiwordnet(processed_text),
            "nrc_score": self.score_nrc(processed_text),
        }

    def score_dataframe(self, dataframe):
        """Return the input dataframe with lexicon score columns appended."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for score_dataframe.") from exc

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("score_dataframe expects a pandas DataFrame.")

        self._validate_dataframe_columns(dataframe)
        scored_dataframe = dataframe.copy()
        scored_dataframe["vader_score"] = scored_dataframe["minimal_body_text"].apply(
            self.score_vader
        )
        scored_dataframe["sentiwordnet_score"] = scored_dataframe[
            "fully_preprocessed_body_text"
        ].apply(self.score_sentiwordnet)
        scored_dataframe["nrc_score"] = scored_dataframe[
            "fully_preprocessed_body_text"
        ].apply(self.score_nrc)

        return scored_dataframe

    def score_csv(self, input_path: str | Path):
        """Read a preprocessed CSV with pandas and return raw scores for each article."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for score_csv.") from exc

        dataframe = pd.read_csv(input_path)
        return self.score_dataframe(dataframe)

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Clean extra whitespace before scoring."""
        if text is None:
            return ""

        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into simple lowercase word tokens."""
        normalized_text = LexiconScorer._normalise_text(text).lower()
        if not normalized_text:
            return []

        return re.findall(r"[a-z]+", normalized_text)

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
