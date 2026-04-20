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
    """Scores from all three lexicon-based sentiment tools for one article."""

    vader: float
    sentiwordnet: float
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
        from nltk.corpus import sentiwordnet as swn
        from nltk.corpus import wordnet as wn

        self.vader = SentimentIntensityAnalyzer()
        self.wordnet = wn
        self.sentiwordnet = swn
        self._nrclex = NRCLex()

    def score_vader(self, text: str) -> float:
        """Return VADER compound score for pre-normalised text."""
        if not text:
            return 0.0

        scores = self.vader.polarity_scores(text)
        return float(scores["compound"])

    def score_sentiwordnet(self, tokens: list[tuple[str, str]]) -> float:
        """Return average SentiWordNet score for (lemma, wn_pos) pairs."""
        if not tokens:
            return 0.0

        scores = []

        for lemma, wn_pos in tokens:
            token_score = self._lookup_sentiwordnet_score(lemma, wn_pos)
            if token_score is not None:
                scores.append(token_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def score_nrc(self, tokens: list[str]) -> NrcScores:
        if not tokens:
            return NrcScores(**{f.name: 0.0 for f in fields(NrcScores)})

        self._nrclex.load_token_list(tokens)
        e = self._nrclex.raw_emotion_scores
        return NrcScores(
            positive=float(e.get("positive", 0)),
            negative=float(e.get("negative", 0)),
            anger=float(e.get("anger", 0)),
            fear=float(e.get("fear", 0)),
            trust=float(e.get("trust", 0)),
            joy=float(e.get("joy", 0)),
            disgust=float(e.get("disgust", 0)),
            surprise=float(e.get("surprise", 0)),
            anticipation=float(e.get("anticipation", 0)),
            sadness=float(e.get("sadness", 0)),
        )

    def score_article(self, article: ProcessedArticle) -> SentimentScores:
        """Return the three raw lexicon scores for one article."""
        nrc = self.score_nrc(article.nrc_tokens)
        return SentimentScores(
            vader=self.score_vader(article.vader_text),
            sentiwordnet=self.score_sentiwordnet(article.sentiwordnet_tokens),

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

    def _lookup_sentiwordnet_score(self, lemma: str, pos: str) -> float | None:
        """Return SentiWordNet score for a lemma with a specific WordNet POS."""
        try:
            synsets = self.wordnet.synsets(lemma, pos=pos)
        except LookupError:
            raise

        if not synsets:
            return None

        senti_synset = self.sentiwordnet.senti_synset(synsets[0].name())
        return float(senti_synset.pos_score() - senti_synset.neg_score())
