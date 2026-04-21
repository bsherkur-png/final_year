from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.spacy_processor import ProcessedArticle

MODAL_VERBS = frozenset({"would", "could", "might", "must", "shall", "should"})

ATTRIBUTION_VERBS = frozenset(
    {
        "claim",
        "admit",
        "insist",
        "deny",
        "argue",
        "assert",
        "allege",
        "declare",
        "suggest",
        "contend",
    }
)

PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass"})

TARGET_ENTITY_LABELS = frozenset({"PERSON", "ORG", "GPE"})

WEB_BOILERPLATE = frozenset(
    {
        # Image/media captions and embeds
        "image",
        "picture",
        "video",
        "play",
        "watch",
        "caption",
        "getty",
        "getty images",
        "pa",
        "reuters",
        "afp",
        "epa",
        "source",
        "credit",
        "copyright",
        # Social and sharing widgets
        "share",
        "facebook",
        "twitter",
        "whatsapp",
        "email",
        "comment",
        "comments",
        "like",
        "follow",
        "subscribe",
        # Navigation and layout
        "card",
        "target",
        "range",
        "skip",
        "menu",
        "search",
        "cookie",
        "cookies",
        "accept",
        "privacy",
        "policy",
        "sign",
        "log",
        "register",
        "newsletter",
        # Related content and cross-promotion
        "read",
        "related",
        "stories",
        "topic",
        "topics",
        "advertisement",
        "sponsored",
        "ad",
        # Broadcast boilerplate
        "itv",
        "morning",
        "bbc radio",
        "channel",
        # UI elements
        "click",
        "tap",
        "swipe",
        "close",
        "open",
        "view",
        "more",
        "show",
        "hide",
        "expand",
        "bbc"
    }
)


@dataclass
class ArticleFeatures:
    """Six linguistic features for one article, computed from its spaCy Doc."""

    adj_rate: float
    adv_rate: float
    modal_rate: float
    attribution_rate: float
    passive_rate: float
    ner_rate: float


class FeatureBuilder:
    """Build TF-IDF feature matrices and linguistic descriptive profiles."""

    def __init__(self, max_tfidf_features: int = 300):
        """Initialize vectorizer for feature extraction."""
        self._tfidf = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            lowercase=False,
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            max_df=0.5,
            min_df=3,
        )
        self._is_fitted = False

    @staticmethod
    def _filter_boilerplate(tokens: list[str]) -> list[str]:
        """Remove known web boilerplate terms from a token list."""
        return [t for t in tokens if t not in WEB_BOILERPLATE]

    def build(self, articles: list[ProcessedArticle]) -> csr_matrix:
        """Fit TF-IDF and return the document-term matrix."""
        if len(articles) < 2:
            raise ValueError(
                f"Need at least 2 articles for TF-IDF; got {len(articles)}."
            )

        corpus = [self._filter_boilerplate(article.lemmas) for article in articles]
        tfidf_matrix = self._tfidf.fit_transform(corpus)

        self._is_fitted = True
        return tfidf_matrix

    @property
    def feature_names(self) -> list[str]:
        """Return fitted TF-IDF feature names."""
        if not self._is_fitted:
            raise RuntimeError("Call build() before accessing feature_names.")
        return list(self._tfidf.get_feature_names_out())

    def linguistic_profile(self, articles: list[ProcessedArticle]) -> pd.DataFrame:
        """Compute six linguistic rates per article as a DataFrame."""
        rows = []
        for article in articles:
            features = self._extract_linguistic(article)
            rows.append({
                "article_id": article.article_id,
                "adj_rate": features.adj_rate,
                "adv_rate": features.adv_rate,
                "modal_rate": features.modal_rate,
                "attribution_rate": features.attribution_rate,
                "passive_rate": features.passive_rate,
                "ner_rate": features.ner_rate,
            })
        return pd.DataFrame(rows)

    def _extract_linguistic(self, article: ProcessedArticle) -> ArticleFeatures:
        """Extract six linguistic rates from one processed article."""
        doc = article.doc
        return ArticleFeatures(
            adj_rate=self._adj_rate(doc),
            adv_rate=self._adv_rate(doc),
            modal_rate=self._modal_rate(doc),
            attribution_rate=self._attribution_rate(doc),
            passive_rate=self._passive_rate(doc),
            ner_rate=self._ner_rate(doc),
        )

    def _adj_rate(self, doc) -> float:
        n_tokens = max(len(doc), 1)
        return sum(1 for token in doc if token.pos_ == "ADJ") / n_tokens * 100

    def _adv_rate(self, doc) -> float:
        n_tokens = max(len(doc), 1)
        return sum(1 for token in doc if token.pos_ == "ADV") / n_tokens * 100

    def _modal_rate(self, doc) -> float:
        n_tokens = max(len(doc), 1)
        return sum(1 for token in doc if token.lower_ in MODAL_VERBS) / n_tokens * 100

    def _attribution_rate(self, doc) -> float:
        n_tokens = max(len(doc), 1)
        return (
                sum(1 for token in doc if token.lemma_.lower() in ATTRIBUTION_VERBS)
                / n_tokens
                * 100
        )

    def _passive_rate(self, doc) -> float:
        n_tokens = max(len(doc), 1)
        return sum(1 for token in doc if token.dep_ in PASSIVE_DEPS) / n_tokens * 100

    def _ner_rate(self, doc) -> float:
        n_tokens = max(len(doc), 1)
        return (
                sum(1 for entity in doc.ents if entity.label_ in TARGET_ENTITY_LABELS)
                / n_tokens
                * 100
        )
