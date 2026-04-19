from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

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
    """Build combined TF-IDF and linguistic feature matrices."""

    def __init__(self, max_tfidf_features: int = 300):
        """Initialize vectorizer and scaler for feature extraction."""
        self._tfidf = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            lowercase=False,
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
        )
        self._scaler = StandardScaler()
        self._is_fitted = False

    def build(self, articles: list[ProcessedArticle]) -> csr_matrix:
        """Fit TF-IDF + linguistic features and return a combined sparse matrix."""
        if len(articles) < 2:
            raise ValueError(
                f"Need at least 2 articles for TF-IDF; got {len(articles)}."
            )

        corpus = [article.lemmas for article in articles]
        tfidf_matrix = self._tfidf.fit_transform(corpus)

        linguistic_rows = [self._extract_linguistic(article) for article in articles]
        linguistic_array = np.array(
            [
                [
                    features.adj_rate,
                    features.adv_rate,
                    features.modal_rate,
                    features.attribution_rate,
                    features.passive_rate,
                    features.ner_rate,
                ]
                for features in linguistic_rows
            ]
        )
        linguistic_scaled = self._scaler.fit_transform(linguistic_array)

        linguistic_sparse = csr_matrix(linguistic_scaled)
        combined = hstack([tfidf_matrix, linguistic_sparse], format="csr")

        self._is_fitted = True
        return combined

    @property
    def feature_names(self) -> list[str]:
        """Return fitted TF-IDF feature names followed by six linguistic names."""
        if not self._is_fitted:
            raise RuntimeError("Call build() before accessing feature_names.")

        tfidf_names = list(self._tfidf.get_feature_names_out())
        linguistic_names = [
            "adj_rate",
            "adv_rate",
            "modal_rate",
            "attribution_rate",
            "passive_rate",
            "ner_rate",
        ]
        return tfidf_names + linguistic_names

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
