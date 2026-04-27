from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.spacy_processor import ProcessedArticle

WEB_BOILERPLATE = frozenset(
    {
        *(
            "image picture video play watch caption getty pa reuters afp epa source "
            "credit copyright share facebook twitter whatsapp email comment comments "
            "like follow subscribe card target range skip menu search cookie cookies "
            "accept privacy policy sign log register newsletter read related stories "
            "topic topics advertisement sponsored ad itv morning channel click tap "
            "swipe close open view more show hide expand bbc"
        ).split(),
        "getty images",
        "bbc radio",
    }
)


class FeatureBuilder:
    """Build TF-IDF feature matrices from processed article lemmas."""

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
        return tfidf_matrix

    @property
    def feature_names(self) -> list[str]:
        """Return fitted TF-IDF feature names."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self._tfidf)
        return list(self._tfidf.get_feature_names_out())
