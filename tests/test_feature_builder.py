import sys
import types
import unittest

try:
    import spacy  # noqa: F401
except ModuleNotFoundError:
    fake_spacy = types.ModuleType("spacy")
    fake_spacy_tokens = types.ModuleType("spacy.tokens")

    class _StubDoc:
        pass

    fake_spacy_tokens.Doc = _StubDoc
    fake_spacy.tokens = fake_spacy_tokens
    fake_spacy.load = lambda _: None

    sys.modules["spacy"] = fake_spacy
    sys.modules["spacy.tokens"] = fake_spacy_tokens

from src.bias.feature_builder import FeatureBuilder
from src.preprocessing.spacy_processor import ProcessedArticle


class FakeToken:
    def __init__(self, text, pos="NOUN", dep="nsubj", lemma=None):
        self.text = text
        self.lower_ = text.lower()
        self.lemma_ = lemma if lemma is not None else text.lower()
        self.pos_ = pos
        self.dep_ = dep
        self.is_stop = False
        self.is_punct = False
        self.is_alpha = text.isalpha()
        self.is_space = not text.strip()


class FakeEntity:
    def __init__(self, label):
        self.label_ = label


class FakeDoc:
    def __init__(self, tokens, ents=None):
        self._tokens = tokens
        self.ents = ents or []

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


def _make_article(article_id, tokens, ents=None):
    doc = FakeDoc(tokens, ents=ents)
    article = object.__new__(ProcessedArticle)
    article.article_id = article_id
    article.raw_text = " ".join(token.text for token in tokens)
    article.doc = doc
    return article


class FeatureBuilderTests(unittest.TestCase):
    def test_adj_rate_counts_adjectives(self):
        tokens = [FakeToken(f"w{i}") for i in range(8)] + [
            FakeToken("bright", pos="ADJ"),
            FakeToken("calm", pos="ADJ"),
        ]
        article = _make_article("a1", tokens)
        builder = FeatureBuilder()

        self.assertEqual(builder._adj_rate(article.doc), 20.0)

    def test_adv_rate_counts_adverbs(self):
        tokens = [FakeToken(f"w{i}") for i in range(7)] + [
            FakeToken("quickly", pos="ADV"),
            FakeToken("clearly", pos="ADV"),
            FakeToken("firmly", pos="ADV"),
        ]
        article = _make_article("a1", tokens)
        builder = FeatureBuilder()

        self.assertEqual(builder._adv_rate(article.doc), 30.0)

    def test_modal_rate_counts_modals(self):
        tokens = [FakeToken(f"w{i}") for i in range(8)] + [
            FakeToken("would"),
            FakeToken("must"),
        ]
        article = _make_article("a1", tokens)
        builder = FeatureBuilder()

        self.assertEqual(builder._modal_rate(article.doc), 20.0)

    def test_attribution_rate_uses_lemma(self):
        tokens = [FakeToken(f"w{i}") for i in range(9)] + [
            FakeToken("claimed", lemma="claim"),
        ]
        article = _make_article("a1", tokens)
        builder = FeatureBuilder()

        self.assertEqual(builder._attribution_rate(article.doc), 10.0)

    def test_passive_rate_counts_passive_deps(self):
        tokens = [FakeToken(f"w{i}") for i in range(8)] + [
            FakeToken("was", dep="auxpass"),
            FakeToken("written", dep="nsubjpass"),
        ]
        article = _make_article("a1", tokens)
        builder = FeatureBuilder()

        self.assertEqual(builder._passive_rate(article.doc), 20.0)

    def test_ner_rate_counts_target_entities(self):
        tokens = [FakeToken(f"w{i}") for i in range(10)]
        entities = [FakeEntity("PERSON"), FakeEntity("ORG"), FakeEntity("DATE")]
        article = _make_article("a1", tokens, ents=entities)
        builder = FeatureBuilder()

        self.assertEqual(builder._ner_rate(article.doc), 20.0)

    def test_build_returns_correct_shape(self):
        article_1 = _make_article(
            "a1",
            [
                FakeToken("alpha"),
                FakeToken("bravo"),
                FakeToken("charlie"),
                FakeToken("delta"),
                FakeToken("echo"),
                FakeToken("foxtrot"),
            ],
        )
        article_2 = _make_article(
            "a2",
            [
                FakeToken("golf"),
                FakeToken("hotel"),
                FakeToken("india"),
                FakeToken("juliet"),
                FakeToken("kilo"),
                FakeToken("lima"),
            ],
        )
        article_3 = _make_article(
            "a3",
            [
                FakeToken("mike"),
                FakeToken("november"),
                FakeToken("oscar"),
                FakeToken("papa"),
                FakeToken("quebec"),
                FakeToken("romeo"),
            ],
        )
        builder = FeatureBuilder(max_tfidf_features=300)

        matrix = builder.build([article_1, article_2, article_3])
        names = builder.feature_names
        tfidf_count = len(builder._tfidf.get_feature_names_out())

        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], len(names))
        self.assertEqual(matrix.shape[1], tfidf_count + 6)

    def test_build_raises_on_single_article(self):
        article = _make_article("a1", [FakeToken("alpha"), FakeToken("beta")])
        builder = FeatureBuilder()

        with self.assertRaisesRegex(ValueError, "Need at least 2 articles for TF-IDF; got 1."):
            builder.build([article])


if __name__ == "__main__":
    unittest.main()
