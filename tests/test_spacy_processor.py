import unittest
import sys
import types

import pandas as pd

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

from src.preprocessing.spacy_processor import ProcessedArticle, SpacyProcessor


class FakeToken:
    def __init__(self, text, lemma=None, is_stop=False, is_punct=False):
        self.text = text
        self.lemma_ = lemma if lemma is not None else text.lower()
        self.is_space = not str(text).strip()
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.is_alpha = str(text).isalpha()


class FakeDoc(list):
    """List of FakeTokens that behaves like a spaCy Doc for iteration."""


class FakeNlp:
    def __init__(self):
        self.last_doc = None

    def __call__(self, text):
        self.last_doc = FakeDoc([FakeToken(w) for w in str(text).split()] if text.strip() else [])
        return self.last_doc


class ProcessedArticleTests(unittest.TestCase):
    def test_minimal_text_strips_whitespace(self):
        article = ProcessedArticle("a1", "  Hello  ", FakeDoc([FakeToken("Hello")]))

        self.assertEqual(article.minimal_text, "Hello")

    def test_lemmas_excludes_stopwords(self):
        article = ProcessedArticle(
            "a1",
            "The story",
            FakeDoc([FakeToken("The", is_stop=True), FakeToken("story", lemma="story")]),
        )

        self.assertEqual(article.lemmas, ["story"])

    def test_lemmas_excludes_punctuation(self):
        article = ProcessedArticle(
            "a1",
            "news .",
            FakeDoc([FakeToken("news", lemma="news"), FakeToken(".", lemma=".", is_punct=True)]),
        )

        self.assertEqual(article.lemmas, ["news"])

    def test_lemmas_lowercases(self):
        article = ProcessedArticle("a1", "Running", FakeDoc([FakeToken("Running", lemma="Run")]))

        self.assertEqual(article.lemmas, ["run"])

    def test_tokens_keeps_stopwords_drops_punctuation(self):
        article = ProcessedArticle(
            "a1",
            "the news !",
            FakeDoc(
                [
                    FakeToken("the", is_stop=True),
                    FakeToken("news", lemma="news"),
                    FakeToken("!", lemma="!", is_punct=True),
                ]
            ),
        )

        self.assertEqual(article.tokens, ["the", "news"])
        self.assertEqual(article.lemmas, ["news"])

    def test_empty_text_gives_empty_properties(self):
        article = ProcessedArticle("a1", "", FakeDoc([]))

        self.assertEqual(article.minimal_text, "")
        self.assertEqual(article.lemmas, [])
        self.assertEqual(article.tokens, [])


class SpacyProcessorTests(unittest.TestCase):
    def test_process_uses_injected_nlp(self):
        fake_nlp = FakeNlp()
        processor = SpacyProcessor(nlp=fake_nlp)

        article = processor.process("a1", "hello world")

        self.assertIs(article.doc, fake_nlp.last_doc)

    def test_process_handles_none_input(self):
        processor = SpacyProcessor(nlp=FakeNlp())

        article = processor.process("a1", None)

        self.assertEqual(article.raw_text, "")
        self.assertEqual(article.doc, FakeDoc([]))

    def test_process_dataframe_returns_correct_count(self):
        processor = SpacyProcessor(nlp=FakeNlp())
        dataframe = pd.DataFrame(
            {"article_id": ["a1", "a2", "a3"], "body": ["first", "second", "third"]}
        )

        articles = processor.process_dataframe(dataframe)

        self.assertEqual(len(articles), 3)

    def test_process_dataframe_raises_on_missing_column(self):
        processor = SpacyProcessor(nlp=FakeNlp())
        dataframe = pd.DataFrame({"article_id": ["a1"], "text": ["content"]})

        with self.assertRaisesRegex(ValueError, "Missing required column: body"):
            processor.process_dataframe(dataframe)


if __name__ == "__main__":
    unittest.main()
