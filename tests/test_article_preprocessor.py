import unittest

import pandas as pd

from src.preprocessing.article_preprocessor import ArticlePreprocessor, ShamimaBegumFilter


class FakeToken:
    def __init__(self, text):
        self.text = text
        self.is_space = not str(text).strip()


class FakeNlp:
    def __call__(self, text):
        return [FakeToken(token) for token in str(text).split()]


class ArticlePreprocessorTests(unittest.TestCase):
    def test_ensure_nlp_returns_existing_model(self):
        fake_nlp = FakeNlp()
        preprocessor = ArticlePreprocessor(nlp=fake_nlp)

        result = preprocessor._ensure_nlp()

        self.assertIs(result, fake_nlp)


class ShamimaBegumFilterTests(unittest.TestCase):
    def test_count_mentions_uses_exact_phrase(self):
        preprocessor = ArticlePreprocessor(nlp=FakeNlp())
        article_filter = ShamimaBegumFilter(preprocessor)

        count = article_filter._count_mentions(
            "Shamima Begum was mentioned. shamima begum appeared again. shamima only."
        )

        self.assertEqual(count, 2)

    def test_filter_articles_keeps_rows_with_two_or_more_mentions(self):
        preprocessor = ArticlePreprocessor(nlp=FakeNlp())
        article_filter = ShamimaBegumFilter(preprocessor)
        articles = pd.DataFrame(
            {
                "article_id": ["a1", "a2", "a3"],
                "body": [
                    "shamima begum once.",
                    "shamima begum twice shamima begum.",
                    "other content",
                ],
            }
        )

        filtered = article_filter.filter_articles(articles)

        self.assertEqual(filtered["article_id"].tolist(), ["a2"])

    def test_filter_articles_requires_body_column(self):
        preprocessor = ArticlePreprocessor(nlp=FakeNlp())
        article_filter = ShamimaBegumFilter(preprocessor)
        articles = pd.DataFrame({"text": ["shamima begum shamima begum"]})

        with self.assertRaisesRegex(ValueError, "body"):
            article_filter.filter_articles(articles)


if __name__ == "__main__":
    unittest.main()
