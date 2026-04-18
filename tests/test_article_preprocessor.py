import unittest

import pandas as pd

from src.preprocessing.article_preprocessor import ArticlePreprocessor, filter_shamima_mentions


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

    def test_preprocess_dataframe_creates_expected_columns(self):
        preprocessor = ArticlePreprocessor(nlp=FakeNlp())
        dataframe = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "body": ["  Hello   World  ", None],
            }
        )

        result = preprocessor.preprocess_dataframe(dataframe, body_column="body")

        self.assertEqual(
            list(result.columns),
            ["article_id", "body", "original_body_text", "minimal_body_text", "fully_preprocessed_body_text"],
        )
        self.assertEqual(result["original_body_text"].tolist(), ["  Hello   World  ", ""])
        self.assertEqual(result["minimal_body_text"].tolist(), ["Hello   World", ""])
        self.assertEqual(result["fully_preprocessed_body_text"].tolist(), ["hello world", ""])

    def test_preprocess_dataframe_uses_original_body_text_by_default(self):
        preprocessor = ArticlePreprocessor(nlp=FakeNlp())
        dataframe = pd.DataFrame({"original_body_text": ["  Mixed CASE  "]})

        result = preprocessor.preprocess_dataframe(dataframe)

        self.assertEqual(result["minimal_body_text"].tolist(), ["Mixed CASE"])
        self.assertEqual(result["fully_preprocessed_body_text"].tolist(), ["mixed case"])

    def test_preprocess_dataframe_requires_body_column(self):
        preprocessor = ArticlePreprocessor(nlp=FakeNlp())
        dataframe = pd.DataFrame({"text": ["hello world"]})

        with self.assertRaisesRegex(ValueError, "Missing required column: body"):
            preprocessor.preprocess_dataframe(dataframe, body_column="body")


class ShamimaBegumFilterTests(unittest.TestCase):
    def test_filter_articles_counts_mentions_across_title_and_body(self):
        articles = pd.DataFrame(
            {
                "article_id": ["a1", "a2", "a3"],
                "title": [
                    "Shamima Begum update",
                    "Other headline",
                    "No mention",
                ],
                "body": [
                    "shamima begum once in body",
                    "shamima begum appears once in body",
                    "other content",
                ],
            }
        )

        filtered = filter_shamima_mentions(articles)

        self.assertEqual(filtered["article_id"].tolist(), ["a1"])

    def test_filter_articles_requires_title_and_body_columns(self):
        articles = pd.DataFrame({"body": ["shamima begum shamima begum"]})

        with self.assertRaisesRegex(ValueError, "title"):
            filter_shamima_mentions(articles)


if __name__ == "__main__":
    unittest.main()
