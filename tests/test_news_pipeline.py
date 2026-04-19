import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

try:
    import spacy  # noqa: F401
except ModuleNotFoundError:
    fake_spacy = types.ModuleType("spacy")
    fake_spacy_language = types.ModuleType("spacy.language")
    fake_spacy_tokens = types.ModuleType("spacy.tokens")

    class _StubLanguage:
        pass

    class _StubDoc:
        pass

    fake_spacy_language.Language = _StubLanguage
    fake_spacy_tokens.Doc = _StubDoc
    fake_spacy.load = lambda _: None

    sys.modules["spacy"] = fake_spacy
    sys.modules["spacy.language"] = fake_spacy_language
    sys.modules["spacy.tokens"] = fake_spacy_tokens

from src.pipeline.news_pipeline import NewsPipeline


class FakeProcessedArticle:
    def __init__(self, article_id, raw_text=""):
        self.article_id = article_id
        self.raw_text = raw_text
        self.doc = []
        self.minimal_text = raw_text.strip()
        self.lemmas = raw_text.lower().split() if raw_text.strip() else []


class FakeSentimentScores:
    def __init__(self):
        self.vader = 0.5
        self.sentiwordnet = 0.2
        self.nrc = -1.0


class FakeSpacyProcessor:
    def __init__(self):
        self.last_dataframe = None
        self.last_body_column = None

    def process_dataframe(self, dataframe, body_column="body"):
        self.last_dataframe = dataframe.copy()
        self.last_body_column = body_column
        return [
            FakeProcessedArticle(article_id=row["article_id"], raw_text=row[body_column])
            for _, row in dataframe.iterrows()
        ]


class NewsPipelineTests(unittest.TestCase):
    def test_process_with_spacy_calls_processor(self):
        pipeline = NewsPipeline(output_dir=tempfile.mkdtemp())
        dataframe = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "body": ["One body", "Two body"],
            }
        )
        fake_processor = FakeSpacyProcessor()

        with patch("src.pipeline.news_pipeline.SpacyProcessor", return_value=fake_processor):
            articles = pipeline._process_with_spacy(dataframe)

        self.assertEqual([article.article_id for article in articles], ["a1", "a2"])
        self.assertEqual(fake_processor.last_body_column, "body")

    def test_score_sentiment_maps_scores_to_dataframe(self):
        pipeline = NewsPipeline(output_dir=tempfile.mkdtemp())
        dataframe = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "news_outlet": ["BBC", "Guardian"],
                "title": ["T1", "T2"],
                "date_link": ["u1", "u2"],
            }
        )
        articles = [FakeProcessedArticle("a1", "Alpha"), FakeProcessedArticle("a2", "Beta")]

        with patch("src.pipeline.news_pipeline.LexiconScorer") as scorer_cls:
            scorer_cls.return_value.score_article.return_value = FakeSentimentScores()
            scored = pipeline._score_sentiment(articles, dataframe)

        self.assertEqual(scored["vader_score"].tolist(), [0.5, 0.5])
        self.assertEqual(scored["sentiwordnet_score"].tolist(), [0.2, 0.2])
        self.assertEqual(scored["nrc_score"].tolist(), [-1.0, -1.0])

    def test_run_preprocessing_returns_articles_and_writes_checkpoint(self):
        output_dir = tempfile.mkdtemp()
        pipeline = NewsPipeline(output_dir=output_dir)
        dataframe = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "body": ["First article", "Second article"],
                "news_outlet": ["BBC", "Guardian"],
                "title": ["Title 1", "Title 2"],
                "date_link": ["u1", "u2"],
            }
        )
        fake_processor = FakeSpacyProcessor()

        with patch("src.pipeline.news_pipeline.SpacyProcessor", return_value=fake_processor):
            articles = pipeline.run_preprocessing(dataframe)

        self.assertEqual([article.article_id for article in articles], ["a1", "a2"])
        self.assertTrue(Path(pipeline.preprocess_output_path).exists())

    def test_run_raw_sentiment_writes_csv_with_expected_columns(self):
        output_dir = tempfile.mkdtemp()
        pipeline = NewsPipeline(output_dir=output_dir)
        dataframe = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "news_outlet": ["BBC", "Guardian"],
                "title": ["Title 1", "Title 2"],
                "date_link": ["u1", "u2"],
            }
        )
        articles = [FakeProcessedArticle("a1", "First"), FakeProcessedArticle("a2", "Second")]

        with patch("src.pipeline.news_pipeline.LexiconScorer") as scorer_cls:
            scorer_cls.return_value.score_article.return_value = FakeSentimentScores()
            pipeline.run_raw_sentiment(articles, dataframe)

        written = pd.read_csv(pipeline.raw_sentiment_output_path)
        self.assertEqual(
            list(written.columns),
            [
                "article_id",
                "news_outlet",
                "title",
                "date_link",
                "vader_score",
                "sentiwordnet_score",
                "nrc_score",
            ],
        )


if __name__ == "__main__":
    unittest.main()
