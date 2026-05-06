import unittest

import pandas as pd

from src.preprocessing.filters import (
    filter_opinion_pieces,
    filter_shamima_mentions,
    filter_short_articles,
)


class TestFilters(unittest.TestCase):
    def test_filter_shamima_mentions_boundary_and_case_insensitive(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "article_id": "a0",
                    "title": "Policy update",
                    "body": "This article has no target name.",
                },
                {
                    "article_id": "a1",
                    "title": "Shamima Begum update",
                    "body": "Only one mention in total.",
                },
                {
                    "article_id": "a2",
                    "title": "SHAMIMA BEGUM timeline",
                    "body": "Another line about Shamima Begum.",
                },
            ]
        )

        result = filter_shamima_mentions(df)
        self.assertEqual(result["article_id"].tolist(), ["a2"])

    def test_filter_short_articles_word_count_boundaries_and_empty(self) -> None:
        df = pd.DataFrame(
            [
                {"article_id": "w249", "title": "Short", "body": " ".join(["word"] * 249)},
                {"article_id": "w250", "title": "Boundary", "body": " ".join(["word"] * 250)},
                {"article_id": "w251", "title": "Long", "body": " ".join(["word"] * 251)},
                {"article_id": "empty", "title": "Empty", "body": ""},
            ]
        )

        result = filter_short_articles(df)
        self.assertEqual(result["article_id"].tolist(), ["w250", "w251"])

    def test_filter_opinion_pieces_url_and_title_markers(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "article_id": "u1",
                    "date_link": "https://example.com/commentisfree/shamima-piece",
                    "title": "News analysis",
                },
                {
                    "article_id": "u2",
                    "date_link": "https://example.com/voices/shamima-piece",
                    "title": "Another report",
                },
                {
                    "article_id": "t1",
                    "date_link": "https://example.com/news/story-1",
                    "title": "Opinion: Why this matters",
                },
                {
                    "article_id": "t2",
                    "date_link": "https://example.com/news/story-2",
                    "title": "Policy debate | letters",
                },
                {
                    "article_id": "clean",
                    "date_link": "https://example.com/news/story-3",
                    "title": "Government statement on case",
                },
            ]
        )

        result = filter_opinion_pieces(df)
        self.assertEqual(result["article_id"].tolist(), ["clean"])


if __name__ == "__main__":
    unittest.main()
