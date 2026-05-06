import unittest

import pandas as pd

from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer
from src.sentiment.scaling import scale_sentiment


class TestSentiment(unittest.TestCase):
    def test_score_vader_chunks_positive_negative_bounds_and_empty(self) -> None:
        scorer = LexiconScorer()
        positive_chunks = ["This outcome is excellent and wonderful."]
        negative_chunks = ["This outcome is terrible and awful."]

        positive_score = scorer.score_vader_chunks(positive_chunks)
        negative_score = scorer.score_vader_chunks(negative_chunks)
        empty_score = scorer.score_vader_chunks([])

        self.assertGreater(positive_score, 0.0)
        self.assertLess(negative_score, 0.0)
        self.assertTrue(-1.0 <= positive_score <= 1.0)
        self.assertTrue(-1.0 <= negative_score <= 1.0)
        self.assertEqual(empty_score, 0.0)

    def test_scale_sentiment_creates_z_columns_zero_mean_no_composite(self) -> None:
        df = pd.DataFrame(
            {
                "article_id": ["a1", "a2", "a3"],
                "vader_score": [0.2, 0.5, 0.8],
                "zeroshot_score": [-0.6, 0.0, 0.6],
            }
        )

        result = scale_sentiment(
            df, polarity_columns=("vader_score", "zeroshot_score")
        )

        self.assertIn("vader_z", result.columns)
        self.assertIn("zeroshot_z", result.columns)
        self.assertAlmostEqual(float(result["vader_z"].mean()), 0.0, places=7)
        self.assertAlmostEqual(float(result["zeroshot_z"].mean()), 0.0, places=7)
        self.assertFalse(any("composite" in column for column in result.columns))


if __name__ == "__main__":
    unittest.main()
