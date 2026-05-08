import unittest

from src.sentiment.zeroshot_scorer import ZeroshotScorer


class TestZeroshotScorer(unittest.TestCase):
    def test_score_chunks_positive_negative_bounds_and_empty(self) -> None:
        scorer = ZeroshotScorer()

        positive_chunks = ["This outcome is excellent and wonderful news."]
        negative_chunks = ["This outcome is terrible and devastating news."]
        empty_chunks = []

        positive_score = scorer.score_chunks(positive_chunks)
        negative_score = scorer.score_chunks(negative_chunks)
        empty_score = scorer.score_chunks(empty_chunks)

        self.assertGreater(positive_score, 0.0)
        self.assertLess(negative_score, 0.0)
        self.assertTrue(-1.0 <= positive_score <= 1.0)
        self.assertTrue(-1.0 <= negative_score <= 1.0)
        self.assertEqual(empty_score, 0.0)


if __name__ == "__main__":
    unittest.main()
