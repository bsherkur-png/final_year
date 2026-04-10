import unittest

import pandas as pd

from src.preprocessing.article_preprocessor import ArticlePreprocessor


class FakeToken:
    def __init__(self, lemma, is_stop=False, is_punct=False, is_space=False):
        self.lemma_ = lemma
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.is_space = is_space


class FakeNLP:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return self.docs[text]

    def pipe(self, texts, batch_size=32):
        for text in texts:
            self.calls.append(text)
            yield self.docs[text]


class ArticlePreprocessorTests(unittest.TestCase):
    def test_preprocess_body_tokenizes_and_filters_with_spacy_attributes(self):
        normalized = "this is a sample article, with extra spaces!"
        fake_nlp = FakeNLP(
            {
                normalized: [
                    FakeToken("this", is_stop=True),
                    FakeToken("be", is_stop=True),
                    FakeToken("a", is_stop=True),
                    FakeToken("sample"),
                    FakeToken("article"),
                    FakeToken(",", is_punct=True),
                    FakeToken("with", is_stop=True),
                    FakeToken("extra"),
                    FakeToken(" ", is_space=True),
                    FakeToken("space"),
                    FakeToken("!", is_punct=True),
                ]
            }
        )
        preprocessor = ArticlePreprocessor(nlp=fake_nlp)

        processed = preprocessor.preprocess_body("This is a sample article,   with extra spaces!")

        self.assertEqual(processed, "sample article extra space")
        self.assertEqual(fake_nlp.calls, [normalized])

    def test_preprocess_bodies_uses_pipe_and_preserves_empty_body(self):
        fake_nlp = FakeNLP(
            {
                "markets are falling fast": [
                    FakeToken("market"),
                    FakeToken("be", is_stop=True),
                    FakeToken("fall"),
                    FakeToken("fast"),
                ],
                "": [],
            }
        )
        preprocessor = ArticlePreprocessor(nlp=fake_nlp)

        processed = preprocessor.preprocess_bodies(["Markets are falling fast", "   "])

        self.assertEqual(processed, ["market fall fast", ""])
        self.assertEqual(fake_nlp.calls, ["markets are falling fast", ""])

    def test_prepare_titles_for_clustering_preserves_identical_duplicate_rows(self):
        preprocessor = ArticlePreprocessor()
        df = pd.DataFrame(
            {
                "article_id": ["a1", "a1", "a2"],
                "title": ["Budget debate today", "Budget debate today", "Migration policy debate"],
            }
        )

        prepared = preprocessor.prepare_titles_for_clustering(df)

        self.assertEqual(list(prepared.columns), ["article_id", "title", "processed_title"])
        self.assertEqual(len(prepared), 3)
        self.assertEqual(prepared.loc[0, "article_id"], "a1")
        self.assertEqual(prepared.loc[1, "article_id"], "a1")
        self.assertEqual(prepared.loc[0, "title"], "Budget debate today")
        self.assertEqual(prepared.loc[1, "title"], "Budget debate today")

    def test_prepare_titles_for_clustering_preserves_same_title_across_article_ids(self):
        preprocessor = ArticlePreprocessor()
        df = pd.DataFrame(
            {
                "article_id": ["a10", "a11"],
                "title": ["Migration policy debate", "Migration policy debate"],
            }
        )

        prepared = preprocessor.prepare_titles_for_clustering(df)

        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared["article_id"].tolist(), ["a10", "a11"])
        self.assertEqual(prepared["title"].tolist(), ["Migration policy debate", "Migration policy debate"])

    def test_prepare_titles_for_clustering_filters_null_and_blank_titles(self):
        preprocessor = ArticlePreprocessor()
        df = pd.DataFrame(
            {
                "article_id": ["a1", "a2", "a3", "a4"],
                "title": [None, "", "   ", "Valid headline"],
            }
        )

        prepared = preprocessor.prepare_titles_for_clustering(df)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.loc[0, "article_id"], "a4")
        self.assertEqual(prepared.loc[0, "title"], "Valid headline")

    def test_prepare_titles_for_clustering_filters_blank_processed_titles(self):
        preprocessor = ArticlePreprocessor()
        df = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "title": ["the and", "Clear policy plan"],
            }
        )

        prepared = preprocessor.prepare_titles_for_clustering(df)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.loc[0, "article_id"], "a2")
        self.assertEqual(prepared.loc[0, "title"], "Clear policy plan")
        self.assertNotEqual(prepared.loc[0, "processed_title"].strip(), "")


if __name__ == "__main__":
    unittest.main()
