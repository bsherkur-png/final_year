import unittest

from src.preprocessing.spacy_processor import SpacyProcessor


class TestPreprocessing(unittest.TestCase):
    def test_cleaned_text_normalises_whitespace(self) -> None:
        processor = SpacyProcessor()
        raw_text = "  First line.\n\nSecond\t\tline with  spaces.   "

        article = processor.process(article_id="a1", raw_text=raw_text)

        self.assertEqual(article.cleaned_text, "First line. Second line with spaces.")

    def test_cleaned_text_empty_string(self) -> None:
        processor = SpacyProcessor()

        article = processor.process(article_id="a2", raw_text="")

        self.assertEqual(article.cleaned_text, "")

    def test_lemmas_excludes_stopwords_punctuation_and_non_alpha(self) -> None:
        processor = SpacyProcessor()
        text = "The government announced 250 new changes."

        article = processor.process(article_id="a3", raw_text=text)
        lemmas = article.lemmas

        self.assertNotIn("the", lemmas)
        self.assertNotIn(".", lemmas)
        self.assertNotIn("250", lemmas)
        self.assertTrue(all(token == token.lower() for token in lemmas))

    def test_chunks_returns_two_chunks_for_eight_sentences(self) -> None:
        processor = SpacyProcessor()
        text = (
            "First sentence has enough clear words. "
            "Second sentence also has enough words. "
            "Third sentence keeps enough alpha words. "
            "Fourth sentence meets the word threshold. "
            "Fifth sentence includes many descriptive words. "
            "Sixth sentence keeps this test simple today. "
            "Seventh sentence continues with ordinary wording. "
            "Eighth sentence closes this fabricated article."
        )

        article = processor.process(article_id="a4", raw_text=text)

        self.assertEqual(len(article.chunks), 2)

    def test_chunks_returns_one_chunk_for_four_sentences(self) -> None:
        processor = SpacyProcessor()
        text = (
            "One sentence has enough words for chunking. "
            "Two sentence has enough words for chunking. "
            "Three sentence has enough words for chunking. "
            "Four sentence has enough words for chunking."
            "5 sentence has enough words."
            "Hello friend."
        )

        article = processor.process(article_id="a5", raw_text=text)

        self.assertEqual(len(article.chunks), 1)


if __name__ == "__main__":
    unittest.main()
