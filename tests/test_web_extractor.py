import unittest
from unittest.mock import patch

import pandas as pd

from src.extraction.web_extractor import ArticleHtmlParser, WebExtractor


class FakeResponse:
    def __init__(self, text: str, error: Exception | None = None):
        self.text = text
        self._error = error
        self.encoding = None
        self.apparent_encoding = "utf-8"

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


class ArticleHtmlParserTests(unittest.TestCase):
    def test_extract_text_prefers_article_container(self):
        html = """
        <html>
            <body>
                <article>
                    <p>First paragraph.</p>
                    <p>Second paragraph.</p>
                </article>
                <p>Ignored outside article.</p>
            </body>
        </html>
        """

        parser = ArticleHtmlParser()

        extracted = parser.extract_text(html)

        self.assertEqual(extracted, "First paragraph. Second paragraph.")


class WebExtractorTests(unittest.TestCase):
    def test_fetch_page_uses_session_and_timeout(self):
        extractor = WebExtractor(timeout_seconds=9)
        extractor.session = FakeSession({"https://example.com": FakeResponse("<html></html>")})

        html = extractor.fetch_page("https://example.com")

        self.assertEqual(html, "<html></html>")
        self.assertEqual(extractor.session.calls, [("https://example.com", 9)])

    def test_extract_populates_text_and_handles_failures(self):
        extractor = WebExtractor(delay_seconds=2)
        extractor.session = FakeSession(
            {
                "https://ok.test": FakeResponse("<article><p>Alpha</p><p>Beta</p></article>"),
                "https://bad.test": RuntimeError("network error"),
            }
        )
        df = pd.DataFrame(
            {
                "article_id": ["a1", "a2"],
                "news_outlet": ["BBC", "BBC"],
                "title": ["One", "Two"],
                "date_link": ["https://ok.test", "https://bad.test"],
            }
        )

        with patch("src.extraction.web_extractor.time.sleep") as sleep_mock:
            extracted = extractor.extract(df)

        self.assertEqual(
            list(extracted.columns),
            ["article_id", "news_outlet", "title", "date_link", "body", "fetch_status", "fetch_error"],
        )
        self.assertEqual(extracted.loc[0, "body"], "Alpha Beta")
        self.assertEqual(extracted.loc[0, "fetch_status"], "ok")
        self.assertEqual(extracted.loc[0, "fetch_error"], "")
        self.assertEqual(extracted.loc[1, "body"], "")
        self.assertEqual(extracted.loc[1, "fetch_status"], "error:RuntimeError")
        self.assertEqual(extracted.loc[1, "fetch_error"], "network error")
        sleep_mock.assert_called_once_with(2)

    def test_extract_requires_url_column(self):
        extractor = WebExtractor()
        df = pd.DataFrame(
            {
                "article_id": ["a1"],
                "news_outlet": ["BBC"],
                "title": ["One"],
                "link": ["https://example.com"],
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing required columns: date_link"):
            extractor.extract(df)


if __name__ == "__main__":
    unittest.main()
