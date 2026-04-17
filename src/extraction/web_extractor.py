import time

import pandas as pd
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class ArticleHtmlParser:
    def extract_text(self, html: str) -> str:
        if BeautifulSoup is None:
            return self._extract_text_fallback(html)

        soup = BeautifulSoup(html, "html.parser")
        container = soup.find(class_="main v-sep") or soup.find("article")

        if container:
            paragraphs = container.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        texts = [paragraph.get_text(" ", strip=True) for paragraph in paragraphs]
        return " ".join(texts)

    @staticmethod
    def _extract_text_fallback(html: str) -> str:
        import re

        article_match = re.search(r"<article\b[^>]*>(.*?)</article>", html, flags=re.IGNORECASE | re.DOTALL)
        source_html = article_match.group(1) if article_match else html
        matches = re.findall(r"<p[^>]*>(.*?)</p>", source_html, flags=re.IGNORECASE | re.DOTALL)
        cleaned = [re.sub(r"<[^>]+>", " ", match) for match in matches]
        cleaned = [re.sub(r"\s+", " ", text).strip() for text in cleaned]
        return " ".join(text for text in cleaned if text)


class WebExtractor:
    def __init__(
        self,
        delay_seconds: int = 5,
        timeout_seconds: int = 5,
    ):
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.parser = ArticleHtmlParser()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def fetch_page(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.content

    def extract_text(self, html: str) -> str:
        return self.parser.extract_text(html)

    def extract(self, df: pd.DataFrame, url_column: str = "date_link") -> pd.DataFrame:
        required_input_columns = [
            "article_id",
            "news_outlet",
            "title",
            "date_link",
        ]
        missing_columns = [column for column in required_input_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        if url_column not in df.columns:
            raise ValueError(f"Missing required column: {url_column}")

        extracted_df = df.copy()
        if "body" not in extracted_df.columns:
            extracted_df["body"] = ""
        extracted_df["body"] = extracted_df["body"].apply(
            lambda value: "" if pd.isna(value) else str(value)
        )

        extracted_df = extracted_df.loc[
            :,
            ["article_id", "news_outlet", "title", "date_link", "body"],
        ]
        extracted_df = extracted_df.set_index("article_id", drop=False)

        for article_id, url in extracted_df[url_column].items():
            try:
                html = self.fetch_page(url)
                extracted_df.at[article_id, "body"] = self.extract_text(html)
                time.sleep(self.delay_seconds)
            except Exception as exc:
                extracted_df.at[article_id, "body"] = ""
                print(f"Failed at {url}: {exc}")

        return extracted_df.reset_index(drop=True).loc[
            :,
            ["article_id", "news_outlet", "title", "date_link", "body"],
        ]


Extractor = WebExtractor
