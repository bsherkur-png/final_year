import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class ArticleHtmlParser:
    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        container = soup.find(class_="main v-sep") or soup.find("article")

        if container:
            paragraphs = container.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        texts = [paragraph.get_text(" ", strip=True) for paragraph in paragraphs]
        return " ".join(texts)


class WebExtractor:
    def __init__(
        self,
        delay_seconds: int = 5,
        timeout_seconds: int = 15,
    ):
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.parser = ArticleHtmlParser()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def fetch_page(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def extract_text(self, html: str) -> str:
        return self.parser.extract_text(html)

    def extract(self, df: pd.DataFrame, url_column: str = "url", text_column: str = "text") -> pd.DataFrame:
        if url_column not in df.columns:
            raise ValueError(f"Missing required column: {url_column}")

        extracted_df = df.copy()
        if text_column not in extracted_df.columns:
            extracted_df[text_column] = None

        for idx, url in extracted_df[url_column].items():
            try:
                html = self.fetch_page(url)
                extracted_df.at[idx, text_column] = self.extract_text(html)
                time.sleep(self.delay_seconds)
            except Exception as exc:
                extracted_df.at[idx, text_column] = None
                print(f"Failed at {url}: {exc}")

        return extracted_df


Extractor = WebExtractor
