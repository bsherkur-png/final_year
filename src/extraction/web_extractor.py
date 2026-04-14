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

    def extract(self, df: pd.DataFrame, url_column: str = "url") -> pd.DataFrame:
        if url_column not in df.columns:
            raise ValueError(f"Missing required column: {url_column}")

        extracted_df = df.copy()

        # Append required output columns
        if "original_body_text" not in extracted_df.columns:
            extracted_df["original_body_text"] = None
        if "extraction_status" not in extracted_df.columns:
            extracted_df["extraction_status"] = None
        if "extraction_error" not in extracted_df.columns:
            extracted_df["extraction_error"] = None

        for idx, url in extracted_df[url_column].items():
            try:
                html = self.fetch_page(url)
                extracted_df.at[idx, "original_body_text"] = self.extract_text(html)
                extracted_df.at[idx, "extraction_status"] = "success"
                extracted_df.at[idx, "extraction_error"] = None
                time.sleep(self.delay_seconds)
            except Exception as exc:
                extracted_df.at[idx, "original_body_text"] = None
                extracted_df.at[idx, "extraction_status"] = "failed"
                extracted_df.at[idx, "extraction_error"] = str(exc)
                print(f"Failed at {url}: {exc}")

        return extracted_df


Extractor = WebExtractor
