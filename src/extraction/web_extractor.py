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
DEFAULT_BROWSER_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def _extract_text_fallback(html: str) -> str:
    import re

    article_match = re.search(r"<article\b[^>]*>(.*?)</article>", html, flags=re.IGNORECASE | re.DOTALL)
    source_html = article_match.group(1) if article_match else html
    matches = re.findall(r"<p[^>]*>(.*?)</p>", source_html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = [re.sub(r"<[^>]+>", " ", match) for match in matches]
    cleaned = [re.sub(r"\s+", " ", text).strip() for text in cleaned]
    return " ".join(text for text in cleaned if text)


def extract_article_text(html: str) -> str:
    if BeautifulSoup is None:
        return _extract_text_fallback(html)

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
        timeout_seconds: int = 20,
    ):
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_BROWSER_HEADERS)

    def fetch_page(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        response.encoding = response.encoding or response.apparent_encoding or "utf-8"
        return response.text

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

        date_cols = ["date"] if "date" in extracted_df.columns else []
        extracted_df = extracted_df.loc[
            :,
            ["article_id", "news_outlet", "title", "date_link"] + date_cols + ["body"],
        ]
        extracted_df = extracted_df.set_index("article_id", drop=False)
        extracted_df["fetch_status"] = "ok"
        extracted_df["fetch_error"] = ""

        for article_id, url in extracted_df[url_column].items():
            try:
                html = self.fetch_page(url)
                extracted_df.at[article_id, "body"] = extract_article_text(html)
                time.sleep(self.delay_seconds)
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                extracted_df.at[article_id, "fetch_status"] = f"http_{status_code}"
                extracted_df.at[article_id, "fetch_error"] = str(exc)
                extracted_df.at[article_id, "body"] = ""
            except Exception as exc:
                extracted_df.at[article_id, "fetch_status"] = f"error:{type(exc).__name__}"
                extracted_df.at[article_id, "fetch_error"] = str(exc)
                extracted_df.at[article_id, "body"] = ""

        final_cols = [
            "article_id",
            "news_outlet",
            "title",
            "date_link",
            "date",
            "body",
            "fetch_status",
            "fetch_error",
        ]
        final_cols = [c for c in final_cols if c in extracted_df.columns]
        return extracted_df.reset_index(drop=True).loc[:, final_cols]

