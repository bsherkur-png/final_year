import hashlib
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "news_meta_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "master_articles.csv"
OUTPUT_COLUMNS = ["article_id", "news_outlet", "title", "date_link"]
ARTICLE_ID_LENGTH = 16


def get_column_name(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def clean_display_text(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_for_key(value):
    return clean_display_text(value).lower()


def build_dedupe_key(news_outlet, title, date_link):
    return "|".join(
        [
            normalize_for_key(news_outlet),
            normalize_for_key(title),
            normalize_for_key(date_link),
        ]
    )


def build_article_id(dedupe_key):
    return hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:ARTICLE_ID_LENGTH]


def load_source_articles(input_file):
    df = pd.read_csv(input_file)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def prepare_master_articles(df):
    title_col = get_column_name(df, ["news_title", "title", "headline", "article_title"])
    news_outlet_col = get_column_name(
        df,
        ["news_outlet", "news_agent", "news_source", "source", "media_name", "domain"],
    )
    date_link_col = get_column_name(
        df,
        ["date_link", "link", "url", "article_url", "date", "publish_date", "published_at", "published", "seendate"],
    )

    missing_columns = []
    if title_col is None:
        missing_columns.append("title")
    if news_outlet_col is None:
        missing_columns.append("news_outlet")
    if date_link_col is None:
        missing_columns.append("date_link")

    if missing_columns:
        raise ValueError(f"Missing required source columns: {missing_columns}")

    rows = pd.DataFrame()
    rows["news_outlet"] = df[news_outlet_col].map(clean_display_text)
    rows["title"] = df[title_col].map(clean_display_text)
    rows["date_link"] = df[date_link_col].map(clean_display_text)

    rows["_dedupe_key"] = rows.apply(
        lambda row: build_dedupe_key(row["news_outlet"], row["title"], row["date_link"]),
        axis=1,
    )
    rows["article_id"] = rows["_dedupe_key"].map(build_article_id)
    rows = rows.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])

    return rows[OUTPUT_COLUMNS]


def build_master_csv(input_file=DEFAULT_INPUT, output_file=DEFAULT_OUTPUT):
    source_df = load_source_articles(input_file)
    final_df = prepare_master_articles(source_df)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    return final_df


if __name__ == "__main__":
    final_df = build_master_csv()
    print(final_df.head())
    print(f"Saved {len(final_df)} rows to {DEFAULT_OUTPUT}")
