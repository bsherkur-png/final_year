from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "news_meta_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "combined_news.csv"


def get_column_name(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def normalize_dates(series):
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    parsed = parsed.dt.tz_localize(None)
    return parsed.dt.strftime("%d/%m/%Y")


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
    article_id_col = get_column_name(df, ["id", "article_id"])
    title_col = get_column_name(df, ["news_title", "title", "headline", "article_title"])
    date_col = get_column_name(df, ["date", "publish_date", "published_at", "published", "seendate"])
    media_name_col = get_column_name(df, ["news_agent", "news_source", "source", "media_name", "domain"])
    url_col = get_column_name(df, ["url", "link", "article_url"])

    missing_columns = []
    if article_id_col is None:
        missing_columns.append("id")
    if title_col is None:
        missing_columns.append("news title")
    if date_col is None:
        missing_columns.append("date")
    if media_name_col is None:
        missing_columns.append("news agent")

    if missing_columns:
        raise ValueError(f"Missing required source columns: {missing_columns}")

    rows = pd.DataFrame()
    rows["article_id"] = df[article_id_col]
    rows["title"] = df[title_col]
    rows["publish_date"] = normalize_dates(df[date_col])
    rows["url"] = df[url_col] if url_col else ""
    rows["media_name"] = df[media_name_col]

    return rows[["article_id", "title", "publish_date", "url", "media_name"]]


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
