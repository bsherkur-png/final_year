from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "news_urls"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "combined_news.csv"
EXCLUDED_INPUT_FILENAMES = {
    "combined_news.csv",
    "clustered_news_topics.csv",
    "cluster_topic_summary.csv",
    "topic_12_only.csv",
}


def get_column_name(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def normalize_dates(series):
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


def guess_news_agent(url, file_path):
    if pd.notna(url):
        parsed_url = urlparse(str(url))
        host = parsed_url.netloc.lower().replace("www.", "")
        if host:
            parts = host.split(".")
            if len(parts) >= 2:
                return parts[-2]
            return host
    return file_path.stem


def read_one_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    news_agent_col = get_column_name(df, ["news_agent", "news_source", "source", "media_name", "domain"])
    news_title_col = get_column_name(df, ["news_title", "title", "headline", "article_title"])
    date_col = get_column_name(df, ["date", "publish_date", "published_at", "published", "seendate"])
    url_col = get_column_name(df, ["url", "link", "article_url"])

    rows = pd.DataFrame()
    rows["news agent"] = df[news_agent_col] if news_agent_col else pd.NA
    rows["news title"] = df[news_title_col] if news_title_col else ""
    rows["date"] = normalize_dates(df[date_col]) if date_col else pd.NaT
    rows["url"] = df[url_col] if url_col else ""

    rows["url"] = rows["url"].fillna("").astype(str).str.strip()
    rows["news title"] = rows["news title"].fillna("").astype(str).str.strip()
    rows["news agent"] = rows["news agent"].fillna("")
    rows["news agent"] = rows.apply(
        lambda row: str(row["news agent"]).strip() if str(row["news agent"]).strip() else guess_news_agent(row["url"], file_path),
        axis=1,
    )

    rows = rows[(rows["url"] != "") & (rows["news title"] != "")]
    rows = rows.drop_duplicates(subset=["url"])
    return rows


def gather_csv_articles(folder_path, excluded_filenames=None):
    folder_path = Path(folder_path)
    excluded_names = {name.lower() for name in (excluded_filenames or EXCLUDED_INPUT_FILENAMES)}
    csv_files = [
        file_path
        for file_path in sorted(folder_path.glob("*.csv"))
        if file_path.name.lower() not in excluded_names
    ]

    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")

    all_dataframes = []
    for file_path in csv_files:
        all_dataframes.append(read_one_csv(file_path))

    combined = pd.concat(all_dataframes, ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"])
    return combined


def prepare_master_articles(df):
    df = df.copy()
    df = df.dropna(subset=["date"])
    df = df.sort_values(["date", "news agent", "news title"], ascending=[False, True, True])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["news agent", "news title", "date", "url"]]


def build_master_csv(input_folder=DEFAULT_INPUT_DIR, output_file=DEFAULT_OUTPUT):
    combined = gather_csv_articles(input_folder)
    final_df = prepare_master_articles(combined)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    return final_df


if __name__ == "__main__":
    final_df = build_master_csv()
    print(final_df.head())
    print(f"Saved {len(final_df)} rows to {DEFAULT_OUTPUT}")
