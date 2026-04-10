from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "src" / "ingestion" / "data" / "clustered_news_topics.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    if "cluster" not in df.columns:
        raise ValueError(f"'cluster' column not found in {INPUT_FILE}")

    cluster_counts = (
        df.groupby("cluster")
        .size()
        .reset_index(name="article_count")
        .sort_values("cluster")
    )

    print(cluster_counts.to_string(index=False))


if __name__ == "__main__":
    main()
