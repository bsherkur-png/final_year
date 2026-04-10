from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "src" / "ingestion" / "data" / "clustered_news_topics.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "intermediate" / "cluster_validation_sample.csv"

SAMPLE_SIZE_PER_TOPIC = 30
RANDOM_STATE = 41


def main():
    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print(f"No rows found in {INPUT_FILE}.")
        return

    if "topic_label" not in df.columns:
        raise ValueError(f"'topic_label' column not found in {INPUT_FILE}")

    df["topic_label"] = df["topic_label"].fillna("").astype(str).str.strip()
    df = df.loc[~df["topic_label"].str.lower().str.startswith("topic")].reset_index(drop=True)

    if df.empty:
        print(f"No labelled topic rows found in {INPUT_FILE}.")
        return

    sample_df = (
        df.groupby("topic_label", group_keys=False)
        .apply(lambda group: group.sample(n=min(SAMPLE_SIZE_PER_TOPIC, len(group)), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    sample_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(sample_df)} sampled rows to: {OUTPUT_FILE}")
    print(sample_df["topic_label"].value_counts().sort_index().to_string())
    print(sample_df.to_string(index=False))


if __name__ == "__main__":
    main()
