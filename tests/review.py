from pathlib import Path
import pandas as pd

INPUT_FILE = Path(__file__).with_name("review_sheet.csv")
OUTPUT_FILE = Path(__file__).with_name("matched_articles.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    for col in ["manual_keep", "same_event", "straight_news"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    matched = df[
        (df["manual_keep"] == "yes") &
        (df["same_event"] == "yes") &
        (df["straight_news"] == "yes")
    ]

    matched.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(matched)} rows to {OUTPUT_FILE}")
    print(matched[["event_id", "target_outlet", "title", "link"]].head(20))


if __name__ == "__main__":
    main()
