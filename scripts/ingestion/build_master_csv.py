import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.preprocessing.filters import filter_opinion_pieces


def build_master_csv(input_file: Path, output_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df = df.rename(columns={"link": "date_link", "source": "news_outlet"})
    df = df.drop(columns=["page", "snippet"])
    df = df.drop_duplicates(subset=["date_link"])
    df = filter_opinion_pieces(df)
    df["article_id"] = range(1, len(df) + 1)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build master CSV from raw metadata.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "raw" / "review_sheet.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "intermediate" / "master_articles.csv",
        help="Path to output CSV.",
    )
    args = parser.parse_args()
    result_df = build_master_csv(args.input, args.output)
    print(f"Wrote {len(result_df)} rows to {args.output}")
