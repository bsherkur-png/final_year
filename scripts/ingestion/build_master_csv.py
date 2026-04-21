import argparse
import hashlib
from pathlib import Path

import pandas as pd


def build_master_csv(input_file: Path, output_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df = df.rename(columns={"link": "date_link", "source": "news_outlet"})
    df = df.drop(columns=["page", "snippet"])
    df["article_id"] = df["date_link"].apply(
        lambda url: hashlib.sha256(str(url).encode()).hexdigest()[:8]
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build master CSV from raw metadata.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "raw" / "news_meta_data.csv",
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
