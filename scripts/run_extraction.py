import argparse
from pathlib import Path

import pandas as pd

from src.extraction.scraper import Extractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "intermediate" / "news_meta_data_sample_1pct.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "sample_extracted.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract article text for URLs in a CSV file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    extractor = Extractor()
    extracted_df = extractor.extract(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extracted_df.to_csv(args.output, index=False)
    print(f"Saved {len(extracted_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
