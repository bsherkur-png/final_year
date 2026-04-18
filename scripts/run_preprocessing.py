import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.article_preprocessor import ArticlePreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "intermediate" / "articles_with_bodies.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "preprocessed_articles.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess article body text in a CSV file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--text-column", default="body", help="Column containing article body text.")
    parser.add_argument(
        "--processed-column",
        default="processed_text",
        help="Column name for the preprocessed article body.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    df = pd.read_csv(input_path)
    preprocessor = ArticlePreprocessor.from_spacy_model()
    preprocessed_df = preprocessor.preprocess_dataframe(
        df,
        body_column=args.text_column,
    )
    if args.processed_column != "fully_preprocessed_body_text":
        preprocessed_df[args.processed_column] = preprocessed_df["fully_preprocessed_body_text"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_df.to_csv(output_path, index=False)
    print(f"Saved {len(preprocessed_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
