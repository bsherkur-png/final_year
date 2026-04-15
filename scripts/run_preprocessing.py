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
    parser.add_argument("--text-column", default="text", help="Column containing article body text.")
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
    if args.text_column not in df.columns:
        raise ValueError(f"Missing required column: {args.text_column}")

    preprocessor = ArticlePreprocessor.from_spacy_model()
    processed_df = preprocessor.preprocess_article_dataframe(
        df.loc[:, [args.text_column]].rename(columns={args.text_column: "original_body_text"}),
        body_column="original_body_text",
    )
    df[args.processed_column] = processed_df["fully_preprocessed_body_text"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
