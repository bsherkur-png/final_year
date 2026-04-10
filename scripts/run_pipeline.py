import argparse
from pathlib import Path

from src.pipeline.news_pipeline import run_news_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the article extraction pipeline for a CSV source.")
    parser.add_argument("input", type=Path, help="Input CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_news_pipeline(args.input)
    print(f"Processed {len(result)} rows from {args.input}")


if __name__ == "__main__":
    main()
