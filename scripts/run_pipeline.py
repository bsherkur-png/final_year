import argparse
from pathlib import Path

from src.pipeline.news_pipeline import DEFAULT_SOURCE, NewsPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the news pipeline.")
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=DEFAULT_SOURCE,
        help="Path to the raw news metadata CSV.",
    )
    args = parser.parse_args()
    final_df = NewsPipeline(source=args.source).run()
    print(f"Pipeline complete. {len(final_df)} rows scored.")


if __name__ == "__main__":
    main()
