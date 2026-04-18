import argparse
from pathlib import Path

from src.pipeline.news_pipeline import DEFAULT_INGESTION_OUTPUT, NewsPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the news pipeline.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INGESTION_OUTPUT,
        help="Master CSV path.",
    )
    args = parser.parse_args()
    final_df = NewsPipeline(ingestion_output=args.input).run()
    print(f"Pipeline complete. {len(final_df)} rows scored.")


if __name__ == "__main__":
    main()
