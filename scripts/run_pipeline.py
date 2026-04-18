import argparse
from pathlib import Path

from src.pipeline.news_pipeline import DEFAULT_INGESTION_OUTPUT, NewsPipeline


STAGE_ORDER = ("extraction", "filtering", "preprocessing", "sentiment", "outlet_comparison")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the news pipeline orchestrator.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INGESTION_OUTPUT,
        help="Existing master CSV path.",
    )
    parser.add_argument(
        "--start-stage",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[0],
        help="Stage to start from.",
    )
    parser.add_argument(
        "--stop-stage",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[-1],
        help="Stage to stop at.",
    )
    args = parser.parse_args()
    if STAGE_ORDER.index(args.start_stage) > STAGE_ORDER.index(args.stop_stage):
        parser.error("--start-stage must be before or equal to --stop-stage.")
    return args


def main():
    args = parse_args()
    pipeline = NewsPipeline(ingestion_output=args.input)

    stage_runners = {
        "extraction": (pipeline.run_extraction, pipeline.extraction_raw_output_path),
        "filtering": (pipeline.run_filtering, pipeline.extraction_output_path),
        "preprocessing": (pipeline.run_preprocessing, pipeline.preprocess_output_path),
        "sentiment": (pipeline.run_raw_sentiment, pipeline.raw_sentiment_output_path),
        "outlet_comparison": (
            pipeline.run_outlet_comparison,
            pipeline.outlet_comparison_output_path,
        ),
    }

    start_idx = STAGE_ORDER.index(args.start_stage)
    stop_idx = STAGE_ORDER.index(args.stop_stage)
    stages_to_run = STAGE_ORDER[start_idx : stop_idx + 1]

    final_df = None
    final_output_path = None
    for stage in stages_to_run:
        runner, output_path = stage_runners[stage]
        final_df = runner()
        final_output_path = output_path

    print(f"Final output path: {final_output_path}")
    print(f"Row count: {len(final_df)}")


if __name__ == "__main__":
    main()
