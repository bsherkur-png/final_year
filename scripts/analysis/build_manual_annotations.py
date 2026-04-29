"""Build article-level manual annotations from chunk-level Label Studio CSV."""

import argparse
from pathlib import Path

import pandas as pd

from src.comparison.aggregate_annotations import aggregate_chunk_labels
from src.pipeline.config import PROJECT_ROOT, PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate chunk-level sentiment labels to article-level scores."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "label_studio_export.csv",
        help="Path to Label Studio CSV export.",
    )
    args = parser.parse_args()

    chunks_df = pd.read_csv(args.input_csv)
    manual_df = aggregate_chunk_labels(chunks_df)

    config = PipelineConfig()
    config.manual_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    manual_df.to_csv(config.manual_annotations_path, index=False)

    print(f"Wrote {len(manual_df)} article labels to {config.manual_annotations_path}")


if __name__ == "__main__":
    main()
