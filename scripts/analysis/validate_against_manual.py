"""Validate VADER and zero-shot z-scores against manual annotations.

Reads:
  - data/intermediate/scaled_sentiment_articles.csv
  - data/manual/manual_annotations.csv

Writes:
  - data/intermediate/manual_validation_results.csv

Manual annotations CSV must have columns:
  - article_id (int, matching the pipeline's article_id)
  - manual_label (-1, 0, or +1)
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.pipeline.config import PipelineConfig


def load_and_merge(config: PipelineConfig) -> pd.DataFrame:
    """Load scaled sentiment and manual annotations, merge on article_id."""
    scaled_df = pd.read_csv(config.scaled_sentiment_output)
    manual_df = pd.read_csv(config.manual_annotations_path)

    required = {"article_id", "manual_label"}
    missing = required - set(manual_df.columns)
    if missing:
        raise ValueError(f"Manual annotations missing columns: {missing}")

    merged = scaled_df.merge(manual_df, on="article_id", how="inner")

    if len(merged) == 0:
        raise ValueError(
            "No matching article_ids between scaled sentiment "
            "and manual annotations."
        )

    return merged


def compute_spearman(
    df: pd.DataFrame,
    model_column: str,
    label_column: str = "manual_label",
) -> dict[str, float]:
    """Compute Spearman rank correlation between a model column and labels."""
    paired = df[[model_column, label_column]].dropna()

    if len(paired) < 3:
        raise ValueError(
            f"Need at least 3 paired observations for {model_column}, "
            f"got {len(paired)}."
        )

    rho, p = spearmanr(paired[model_column], paired[label_column])

    return {
        "model": model_column,
        "rho": float(rho),
        "p": float(p),
        "n": len(paired),
    }


def validate(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Run Spearman validation for both models and save results."""
    config = config or PipelineConfig()
    merged = load_and_merge(config)

    results = []
    for col in ("vader_z", "zeroshot_z"):
        if col in merged.columns:
            results.append(compute_spearman(merged, col))

    if not results:
        raise ValueError("No z-score columns found in scaled sentiment CSV.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(config.manual_validation_output, index=False)

    return results_df


def main() -> None:
    results_df = validate()
    print("Manual validation results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved to {PipelineConfig().manual_validation_output}")


if __name__ == "__main__":
    main()
