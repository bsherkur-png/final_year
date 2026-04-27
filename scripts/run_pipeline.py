from pathlib import Path

import pandas as pd

from src.pipeline.config import PipelineConfig
from src.pipeline.news_pipeline import (
    run_ingestion,
    run_extraction,
    run_filtering,
    run_preprocessing,
    run_lexicon_sentiment,
    run_chunk_diagnostics,
    run_zeroshot_sentiment,
    run_scaled_sentiment,
    run_clustering,
    run_outlet_comparison,
    run_statistical_tests,
)
from src.preprocessing.filters import filter_small_outlets


def _check_file(path: Path, label: str) -> bool:
    if path.exists():
        return True
    print(f"  {label} not found at {path}")
    print(f"  Run the required earlier stage first.")
    return False


def _load_filtered_df(config: PipelineConfig) -> pd.DataFrame | None:
    """Read the filtered CSV and apply the minimum-articles-per-outlet filter."""
    if not _check_file(config.extraction_output, "Filtered articles"):
        return None
    filtered_df = pd.read_csv(config.extraction_output)
    filtered_df = filter_small_outlets(filtered_df, min_articles=6, group_column="news_outlet")
    return filtered_df


MENU = """
========================================
  Shamima Begum — News Analysis Pipeline
========================================

  Pipeline stages
  ---------------
  1. (reserved)
  2. Run ingestion only
  3. Run extraction only
  4. Run filtering only
  5. Run preprocessing + lexicon sentiment (VADER)
  6. Run zero-shot sentiment (slow, run after 5)
  7. Run z-standardisation
  8. Run clustering only
  9. Run outlet comparison only
  10. Run statistical tests (Kruskal-Wallis + Wilcoxon)
  11. Build manual annotations from Label Studio export
  12. Run Spearman validation against manual labels

  0. Exit
"""


def menu_run_ingestion():
    config = PipelineConfig()
    df = run_ingestion(config)
    print(f"Ingestion complete. {len(df)} rows.")


def menu_run_extraction():
    config = PipelineConfig()
    if not _check_file(config.ingestion_output, "Master CSV"):
        return
    df = run_extraction(config)
    print(f"Extraction complete. {len(df)} rows.")


def menu_run_filtering():
    config = PipelineConfig()
    if not _check_file(config.extraction_raw_output, "Extracted articles"):
        return
    df = run_filtering(config)
    print(f"Filtering complete. {len(df)} articles remain.")


def menu_run_preprocessing_and_sentiment():
    """Run preprocessing and sentiment scoring from the filtered CSV."""
    config = PipelineConfig()
    filtered_df = _load_filtered_df(config)
    if filtered_df is None:
        return

    print(f"Processing {len(filtered_df)} articles...")
    articles = run_preprocessing(filtered_df, config)
    print("Preprocessing complete.")

    run_lexicon_sentiment(articles, filtered_df, config)
    run_chunk_diagnostics(articles, filtered_df, config)
    print("Preprocessing and VADER scoring complete.")


def menu_run_zeroshot():
    """Run zero-shot sentiment scoring on already-preprocessed articles."""
    config = PipelineConfig()
    filtered_df = _load_filtered_df(config)
    if filtered_df is None:
        return

    print(f"Running zero-shot classification on {len(filtered_df)} articles...")
    print("  (This may take 15-25 minutes on CPU)")
    articles = run_preprocessing(filtered_df, config)
    run_zeroshot_sentiment(articles, filtered_df, config)
    print("Zero-shot scoring complete.")


def menu_run_scaling():
    """Run z-standardisation on raw sentiment scores."""
    config = PipelineConfig()
    if not _check_file(config.raw_sentiment_output, "Raw sentiment CSV"):
        return
    raw_df = pd.read_csv(config.raw_sentiment_output)
    run_scaled_sentiment(raw_df, config)
    print("Z-standardisation complete.")


def menu_run_clustering():
    """Run clustering from the filtered CSV."""
    config = PipelineConfig()
    filtered_df = _load_filtered_df(config)
    if filtered_df is None:
        return

    print(f"Processing {len(filtered_df)} articles for clustering...")
    articles = run_preprocessing(filtered_df, config)

    result = run_clustering(articles, filtered_df, config)
    print(f"Clustering complete. k={result.k}, silhouette={result.silhouette_score:.4f}")


def menu_run_outlet_comparison():
    config = PipelineConfig()
    if not _check_file(config.scaled_sentiment_output, "Scaled sentiment scores"):
        return
    run_outlet_comparison(config)
    print("Outlet comparison complete.")


def menu_run_statistical_tests():
    config = PipelineConfig()
    if not _check_file(config.scaled_sentiment_output, "Scaled sentiment scores"):
        return
    result = run_statistical_tests(config)
    print(f"Kruskal-Wallis H={result['H']:.4f}, p={result['p']:.6f}")
    print(f"Effect size (epsilon²): {result['epsilon_squared']:.4f} ({result['label']})")
    if result["p"] < 0.05:
        print("Dunn's post-hoc pairwise p-values saved.")
    else:
        print("Dunn's post-hoc skipped (K-W not significant).")
    wc = result["wilcoxon"]
    print(f"Wilcoxon W={wc['W']:.4f}, p={wc['p']:.6f}, r={wc['r_effect_size']:.4f}")


def menu_build_annotations():
    """Build article-level manual labels from chunk-level Label Studio export."""
    config = PipelineConfig()
    from src.comparison.aggregate_annotations import aggregate_chunk_labels
    from src.pipeline.config import PROJECT_ROOT

    label_studio_path = PROJECT_ROOT / "data" / "raw" / "label_studio_export.csv"
    if not _check_file(label_studio_path, "Label Studio export"):
        print(f"  Place your Label Studio CSV export at {label_studio_path}.")
        return

    chunks_df = pd.read_csv(label_studio_path)
    manual_df = aggregate_chunk_labels(chunks_df)

    config.manual_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    manual_df.to_csv(config.manual_annotations_path, index=False)
    print(f"Wrote {len(manual_df)} article labels to {config.manual_annotations_path}")


def menu_spearman_validation():
    """Run Spearman validation against manual annotations."""
    config = PipelineConfig()
    from scripts.analysis.validate_against_manual import validate

    if not _check_file(config.manual_annotations_path, "Manual annotations CSV"):
        print("  Create data/manual/manual_annotations.csv first.")
        print("  Columns: article_id, manual_label (-1, 0, or +1)")
        return
    results_df = validate(config)
    print("Spearman validation results:")
    print(results_df.to_string(index=False))


ACTIONS = {
    "2": menu_run_ingestion,
    "3": menu_run_extraction,
    "4": menu_run_filtering,
    "5": menu_run_preprocessing_and_sentiment,
    "6": menu_run_zeroshot,
    "7": menu_run_scaling,
    "8": menu_run_clustering,
    "9": menu_run_outlet_comparison,
    "10": menu_run_statistical_tests,
    "11": menu_build_annotations,
    "12": menu_spearman_validation,
}


def main():
    while True:
        print(MENU)
        choice = input("Select option: ").strip()

        if choice == "0":
            print("Exiting.")
            break

        action = ACTIONS.get(choice)
        if action is None:
            print(f"Invalid option: {choice}")
            continue

        try:
            action()
        except Exception:
            import traceback
            traceback.print_exc()

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
