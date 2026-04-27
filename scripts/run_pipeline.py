from pathlib import Path

import pandas as pd

from src.pipeline.news_pipeline import NewsPipeline

def _pipeline() -> NewsPipeline:
    return NewsPipeline()


def _check_file(path: Path, label: str) -> bool:
    if path.exists():
        return True
    print(f"  {label} not found at {path}")
    print(f"  Run the required earlier stage first.")
    return False


def _load_filtered_df(pipeline: NewsPipeline) -> pd.DataFrame | None:
    """Read the filtered CSV and apply the minimum-articles-per-outlet filter."""
    if not _check_file(pipeline.config.extraction_output, "Filtered articles"):
        return None
    filtered_df = pd.read_csv(pipeline.config.extraction_output)
    filtered_df = filtered_df.groupby("news_outlet").filter(lambda g: len(g) >= 6)
    return filtered_df


MENU = """
========================================
  Shamima Begum — News Analysis Pipeline
========================================

  Pipeline stages
  ---------------
  1. Run full pipeline (ingestion → comparison)
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


def run_full_pipeline():
    pipeline = _pipeline()
    pipeline.run()
    print("Full pipeline complete.")


def run_ingestion():
    pipeline = _pipeline()
    df = pipeline.run_ingestion()
    print(f"Ingestion complete. {len(df)} rows.")


def run_extraction():
    pipeline = _pipeline()
    if not _check_file(pipeline.config.ingestion_output, "Master CSV"):
        return
    df = pipeline.run_extraction()
    print(f"Extraction complete. {len(df)} rows.")


def run_filtering():
    pipeline = _pipeline()
    if not _check_file(pipeline.config.extraction_raw_output, "Extracted articles"):
        return
    df = pipeline.run_filtering()
    print(f"Filtering complete. {len(df)} articles remain.")


def run_preprocessing_and_sentiment():
    """Run preprocessing and sentiment scoring from the filtered CSV."""
    pipeline = _pipeline()
    filtered_df = _load_filtered_df(pipeline)
    if filtered_df is None:
        return

    print(f"Processing {len(filtered_df)} articles...")
    articles = pipeline.run_preprocessing(filtered_df)
    print("Preprocessing complete.")

    pipeline.run_raw_sentiment(articles, filtered_df)
    print("Preprocessing and VADER scoring complete.")


def run_zeroshot():
    """Run zero-shot sentiment scoring on already-preprocessed articles."""
    pipeline = _pipeline()
    filtered_df = _load_filtered_df(pipeline)
    if filtered_df is None:
        return
    if not _check_file(pipeline.config.raw_sentiment_output, "Raw sentiment CSV"):
        print("  Run option 5 (preprocessing + sentiment) first.")
        return

    print(f"Running zero-shot classification on {len(filtered_df)} articles...")
    print("  (This may take 15-25 minutes on CPU)")
    articles = pipeline.run_preprocessing(filtered_df)
    pipeline.run_zeroshot_sentiment(articles, filtered_df)
    print("Zero-shot scoring complete.")


def run_scaling():
    """Run z-standardisation on raw sentiment scores."""
    pipeline = _pipeline()
    if not _check_file(pipeline.config.raw_sentiment_output, "Raw sentiment CSV"):
        return
    raw_df = pd.read_csv(pipeline.config.raw_sentiment_output)
    pipeline.run_scaled_sentiment(raw_df)
    print("Z-standardisation complete.")


def run_clustering():
    """Run clustering from the filtered CSV."""
    pipeline = _pipeline()
    filtered_df = _load_filtered_df(pipeline)
    if filtered_df is None:
        return

    print(f"Processing {len(filtered_df)} articles for clustering...")
    articles = pipeline.run_preprocessing(filtered_df)

    result = pipeline.run_clustering(articles, filtered_df)
    print(f"Clustering complete. k={result.k}, silhouette={result.silhouette_score:.4f}")


def run_outlet_comparison():
    pipeline = _pipeline()
    if not _check_file(pipeline.config.scaled_sentiment_output, "Scaled sentiment scores"):
        return
    pipeline.run_outlet_comparison()
    print("Outlet comparison complete.")


def run_statistical_tests():
    pipeline = _pipeline()
    if not _check_file(pipeline.config.scaled_sentiment_output, "Scaled sentiment scores"):
        return
    result = pipeline.run_statistical_tests()
    print(f"Kruskal-Wallis H={result['H']:.4f}, p={result['p']:.6f}")
    print(f"Effect size (epsilon²): {result['epsilon_squared']:.4f} ({result['label']})")
    if result["p"] < 0.05:
        print("Dunn's post-hoc pairwise p-values saved.")
    else:
        print("Dunn's post-hoc skipped (K-W not significant).")
    wc = result["wilcoxon"]
    print(f"Wilcoxon W={wc['W']:.4f}, p={wc['p']:.6f}, r={wc['r_effect_size']:.4f}")


def run_build_manual_annotations():
    """Build article-level manual labels from chunk-level Label Studio export."""
    from src.comparison.aggregate_annotations import aggregate_chunk_labels
    from src.pipeline.config import PROJECT_ROOT, PipelineConfig

    label_studio_path = PROJECT_ROOT / "data" / "raw" / "label_studio_export.csv"
    if not _check_file(label_studio_path, "Label Studio export"):
        print(f"  Place your Label Studio CSV export at {label_studio_path}.")
        return

    chunks_df = pd.read_csv(label_studio_path)
    manual_df = aggregate_chunk_labels(chunks_df)

    config = PipelineConfig()
    config.manual_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    manual_df.to_csv(config.manual_annotations_path, index=False)
    print(f"Wrote {len(manual_df)} article labels to {config.manual_annotations_path}")


def run_spearman_validation():
    """Run Spearman validation against manual annotations."""
    from scripts.analysis.validate_against_manual import validate

    from src.pipeline.config import PipelineConfig

    config = PipelineConfig()
    if not _check_file(config.manual_annotations_path, "Manual annotations CSV"):
        print("  Create data/manual/manual_annotations.csv first.")
        print("  Columns: article_id, manual_label (-1, 0, or +1)")
        return
    results_df = validate(config)
    print("Spearman validation results:")
    print(results_df.to_string(index=False))


ACTIONS = {
    "1": run_full_pipeline,
    "2": run_ingestion,
    "3": run_extraction,
    "4": run_filtering,
    "5": run_preprocessing_and_sentiment,
    "6": run_zeroshot,
    "7": run_scaling,
    "8": run_clustering,
    "9": run_outlet_comparison,
    "10": run_statistical_tests,
    "11": run_build_manual_annotations,
    "12": run_spearman_validation,
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
        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
