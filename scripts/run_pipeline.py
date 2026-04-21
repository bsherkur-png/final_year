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
  5. Run preprocessing + sentiment
  6. Run clustering only
  7. Run outlet comparison only

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

    raw_df = pipeline.run_raw_sentiment(articles, filtered_df)
    pipeline.run_scaled_sentiment(raw_df)
    print("Sentiment scoring complete.")


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


ACTIONS = {
    "1": run_full_pipeline,
    "2": run_ingestion,
    "3": run_extraction,
    "4": run_filtering,
    "5": run_preprocessing_and_sentiment,
    "6": run_clustering,
    "7": run_outlet_comparison,
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
