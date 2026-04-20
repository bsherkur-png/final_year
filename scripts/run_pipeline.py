from pathlib import Path

import pandas as pd

from src.pipeline.news_pipeline import DEFAULT_SOURCE, NewsPipeline



def _pipeline() -> NewsPipeline:
    return NewsPipeline(source=DEFAULT_SOURCE)


def _check_file(path: Path, label: str) -> bool:
    if path.exists():
        return True
    print(f"  {label} not found at {path}")
    print(f"  Run the required earlier stage first.")
    return False


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
  5. Run preprocessing + sentiment + clustering
  6. Run outlet comparison only

  Analysis
  --------
  7. Run statistical tests (Kruskal-Wallis + Mann-Whitney)

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
    if not _check_file(pipeline.ingestion_output_path, "Master CSV"):
        return
    df = pipeline.run_extraction()
    print(f"Extraction complete. {len(df)} rows.")


def run_filtering():
    pipeline = _pipeline()
    if not _check_file(pipeline.extraction_raw_output_path, "Extracted articles"):
        return
    df = pipeline.run_filtering()
    print(f"Filtering complete. {len(df)} articles remain.")


def run_analysis_stages():
    """Run preprocessing, sentiment, and clustering from the filtered CSV."""
    pipeline = _pipeline()
    if not _check_file(pipeline.extraction_output_path, "Filtered articles"):
        return

    filtered_df = pd.read_csv(pipeline.extraction_output_path)
    filtered_df = pipeline._ensure_article_id(filtered_df)
    filtered_df = filtered_df.groupby("news_outlet").filter(lambda g: len(g) >= 6)

    print(f"Processing {len(filtered_df)} articles...")
    articles = pipeline.run_preprocessing(filtered_df)
    print("Preprocessing complete.")

    pipeline.run_raw_sentiment(articles, filtered_df)
    print("Sentiment scoring complete.")

    result = pipeline.run_clustering(articles, filtered_df)
    print(f"Clustering complete. k={result.k}, silhouette={result.silhouette_score:.4f}")


def run_outlet_comparison():
    pipeline = _pipeline()
    if not _check_file(pipeline.raw_sentiment_output_path, "Sentiment scores"):
        return
    pipeline.run_outlet_comparison()
    print("Outlet comparison complete.")


def run_statistical_tests():
    pipeline = _pipeline()
    if not _check_file(pipeline.raw_sentiment_output_path, "Sentiment scores"):
        return

    df = pd.read_csv(pipeline.raw_sentiment_output_path)
    results = run_all_tests(df)
    print(format_results(results))


ACTIONS = {
    "1": run_full_pipeline,
    "2": run_ingestion,
    "3": run_extraction,
    "4": run_filtering,
    "5": run_analysis_stages,
    "6": run_outlet_comparison,
    "7": run_statistical_tests,
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