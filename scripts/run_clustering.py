import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.topic_clusterer.service import TopicFilterService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/intermediate/master_articles.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/intermediate/clustered_articles.csv")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "data/intermediate/cluster_summary.csv",
    )
    args = parser.parse_args()

    titles_df = pd.read_csv(args.input)
    result = TopicFilterService().run(titles_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    result.clustered_titles.to_csv(args.output, index=False)
    result.summary.to_csv(args.summary_output, index=False)


if __name__ == "__main__":
    main()
