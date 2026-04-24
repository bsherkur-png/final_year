from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "scaled_sentiment_articles.csv"
    output_dir = PROJECT_ROOT / "data" / "figures"
    output_path = output_dir / "boxplot_composite_by_outlet.png"

    sentiment_df = pd.read_csv(input_path)

    outlets = sorted(sentiment_df["news_outlet"].dropna().unique().tolist())
    grouped_scores = [
        sentiment_df.loc[sentiment_df["news_outlet"] == outlet, "composite_score"]
        .dropna()
        .tolist()
        for outlet in outlets
    ]

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(
        grouped_scores,
        tick_labels=outlets,
        patch_artist=True,
    )

    color_count = max(len(outlets) - 1, 1)
    box_colors = [plt.cm.Set2(i / color_count) for i in range(len(outlets))]
    for box, color in zip(bp["boxes"], box_colors):
        box.set_facecolor(color)

    plt.xlabel("News Outlet")
    plt.ylabel("Composite Sentiment (z-score)")
    plt.title("Composite Sentiment Score Distribution by Outlet")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print("Saved boxplot_composite_by_outlet.png")


if __name__ == "__main__":
    main()
