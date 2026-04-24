from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "raw_sentiment_articles.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "cross_method_scatter.png"

    df = pd.read_csv(input_path)

    pairs: list[tuple[str, str]] = [("vader_score", "nrc_score")]
    if "zeroshot_score" in df.columns:
        pairs.extend(
            [
                ("vader_score", "zeroshot_score"),
                ("nrc_score", "zeroshot_score"),
            ]
        )

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (x_col, y_col) in zip(axes, pairs):
        pair_df = df[[x_col, y_col]].dropna()
        x = pair_df[x_col]
        y = pair_df[y_col]

        ax.scatter(x, y, alpha=0.5, s=30, edgecolors="none")

        rho, p = spearmanr(x, y)
        ax.text(
            0.05,
            0.95,
            f"ρ = {rho:.3f}\np = {p:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        x_label = x_col.replace("_score", "").capitalize()
        y_label = y_col.replace("_score", "").capitalize()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

    fig.suptitle("Cross-Method Sentiment Agreement", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Saved cross_method_scatter.png")


if __name__ == "__main__":
    main()
