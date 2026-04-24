from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "scaled_sentiment_articles.csv"
    manual_path = PROJECT_ROOT / "data" / "manual" / "manual_annotations.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "scatter_vader_vs_zeroshot.png"

    df = pd.read_csv(input_path)

    if manual_path.exists():
        manual_df = pd.read_csv(manual_path)
        df = df.merge(manual_df[["article_id", "manual_label"]], on="article_id", how="left")
    else:
        df["manual_label"] = float("nan")

    pair_df = df[["vader_z", "zeroshot_z", "manual_label"]].dropna(
        subset=["vader_z", "zeroshot_z"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    label_colours = {-1: "#d62728", 0: "#7f7f7f", 1: "#2ca02c"}
    label_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    unlabelled = pair_df[pair_df["manual_label"].isna()]
    ax.scatter(
        unlabelled["vader_z"],
        unlabelled["zeroshot_z"],
        alpha=0.3,
        s=25,
        c="#cccccc",
        edgecolors="none",
        label="Unlabelled",
    )

    for label_val, colour in label_colours.items():
        subset = pair_df[pair_df["manual_label"] == label_val]
        if subset.empty:
            continue
        ax.scatter(
            subset["vader_z"],
            subset["zeroshot_z"],
            alpha=0.9,
            s=60,
            c=colour,
            edgecolors="white",
            linewidths=0.5,
            label=label_names[label_val],
            zorder=3,
        )

    rho, p = spearmanr(pair_df["vader_z"], pair_df["zeroshot_z"])
    ax.text(
        0.05, 0.95,
        f"ρ = {rho:.3f}\np = {p:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    lims = [
        min(pair_df["vader_z"].min(), pair_df["zeroshot_z"].min()),
        max(pair_df["vader_z"].max(), pair_df["zeroshot_z"].max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3)

    ax.set_xlabel("VADER z-score")
    ax.set_ylabel("Zero-shot z-score")
    ax.set_title("Cross-Method Agreement: VADER vs Zero-Shot")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
