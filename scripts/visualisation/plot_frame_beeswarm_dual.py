from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAME_LABELS = {
    0: "Lawyer / media",
    1: "Legal / citizenship",
    2: "Family / hope",
    3: "Jihadi bride",
    4: "Statelessness",
    5: "Security / terror",
    6: "Dutch husband",
    7: "Extremism / crime",
}

plt.style.use("seaborn-v0_8-whitegrid")


def kruskal_by_frame(df: pd.DataFrame, score_col: str, frame_order: list[str]) -> tuple[float, float]:
    groups = [
        df.loc[df["frame"] == frame, score_col].dropna().tolist()
        for frame in frame_order
    ]
    groups = [group for group in groups if group]
    if len(groups) < 2:
        return float("nan"), float("nan")
    h_stat, p_value = kruskal(*groups)
    return float(h_stat), float(p_value)


def main() -> None:
    sentiment_path = PROJECT_ROOT / "data" / "intermediate" / "scaled_sentiment_articles.csv"
    cluster_path = PROJECT_ROOT / "data" / "intermediate" / "cluster_assignments.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "beeswarm_dual_vader_vs_zeroshot_by_frame.png"

    sentiment_df = pd.read_csv(sentiment_path)
    cluster_df = pd.read_csv(cluster_path)

    merged_df = sentiment_df.merge(
        cluster_df[["article_id", "cluster"]],
        on="article_id",
        how="inner",
    )
    merged_df["frame"] = merged_df["cluster"].map(FRAME_LABELS)

    missing_clusters = sorted(
        merged_df.loc[merged_df["frame"].isna(), "cluster"].dropna().unique().tolist()
    )
    if missing_clusters:
        raise ValueError(f"Unmapped cluster values found: {missing_clusters}")

    frame_order = (
        merged_df.groupby("frame", as_index=False)["zeroshot_z"]
        .mean()
        .sort_values("zeroshot_z", ascending=True)["frame"]
        .tolist()
    )

    vader_h, vader_p = kruskal_by_frame(merged_df, "vader_z", frame_order)
    zeroshot_h, zeroshot_p = kruskal_by_frame(merged_df, "zeroshot_z", frame_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.swarmplot(
        data=merged_df,
        x="frame",
        y="vader_z",
        order=frame_order,
        color="#7fbfff",
        size=40**0.5,
        alpha=0.8,
        ax=ax1,
    )
    sns.swarmplot(
        data=merged_df,
        x="frame",
        y="zeroshot_z",
        order=frame_order,
        color="#ff9f7f",
        size=40**0.5,
        alpha=0.8,
        ax=ax2,
    )

    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Topic frame")
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")

    ax1.set_title("VADER (lexicon)")
    ax1.set_ylabel("Sentiment (z-score)")
    ax1.text(
        0.05,
        0.95,
        f"H = {vader_h:.2f}\np = {vader_p:.4f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2.set_title("Zero-shot (DeBERTa)")
    ax2.set_ylabel("")
    ax2.text(
        0.05,
        0.95,
        f"H = {zeroshot_h:.2f}\np = {zeroshot_p:.4f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
