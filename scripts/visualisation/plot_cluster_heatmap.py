"""Normalised cluster × outlet heatmap."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")

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


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "cluster_assignments.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "cluster_heatmap.png"

    df = pd.read_csv(input_path)

    ct = pd.crosstab(df["news_outlet"], df["cluster"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, (ax_heatmap, ax_key) = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        gridspec_kw={"width_ratios": [2.3, 1.2]},
    )

    sns.heatmap(
        ct_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Proportion"},
        annot_kws={"size": 10},
        ax=ax_heatmap,
    )

    ax_heatmap.set_xlabel("Cluster")
    ax_heatmap.set_ylabel("News Outlet")
    ax_heatmap.set_title("Cluster Distribution by Outlet (Normalised)")

    cluster_values = sorted(ct_norm.columns.tolist())
    frame_lines = [
        f"{cluster}: {FRAME_LABELS.get(cluster, 'Unknown frame')}"
        for cluster in cluster_values
    ]

    ax_key.axis("off")
    ax_key.set_title("Frame Key", fontsize=13, loc="left", pad=8)
    ax_key.text(
        0.0,
        0.98,
        "\n".join(frame_lines),
        transform=ax_key.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.5,
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
