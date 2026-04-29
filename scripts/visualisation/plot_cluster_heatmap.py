"""Normalised cluster × outlet heatmap."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "cluster_assignments.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "cluster_heatmap.png"

    df = pd.read_csv(input_path)

    ct = pd.crosstab(df["news_outlet"], df["cluster"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        ct_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Proportion"},
    )

    plt.xlabel("Cluster")
    plt.ylabel("News Outlet")
    plt.title("Cluster Distribution by Outlet (Normalised)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
