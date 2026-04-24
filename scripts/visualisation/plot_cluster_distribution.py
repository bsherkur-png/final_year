from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "cluster_assignments.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "cluster_distribution.png"

    df = pd.read_csv(input_path)

    ct = pd.crosstab(df["news_outlet"], df["cluster"])
    ct = ct.div(ct.sum(axis=1), axis=0)
    ct = ct.sort_index()

    ct.plot(
        kind="bar",
        stacked=True,
        colormap="Set2",
        edgecolor="white",
        linewidth=0.5,
    )

    plt.xlabel("News Outlet")
    plt.ylabel("Proportion of Articles")
    plt.title("Cluster Distribution by Outlet")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Saved cluster_distribution.png")


if __name__ == "__main__":
    main()
