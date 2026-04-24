from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "dunns_posthoc_pvalues.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "dunns_heatmap.png"

    pvalues_df = pd.read_csv(input_path, index_col=0)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        data=pvalues_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "p-value"},
    )

    plt.title("Dunn's Post-Hoc Pairwise p-values (Bonferroni-corrected)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print("Saved dunns_heatmap.png")


if __name__ == "__main__":
    main()
