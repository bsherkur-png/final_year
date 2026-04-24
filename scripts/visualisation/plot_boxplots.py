from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "scaled_sentiment_articles.csv"
    output_dir = PROJECT_ROOT / "data" / "figures"
    output_path = output_dir / "boxplot_vader_vs_zeroshot_by_outlet.png"

    df = pd.read_csv(input_path)

    outlets = sorted(df["news_outlet"].dropna().unique().tolist())
    n_outlets = len(outlets)

    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.35
    positions_vader = [i - width / 2 for i in range(n_outlets)]
    positions_zs = [i + width / 2 for i in range(n_outlets)]

    vader_data = [
        df.loc[df["news_outlet"] == o, "vader_z"].dropna().tolist()
        for o in outlets
    ]
    zs_data = [
        df.loc[df["news_outlet"] == o, "zeroshot_z"].dropna().tolist()
        for o in outlets
    ]

    bp_vader = ax.boxplot(
        vader_data,
        positions=positions_vader,
        widths=width * 0.8,
        patch_artist=True,
        tick_labels=[""] * n_outlets,
    )
    bp_zs = ax.boxplot(
        zs_data,
        positions=positions_zs,
        widths=width * 0.8,
        patch_artist=True,
        tick_labels=[""] * n_outlets,
    )

    for box in bp_vader["boxes"]:
        box.set_facecolor("#7fbfff")
    for box in bp_zs["boxes"]:
        box.set_facecolor("#ff9f7f")

    ax.set_xticks(range(n_outlets))
    ax.set_xticklabels(outlets, rotation=45, ha="right")
    ax.set_xlabel("News Outlet")
    ax.set_ylabel("Sentiment (z-score)")
    ax.set_title("VADER vs Zero-Shot Sentiment by Outlet")
    ax.legend(
        [bp_vader["boxes"][0], bp_zs["boxes"][0]],
        ["VADER z", "Zero-shot z"],
        loc="upper right",
    )

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
