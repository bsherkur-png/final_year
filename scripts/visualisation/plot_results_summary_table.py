from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.style.use("seaborn-v0_8-whitegrid")


def format_float(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.{decimals}f}"


def run_kruskal(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    comparison_label: str,
) -> dict[str, object]:
    groups = [
        series.dropna()
        for _, series in df.groupby(group_col)[value_col]
    ]
    groups = [series for series in groups if not series.empty]

    k = len(groups)
    n = int(sum(len(series) for series in groups))

    if k < 2:
        h_stat = float("nan")
        p_value = float("nan")
        eta_sq = float("nan")
    else:
        h_stat, p_value = kruskal(*groups)
        eta_sq = (h_stat - k + 1) / (n - k) if n > k else float("nan")

    return {
        "Comparison": comparison_label,
        "Test": "Kruskal-Wallis",
        "Statistic": float(h_stat),
        "p": float(p_value),
        "Effect size": float(eta_sq),
        "n": n,
    }


def run_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    comparison_label: str,
) -> dict[str, object]:
    pair_df = df[[x_col, y_col]].dropna()
    n = int(len(pair_df))

    if n < 2:
        rho = float("nan")
        p_value = float("nan")
    else:
        rho, p_value = spearmanr(pair_df[x_col], pair_df[y_col])

    return {
        "Comparison": comparison_label,
        "Test": "Spearman",
        "Statistic": float(rho),
        "p": float(p_value),
        "Effect size": "—",
        "n": n,
    }


def main() -> None:
    scaled_path = PROJECT_ROOT / "data" / "intermediate" / "scaled_sentiment_articles.csv"
    cluster_path = PROJECT_ROOT / "data" / "intermediate" / "cluster_assignments.csv"
    manual_path = PROJECT_ROOT / "data" / "manual" / "manual_annotations.csv"
    output_path = PROJECT_ROOT / "data" / "figures" / "results_summary_table.png"

    scaled_df = pd.read_csv(scaled_path)
    cluster_df = pd.read_csv(cluster_path)

    df = scaled_df.merge(
        cluster_df[["article_id", "cluster"]],
        on="article_id",
        how="inner",
    )

    if manual_path.exists():
        manual_df = pd.read_csv(manual_path)[["article_id", "manual_label"]]
        df = df.merge(manual_df, on="article_id", how="left")
    else:
        df["manual_label"] = float("nan")

    results = [
        run_kruskal(df, "news_outlet", "vader_z", "Outlet → VADER"),
        run_kruskal(df, "news_outlet", "zeroshot_z", "Outlet → Zero-shot"),
        run_kruskal(df, "cluster", "vader_z", "Frame → VADER"),
        run_kruskal(df, "cluster", "zeroshot_z", "Frame → Zero-shot"),
        run_spearman(df, "vader_z", "zeroshot_z", "VADER ↔ Zero-shot"),
    ]

    if manual_path.exists():
        results.append(run_spearman(df, "vader_z", "manual_label", "VADER ↔ Manual"))
        results.append(run_spearman(df, "zeroshot_z", "manual_label", "Zero-shot ↔ Manual"))

    table_rows: list[list[str]] = []
    significant_rows: list[bool] = []

    for row in results:
        p_value = row["p"]
        is_significant = bool(not pd.isna(p_value) and p_value < 0.05)
        significant_rows.append(is_significant)

        if isinstance(row["Effect size"], str):
            effect_size_str = row["Effect size"]
        else:
            effect_size_str = format_float(float(row["Effect size"]), 2)

        table_rows.append(
            [
                str(row["Comparison"]),
                str(row["Test"]),
                format_float(float(row["Statistic"]), 2),
                format_float(float(row["p"]), 4),
                effect_size_str,
                str(int(row["n"])),
            ]
        )

    columns = ["Comparison", "Test", "Statistic", "p", "Effect size", "n"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.0)

    n_rows = len(table_rows)
    n_cols = len(columns)

    for col_idx in range(n_cols):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#2C3E50")
        cell.set_height(0.08)
        if col_idx == 0:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("center")
        cell.set_text_props(color="white", weight="bold")

    for row_idx in range(1, n_rows + 1):
        data_idx = row_idx - 1
        base_color = "#FFFFFF" if data_idx % 2 == 0 else "#F2F2F2"
        row_color = "#D5F5E3" if significant_rows[data_idx] else base_color
        row_weight = "bold" if significant_rows[data_idx] else "normal"

        for col_idx in range(n_cols):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(row_color)
            cell.set_height(0.08)
            if col_idx == 0:
                cell.get_text().set_ha("left")
            else:
                cell.get_text().set_ha("center")
            cell.set_text_props(weight=row_weight)

    ax.set_title("Summary of Statistical Tests", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
