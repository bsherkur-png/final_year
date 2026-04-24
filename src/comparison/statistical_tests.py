"""Non-parametric statistical tests for outlet sentiment comparison."""

import pandas as pd
import scipy.stats as stats


def kruskal_wallis(
    df: pd.DataFrame,
    score_column: str = "composite_score",
    group_column: str = "news_outlet",
) -> dict[str, float | int]:
    """Run a Kruskal-Wallis H-test across outlet groups."""
    if score_column not in df.columns:
        raise ValueError(f"Missing required score column: {score_column}")
    if group_column not in df.columns:
        raise ValueError(f"Missing required group column: {group_column}")

    groups: list[list[float]] = []
    for _, group in df.groupby(group_column)[score_column]:
        valid_scores = group.dropna()
        if len(valid_scores) >= 2:
            groups.append(valid_scores.tolist())

    if len(groups) < 2:
        raise ValueError("At least 2 groups with >=2 observations are required.")

    result = stats.kruskal(*groups)

    return {
        "H": float(result.statistic),
        "p": float(result.pvalue),
        "k": len(groups),
        "n": sum(len(group) for group in groups),
    }
