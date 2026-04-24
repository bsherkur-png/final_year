"""Non-parametric statistical tests for outlet sentiment comparison."""

import pandas as pd
import scikit_posthocs as sp
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


def dunns_posthoc(
    df: pd.DataFrame,
    score_column: str = "composite_score",
    group_column: str = "news_outlet",
    p_adjust: str = "bonferroni",
) -> pd.DataFrame:
    """Run Dunn's post-hoc pairwise test across outlet groups."""
    if score_column not in df.columns:
        raise ValueError(f"Missing required score column: {score_column}")
    if group_column not in df.columns:
        raise ValueError(f"Missing required group column: {group_column}")

    return sp.posthoc_dunn(
        df,
        val_col=score_column,
        group_col=group_column,
        p_adjust=p_adjust,
    )


def effect_sizes(
    h_stat: float,
    n: int,
    k: int,
) -> dict[str, float | str]:
    """Compute epsilon-squared effect size and interpretive label."""
    if n <= k:
        raise ValueError("n must be greater than k.")

    epsilon_sq = (h_stat - k + 1) / (n - k)
    epsilon_sq = max(0.0, min(1.0, epsilon_sq))

    if epsilon_sq < 0.06:
        label = "small"
    elif epsilon_sq < 0.14:
        label = "medium"
    else:
        label = "large"

    return {"epsilon_squared": epsilon_sq, "label": label}
