from scipy.stats import zscore

import pandas as pd


def scale_sentiment(
    df: pd.DataFrame,
    polarity_columns: list[str] = ("vader_score", "nrc_score"),
) -> pd.DataFrame:
    """Z-score polarity columns and compute their mean as a composite.

    Adds columns: ``<col>_z`` for each input column, plus ``composite_score``.
    """
    scaled_df = df.copy()

    z_columns = []
    for col in polarity_columns:
        z_col = f"{col.removesuffix('_score')}_z"
        scaled_df[z_col] = zscore(scaled_df[col])
        z_columns.append(z_col)

    scaled_df["composite_score"] = scaled_df[z_columns].mean(axis=1)
    return scaled_df
