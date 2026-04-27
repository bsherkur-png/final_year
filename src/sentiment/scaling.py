from scipy.stats import zscore

import pandas as pd


def scale_sentiment(
    df: pd.DataFrame,
    polarity_columns: tuple[str, ...] = ("vader_score",),
) -> pd.DataFrame:
    """Z-score each polarity column independently.

    Adds ``<method>_z`` columns (e.g. ``vader_z``, ``zeroshot_z``).
    No composite score is computed.
    """
    scaled_df = df.copy()

    for col in polarity_columns:
        z_col = f"{col.removesuffix('_score')}_z"
        scaled_df[z_col] = zscore(scaled_df[col])

    return scaled_df
