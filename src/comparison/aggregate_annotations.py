"""Aggregate chunk-level sentiment annotations to article-level scores."""

import pandas as pd


LABEL_MAP: dict[str, int] = {"Negative": -1, "Neutral": 0, "Positive": 1}


def aggregate_chunk_labels(
    chunks: pd.DataFrame,
    label_column: str = "sentiment",
    article_column: str = "article_id",
) -> pd.DataFrame:
    """Map chunk labels to numeric values and average them per article."""
    missing_columns = [
        column
        for column in (label_column, article_column)
        if column not in chunks.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing_columns}"
        )

    mapped = chunks[[article_column, label_column]].copy()
    mapped["manual_label"] = mapped[label_column].map(LABEL_MAP)

    valid = mapped.dropna(subset=["manual_label"])
    if valid.empty:
        raise ValueError(
            "No rows remain after label mapping; expected labels are "
            f"{list(LABEL_MAP)}."
        )

    aggregated = (
        valid.groupby(article_column, as_index=False)["manual_label"]
        .mean()
        .rename(columns={article_column: "article_id"})
        .sort_values("article_id")
        .reset_index(drop=True)
    )
    aggregated["article_id"] = aggregated["article_id"].astype(int)
    aggregated["manual_label"] = aggregated["manual_label"].astype(float)

    return aggregated[["article_id", "manual_label"]]
