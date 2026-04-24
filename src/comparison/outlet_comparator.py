import pandas as pd


SCORE_COLUMNS = ("vader_score", "vader_z", "zeroshot_score", "zeroshot_z")


def summarize_outlets(df: pd.DataFrame, polarity_column: str = "vader_score") -> pd.DataFrame:
    if "news_outlet" not in df.columns:
        raise ValueError('Missing required column: "news_outlet"')
    if polarity_column not in df.columns:
        raise ValueError(f'Missing required column: "{polarity_column}"')

    present = [c for c in SCORE_COLUMNS if c in df.columns]
    if not present:
        return df[["news_outlet"]].drop_duplicates().reset_index(drop=True)

    summary = df.groupby("news_outlet", dropna=False).agg(
        {c: ["mean", "std", "count"] for c in present}
    ).reset_index()
    summary.columns = [
        "_".join(p for p in parts if p) if isinstance(parts, tuple) else parts
        for parts in summary.columns
    ]
    return summary
