import re

import pandas as pd


_SHAMIMA_PATTERN = re.compile(r"\bshamima\s+begum\b", re.IGNORECASE)


def filter_shamima_mentions(
    df: pd.DataFrame,
    min_mentions: int = 2,
    text_columns: tuple[str, ...] = ("title", "body"),
) -> pd.DataFrame:
    #Keep only rows where Shamima Begum is mentioned at least min_mentions times
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame must contain columns {missing}. Available: {list(df.columns)}"
        )

    def count(row):
        combined = " ".join(
            "" if pd.isna(row[c]) else str(row[c]) for c in text_columns
        )
        return len(_SHAMIMA_PATTERN.findall(combined))

    counts = df.apply(count, axis=1)
    return df.loc[counts >= min_mentions].reset_index(drop=True)
