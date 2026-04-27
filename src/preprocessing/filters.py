import re

import pandas as pd


_SHAMIMA_PATTERN = re.compile(r"\bshamima\s+begum\b", re.IGNORECASE)


def filter_shamima_mentions(
    df: pd.DataFrame,
    min_mentions: int = 2,
    text_columns: tuple[str, ...] = ("title", "body"),
) -> pd.DataFrame:
    """Keep articles mentioning Shamima Begum at least min_mentions times."""
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


def filter_short_articles(
    df: pd.DataFrame,
    min_words: int = 250,
    body_column: str = "body",
) -> pd.DataFrame:
    """Remove articles with fewer than min_words words in the body."""
    if body_column not in df.columns:
        raise ValueError(f"Missing required column: {body_column}")

    word_counts = df[body_column].fillna("").str.split().str.len()
    return df.loc[word_counts >= min_words].reset_index(drop=True)


_OPINION_URL_MARKERS = [
    "/commentisfree/",
    "/voices/",
    "/opinion/",
    "/comment/",
]

_OPINION_TITLE_MARKERS = [
    "opinion:",
    "observer view",
    "observer letters",
    "| letters",
]


def filter_opinion_pieces(
    df: pd.DataFrame,
    url_column: str = "date_link",
    title_column: str = "title",
) -> pd.DataFrame:
    """Remove opinion pieces, editorials, and comment columns.

    Identifies opinion content by checking URL path segments
    (e.g. /commentisfree/, /voices/) and title markers (e.g. 'Opinion:').
    """
    missing = [c for c in (url_column, title_column) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}"
        )

    urls = df[url_column].fillna("").str.lower()
    titles = df[title_column].fillna("").str.lower()

    is_opinion = pd.Series(False, index=df.index)
    for marker in _OPINION_URL_MARKERS:
        is_opinion = is_opinion | urls.str.contains(marker, regex=False)
    for marker in _OPINION_TITLE_MARKERS:
        is_opinion = is_opinion | titles.str.contains(marker, regex=False)

    return df.loc[~is_opinion].reset_index(drop=True)
