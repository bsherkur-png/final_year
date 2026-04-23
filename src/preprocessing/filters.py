import re

import pandas as pd


_SHAMIMA_PATTERN = re.compile(r"\bshamima\s+begum\b", re.IGNORECASE)
EVENT_RULES = [
    {"event_id": "E1_discovery", "date_start": "2019-02-14", "date_end": "2019-02-17", "keywords": ["found", "camp", "come home", "come back", "wants to return", "runaway", "fled", "schoolgirl", "pregnant", "birth"]},
    {"event_id": "E2_citizenship", "date_start": "2019-02-19", "date_end": "2019-02-22", "keywords": ["citizenship", "stripped", "revoke", "passport", "stateless", "bangladesh", "javid"]},
    {"event_id": "E3_reaction", "date_start": "2019-02-17", "date_end": "2019-02-22", "keywords": ["corbyn", "right to return", "legal aid", "prosecution", "deradicalise", "manchester", "justified", "threat"]},
    {"event_id": "E4_baby_family", "date_start": "2019-02-17", "date_end": "2019-02-19", "keywords": ["birth", "baby", "boy", "son", "jerah", "family", "husband", "lawyer", "father"]},
]


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


def tag_events(
    df: pd.DataFrame,
    date_column: str = "date",
    text_columns: tuple[str, ...] = ("title", "body"),
) -> pd.DataFrame:
    """Tag each article with an event_id using date windows and keyword hits."""
    missing = [c for c in [date_column, *text_columns] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    dates = pd.to_datetime(df[date_column], dayfirst=True, format="mixed", errors="coerce")
    combined = df.apply(
        lambda row: " ".join("" if pd.isna(row[c]) else str(row[c]) for c in text_columns).lower(),
        axis=1,
    )
    rules = [
        (pd.Timestamp(rule["date_start"]), pd.Timestamp(rule["date_end"]), rule["event_id"], rule["keywords"])
        for rule in EVENT_RULES
    ]
    event_ids: list[str | None] = []
    for date, text in zip(dates, combined):
        if pd.isna(date):
            event_ids.append(None)
            continue
        article_day = date.normalize()
        best_id, best_hits = None, 0
        for start, end, event_id, keywords in rules:
            if not (start <= article_day <= end):
                continue
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits >= 2 and hits > best_hits:
                best_id, best_hits = event_id, hits
        event_ids.append(best_id)
    tagged = df.copy()
    tagged["event_id"] = event_ids
    return tagged
