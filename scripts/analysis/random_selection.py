"""Chunk sampled articles into sentence groups for manual annotation.

Reads:
  - data/intermediate/preprocessed_articles.csv

Writes:
  - data/intermediate/annotation_chunks.csv

Each article is split into chunks of N sentences (default 4) using
spaCy's sentence segmenter. Output CSV columns: article_id,
news_outlet, chunk_number, chunk_text.
"""

import sys
from pathlib import Path

import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def chunk_article(
    text: str,
    nlp: spacy.language.Language,
    sentences_per_chunk: int = 4,
) -> list[str]:
    """Split text into chunks of N sentences."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks


def main(sentences_per_chunk: int = 4) -> None:
    input_path = PROJECT_ROOT / "data" / "intermediate" / "preprocessed_articles.csv"
    output_path = PROJECT_ROOT / "data" / "intermediate" / "annotation_chunks.csv"

    df = pd.read_csv(input_path)

    sampled = (
        df.groupby("news_outlet", group_keys=False)
        .apply(lambda group: group.sample(n=min(3, len(group)), random_state=41))
        .reset_index(drop=True)
    )

    nlp = spacy.load("en_core_web_sm")

    rows = []
    for _, article in sampled.iterrows():
        chunks = chunk_article(
            str(article["cleaned_text"]),
            nlp,
            sentences_per_chunk,
        )
        for i, chunk_text in enumerate(chunks, start=1):
            rows.append({
                "article_id": article["article_id"],
                "news_outlet": article["news_outlet"],
                "chunk_number": i,
                "chunk_text": chunk_text,
            })

    output_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(
        f"Saved {len(output_df)} chunks from "
        f"{sampled['article_id'].nunique()} articles to {output_path}."
    )


if __name__ == "__main__":
    main()
