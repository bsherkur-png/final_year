"""Zero-shot sentiment classification using DeBERTa-v3 NLI model."""
#

from __future__ import annotations

import pandas as pd

from src.preprocessing.spacy_processor import ProcessedArticle

MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

CANDIDATE_LABELS = [
    "positive sentiment",
    "negative sentiment",
    "neutral sentiment",
]

HYPOTHESIS_TEMPLATE = "The sentiment of this news article is {}."


class ZeroshotScorer:
    """Score articles using zero-shot NLI-based sentiment classification."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from transformers import pipeline

        self._classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            truncation=True,
        )

    def score_chunks(self, chunks: list[str]) -> float:
        """Mean of P(pos)-P(neg) across non-empty chunks in one batched call.

        Returns 0.0 if the list is empty or contains only empty strings.
        """
        filtered = [c for c in chunks if c.strip()]
        if not filtered:
            return 0.0

        results = self._classifier(
            filtered,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        scores = []
        for result in results:
            label_scores = dict(zip(result["labels"], result["scores"]))
            scores.append(
                label_scores.get("positive sentiment", 0.0)
                - label_scores.get("negative sentiment", 0.0)
            )
        return float(sum(scores) / len(scores))

    def score_all(
        self, articles: list[ProcessedArticle],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score all articles, returning (article_df, chunk_df).

        article_df: indexed by article_id, column "zeroshot" (float).
        chunk_df: columns article_id, chunk_index, chunk_text, zeroshot_score.
        """
        batch = [
            (a.article_id, i, c)
            for a in articles
            for i, c in enumerate(a.chunks)
            if c.strip()
        ]

        all_ids = [a.article_id for a in articles]

        if not batch:
            article_df = pd.DataFrame(
                {"zeroshot": 0.0},
                index=pd.Index(all_ids, name="article_id"),
            )
            chunk_df = pd.DataFrame(
                columns=["article_id", "chunk_index", "chunk_text", "zeroshot_score"],
            )
            return article_df, chunk_df

        article_ids, chunk_indices, chunk_texts = zip(*batch)

        results = self._classifier(
            list(chunk_texts),
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=8,
        )

        chunk_scores = [
            dict(zip(r["labels"], r["scores"])).get("positive sentiment", 0.0)
            - dict(zip(r["labels"], r["scores"])).get("negative sentiment", 0.0)
            for r in results
        ]

        chunk_df = pd.DataFrame({
            "article_id": article_ids,
            "chunk_index": chunk_indices,
            "chunk_text": chunk_texts,
            "zeroshot_score": chunk_scores,
        })

        article_df = (
            chunk_df.groupby("article_id")["zeroshot_score"]
            .mean()
            .rename("zeroshot")
            .to_frame()
        )
        article_df.index.name = "article_id"

        # Articles with no valid chunks get 0.0
        missing = set(all_ids) - set(article_df.index)
        if missing:
            missing_df = pd.DataFrame(
                {"zeroshot": 0.0},
                index=pd.Index(list(missing), name="article_id"),
            )
            article_df = pd.concat([article_df, missing_df])

        return article_df, chunk_df
