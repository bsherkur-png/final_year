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

    def score_article(self, article: ProcessedArticle) -> float:
        """Return a continuous sentiment score for one article.

        Score = P(positive) - P(negative), giving a value in [-1, 1].
        """
        return self.score_chunks(article.chunks)

    def score_all(self, articles: list[ProcessedArticle]) -> pd.DataFrame:
        """Score all articles, returning a DataFrame indexed by article_id.

        Columns: zeroshot (float, the P(pos) - P(neg) score).
        """
        article_scores: dict[str, list[float]] = {article.article_id: [] for article in articles}
        chunk_texts: list[str] = []
        chunk_article_ids: list[str] = []
        for article in articles:
            filtered_chunks = [chunk for chunk in article.chunks if chunk.strip()]
            chunk_texts.extend(filtered_chunks)
            chunk_article_ids.extend([article.article_id] * len(filtered_chunks))

        if chunk_texts:
            results = self._classifier(
                chunk_texts,
                candidate_labels=CANDIDATE_LABELS,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                multi_label=False,
                batch_size=8,
            )
            for article_id, result in zip(chunk_article_ids, results):
                label_scores = dict(zip(result["labels"], result["scores"]))
                article_scores[article_id].append(
                    label_scores.get("positive sentiment", 0.0)
                    - label_scores.get("negative sentiment", 0.0)
                )

        rows = {
            article_id: {"zeroshot": float(sum(scores) / len(scores)) if scores else 0.0}
            for article_id, scores in article_scores.items()
        }
        scores_df = pd.DataFrame.from_dict(rows, orient="index")
        scores_df.index.name = "article_id"
        return scores_df
