from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv
from src.bias.feature_builder import FeatureBuilder
from src.bias.topic_clusterer import TopicClusterer, ClusteringResult
from src.comparison.outlet_comparator import summarize_outlets
from src.comparison.statistical_tests import (
    kruskal_wallis,
    dunns_posthoc,
    effect_sizes,
    wilcoxon_signed_rank,
)
from src.extraction.web_extractor import WebExtractor
from src.pipeline.config import PipelineConfig
from src.preprocessing.filters import filter_shamima_mentions, filter_short_articles
from src.preprocessing.spacy_processor import SpacyProcessor, ProcessedArticle
from src.sentiment.lexicons.sentiment_analyzer import LexiconScorer
from src.sentiment.zeroshot_scorer import ZeroshotScorer
from src.sentiment.scaling import scale_sentiment


def _write_csv(df: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def run_ingestion(config: PipelineConfig) -> pd.DataFrame:
    ingested_df = build_master_csv(
        input_file=config.source_path,
        output_file=config.ingestion_output,
    )
    return ingested_df


def run_extraction(config: PipelineConfig) -> pd.DataFrame:
    master_df = pd.read_csv(config.ingestion_output)

    extracted_df = WebExtractor().extract(master_df)

    _write_csv(extracted_df, config.extraction_raw_output)
    return extracted_df


def run_filtering(config: PipelineConfig) -> pd.DataFrame:
    extracted_df = pd.read_csv(config.extraction_raw_output)
    filtered_df = filter_shamima_mentions(
        extracted_df,
        min_mentions=2,
        text_columns=("title", "body"),
    )
    filtered_df = filter_short_articles(filtered_df, min_words=250)

    _write_csv(filtered_df, config.extraction_output)
    return filtered_df


def run_preprocessing(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> list[ProcessedArticle]:
    articles = SpacyProcessor().process_dataframe(df)

    # CSV checkpoint — write a flat version for debugging/inspection
    rows = []
    for a in articles:
        rows.append(
            {
                "article_id": a.article_id,
                "cleaned_text": a.cleaned_text,
                "lemmas": " ".join(a.lemmas),
            }
        )
    checkpoint_df = pd.DataFrame(rows)
    # Merge metadata from the input df so the checkpoint is self-contained
    meta_cols = [c for c in ("news_outlet", "title", "date_link") if c in df.columns]
    if meta_cols:
        checkpoint_df = checkpoint_df.merge(
            df[["article_id"] + meta_cols], on="article_id", how="left"
        )
    _write_csv(checkpoint_df, config.preprocess_output)

    return articles


def run_lexicon_sentiment(
    articles: list[ProcessedArticle],
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    scorer = LexiconScorer()
    lexicon_df = scorer.score_all(articles)

    scored_df = df.set_index("article_id").join(lexicon_df).reset_index()
    scored_df = scored_df.rename(columns={"vader": "vader_score"})
    final_df = scored_df.loc[:, ["article_id", "news_outlet", "vader_score"]]

    _write_csv(final_df, config.raw_sentiment_output)
    return final_df


def run_chunk_diagnostics(
    articles: list[ProcessedArticle],
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    scorer = LexiconScorer()
    chunk_rows = [
        {
            "article_id": int(a.article_id),
            "chunk_index": i,
            "chunk_text": c,
            "vader_score": scorer.score_vader(c),
        }
        for a in articles
        for i, c in enumerate(a.chunks)
    ]
    chunk_df = pd.DataFrame(
        chunk_rows,
        columns=["article_id", "chunk_index", "chunk_text", "vader_score"],
    )
    chunk_df = chunk_df.merge(df[["article_id", "news_outlet"]], on="article_id", how="left")
    output_df = chunk_df[
        ["article_id", "news_outlet", "chunk_index", "chunk_text", "vader_score"]
    ]
    _write_csv(output_df, config.chunk_sentiment_output)
    return output_df


def run_zeroshot_sentiment(
    articles: list[ProcessedArticle],
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    zeroshot = ZeroshotScorer()
    zeroshot_df = zeroshot.score_all(articles)

    if not config.raw_sentiment_output.exists():
        raise ValueError(
            "Raw sentiment CSV not found. Run lexicon sentiment first."
        )

    raw_df = pd.read_csv(config.raw_sentiment_output)
    raw_df = raw_df.set_index("article_id").join(
        zeroshot_df.rename(columns={"zeroshot": "zeroshot_score"})
    ).reset_index()

    _write_csv(raw_df, config.raw_sentiment_output)
    return raw_df


def run_scaled_sentiment(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Z-score polarity columns and write scaled CSV."""
    polarity_cols = [c for c in ("vader_score", "zeroshot_score") if c in df.columns]
    scaled_df = scale_sentiment(df, polarity_columns=polarity_cols)

    checkpoint_columns = [
        c for c in scaled_df.columns
        if c not in ("title", "date_link", "vader_score", "zeroshot_score")
    ]
    _write_csv(scaled_df[checkpoint_columns], config.scaled_sentiment_output)
    return scaled_df


def run_outlet_comparison(config: PipelineConfig) -> pd.DataFrame:
    sentiment_df = pd.read_csv(config.scaled_sentiment_output)

    summary_df = summarize_outlets(
        sentiment_df,
        polarity_column="zeroshot_z",
    )

    _write_csv(summary_df, config.outlet_comparison_output)
    return summary_df


def run_statistical_tests(config: PipelineConfig) -> dict:
    """Run Kruskal-Wallis on zeroshot_z, Wilcoxon on vader_z vs zeroshot_z.

    Dunn's post-hoc is only run if Kruskal-Wallis is significant
    (p < 0.05).
    """
    sentiment_df = pd.read_csv(config.scaled_sentiment_output)

    kw = kruskal_wallis(sentiment_df)
    es = effect_sizes(kw["H"], kw["n"], kw["k"])

    kw_row = {**kw, **es}
    kw_df = pd.DataFrame([kw_row])
    _write_csv(kw_df, config.kruskal_wallis_output)

    if kw["p"] < 0.05:
        dunn_df = dunns_posthoc(sentiment_df)
        _write_csv(dunn_df, config.dunns_posthoc_output)

    wc = wilcoxon_signed_rank(sentiment_df)
    wc_df = pd.DataFrame([wc])
    _write_csv(wc_df, config.wilcoxon_output)

    return {**kw_row, "wilcoxon": wc}


def run_clustering(
    articles: list[ProcessedArticle],
    df: pd.DataFrame,
    config: PipelineConfig,
) -> ClusteringResult:
    if "news_outlet" not in df.columns:
        raise ValueError("Missing required column: news_outlet")

    builder = FeatureBuilder()
    feature_matrix = builder.build(articles)
    feature_names = builder.feature_names

    article_ids = [a.article_id for a in articles]
    id_to_outlet = dict(zip(df["article_id"], df["news_outlet"]))
    outlets = [id_to_outlet[article_id] for article_id in article_ids]
    result = TopicClusterer().run(
        feature_matrix, article_ids, outlets, feature_names
    )
    _write_csv(result.assignments, config.cluster_assignments_output)
    top_terms_df = pd.DataFrame(
        [
            {"cluster": cluster, "terms": ", ".join(terms)}
            for cluster, terms in result.top_terms.items()
        ]
    )
    _write_csv(top_terms_df, config.cluster_top_terms_output)

    return result


def run_full_pipeline(config: PipelineConfig) -> pd.DataFrame:
    run_ingestion(config)
    run_extraction(config)
    filtered_df = run_filtering(config)
    filtered_df = filtered_df.groupby("news_outlet").filter(lambda g: len(g) >= 6)

    # spaCy processes once — result shared by sentiment + clustering
    articles = run_preprocessing(filtered_df, config)

    run_lexicon_sentiment(articles, filtered_df, config)
    run_chunk_diagnostics(articles, filtered_df, config)
    scaled_df = run_scaled_sentiment(
        pd.read_csv(config.raw_sentiment_output), config
    )
    run_clustering(articles, filtered_df, config)
    run_outlet_comparison(config)
    return scaled_df
