from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineConfig:
    """All filesystem paths used by the pipeline."""

    source_path: Path = PROJECT_ROOT / "data" / "raw" / "news_meta_data.csv"
    output_dir: Path = PROJECT_ROOT / "data" / "intermediate"

    @property
    def manual_annotations_path(self) -> Path:
        return PROJECT_ROOT / "data" / "manual" / "manual_annotations.csv"

    @property
    def ingestion_output(self) -> Path:
        return self.output_dir / "master_articles.csv"

    @property
    def extraction_raw_output(self) -> Path:
        return self.output_dir / "articles_with_bodies_raw.csv"

    @property
    def extraction_output(self) -> Path:
        return self.output_dir / "articles_with_bodies.csv"

    @property
    def preprocess_output(self) -> Path:
        return self.output_dir / "preprocessed_articles.csv"

    @property
    def raw_sentiment_output(self) -> Path:
        return self.output_dir / "raw_sentiment_articles.csv"

    @property
    def chunk_sentiment_output(self) -> Path:
        return self.output_dir / "chunk_sentiment.csv"

    @property
    def chunk_zeroshot_output(self) -> Path:
        return self.output_dir / "chunk_zeroshot_sentiment.csv"

    @property
    def scaled_sentiment_output(self) -> Path:
        return self.output_dir / "scaled_sentiment_articles.csv"

    @property
    def outlet_comparison_output(self) -> Path:
        return self.output_dir / "outlet_comparison_summary.csv"

    @property
    def kruskal_wallis_output(self) -> Path:
        return self.output_dir / "kruskal_wallis_result.csv"

    @property
    def dunns_posthoc_output(self) -> Path:
        return self.output_dir / "dunns_posthoc_pvalues.csv"

    @property
    def wilcoxon_output(self) -> Path:
        return self.output_dir / "wilcoxon_result.csv"

    @property
    def manual_validation_output(self) -> Path:
        return self.output_dir / "manual_validation_results.csv"

    @property
    def cluster_assignments_output(self) -> Path:
        return self.output_dir / "cluster_assignments.csv"

    @property
    def cluster_top_terms_output(self) -> Path:
        return self.output_dir / "cluster_top_terms.csv"
