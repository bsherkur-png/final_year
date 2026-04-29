# AGENTS.md

## Project overview

A Python NLP pipeline analysing UK news coverage of Shamima Begum across five
outlets (BBC, Guardian, Daily Mail, The Mirror, The Independent). The pipeline
extracts article text from the web, cleans boilerplate contamination,
preprocesses with spaCy, scores sentiment with two methods (VADER lexicon and
DeBERTa-v3 zero-shot), and includes a k-means clustering module for
exploratory framing analysis.

The core research question is a CS model evaluation: which sentiment method
better matches human judgement on this corpus? 15 manually annotated articles
serve as ground truth.

The dataset is small (~77 articles after filtering) — this is a research
prototype, not a production system.

## Architecture

```
scripts/
  ingestion/build_master_csv.py        — raw CSV → cleaned master CSV
  run_pipeline.py                      — CLI entry point (11 menu options)
  analysis/validate_against_manual.py  — Spearman validation vs manual labels
  visualisation/                       — matplotlib/seaborn chart scripts

src/
  extraction/web_extractor.py          — fetches article HTML, extracts body
  preprocessing/
    text_cleaner.py                    — deterministic boilerplate removal
    spacy_processor.py                 — SpacyProcessor + ProcessedArticle
    filters.py                         — Shamima mentions, word count, opinion filter
  sentiment/
    lexicons/sentiment_analyzer.py     — LexiconScorer (VADER + NRC internally)
    zeroshot_scorer.py                 — ZeroshotScorer (DeBERTa-v3 zero-shot NLI)
    scaling.py                         — z-standardisation (independent per method)
  bias/
    feature_builder.py                 — TF-IDF + linguistic features
    topic_clusterer.py                 — k-means clustering
  comparison/
    outlet_comparator.py               — outlet-level summary statistics
    statistical_tests.py               — Kruskal-Wallis, Dunn's, Wilcoxon, effect sizes
  pipeline/
    config.py                          — PipelineConfig (all output paths)
    news_pipeline.py                   — orchestrates the full pipeline

data/
  raw/                                 — source CSV with article metadata
  manual/                              — manual_annotations.csv (human labels)
  intermediate/                        — pipeline stage outputs (CSV checkpoints)
  figures/                             — generated PNGs
```

## Key design decisions

### Boilerplate removal via deterministic pattern matching

53% of articles contain non-editorial text (comment prompts, bylines,
privacy notices, image captions) that survived HTML extraction.
Contamination is systematic and unevenly distributed across outlets.
`text_cleaner.strip_boilerplate()` removes these using regex patterns
identified by manual inspection of all 77 articles. This is applied
once during preprocessing — `ProcessedArticle.cleaned_text` stores
the result as a plain field.

### Two sentiment methods, not three

NRC Emotion Lexicon was dropped from the analysis scope. It is still
computed internally by LexiconScorer but its columns are excluded from
pipeline CSV output. Only VADER compound score and DeBERTa-v3 zero-shot
P(pos)-P(neg) are carried forward.

### Shared cleaned text for both models

Both VADER and zero-shot scorers receive identical input via
`article.cleaned_text`. This ensures any divergence in scores reflects
model behaviour, not preprocessing differences.

### No composite score

Each method is z-standardised independently (`vader_z`, `zeroshot_z`).
No composite mean is computed. The two z-scored columns are compared
directly via Wilcoxon signed-rank and validated independently against
manual labels via Spearman correlation.

### Manual annotation as ground truth

15 articles (3 per outlet) are hand-labelled as -1 (negative), 0
(neutral), or +1 (positive). This CSV lives at
`data/manual/manual_annotations.csv` and is created by the human.

### Statistical tests

| Test | Purpose | Input |
|------|---------|-------|
| Kruskal-Wallis | Do outlets differ in tone? | zeroshot_z by outlet |
| Dunn's post-hoc | Which outlet pairs differ? | Only if K-W p < 0.05 |
| Wilcoxon signed-rank | Do the two models systematically differ? | vader_z vs zeroshot_z |
| Spearman × 2 | Which model matches human judgement? | manual vs vader_z, manual vs zeroshot_z |

### ProcessedArticle is the shared data object

See `src/preprocessing/spacy_processor.py`. A `@dataclass` holding:
- `article_id: str`
- `raw_text: str` — original extracted text, unmodified
- `cleaned_text: str` — boilerplate-stripped, whitespace-normalised
- `doc: spacy.tokens.Doc` — spaCy Doc built from raw_text

Computed `@property` values:
- `lemmas -> list[str]` — lowercased lemmas, stops/punct/non-alpha removed
- `nrc_tokens -> list[str]` — alias for `lemmas`

Created once per article by `SpacyProcessor`. Passed to every downstream
stage. No stage calls `nlp()` again.

### PipelineConfig centralises paths

All output paths live in `src/pipeline/config.py`. No path strings are
hardcoded in `news_pipeline.py`.

## Coding conventions

- Python 3.11+. Type hints on all public method signatures.
- PEP 8. No logging framework — `print()` sparingly for progress only.
- `ValueError` for invalid input. Messages must name the specific problem.
- `unittest.TestCase` for tests. No pytest fixtures or parametrize.
- No abstract base classes unless the prompt explicitly requests them.

## Dependencies

spaCy (en_core_web_sm), pandas, scikit-learn, NLTK (VADER), nrclex,
scipy, numpy, matplotlib, seaborn, transformers (DeBERTa-v3),
scikit-posthocs.
