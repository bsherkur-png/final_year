# AGENTS.md

## Project overview

A Python NLP pipeline analysing UK news coverage of Shamima Begum across five
outlets (BBC, Guardian, Daily Mail, The Sun, The Independent). The pipeline
extracts article text from the web, preprocesses it with spaCy, scores
sentiment with three lexicon-based tools (VADER, NRC, SentiWordNet), and
includes a k-means clustering module that groups articles by TF-IDF and
linguistic features to identify framing patterns. The dataset is small
(88 articles) — this is a research prototype, not a production system.

## Architecture

```
scripts/
  ingestion/build_master_csv.py    — raw CSV → cleaned master CSV (assigns article_id)
  run_pipeline.py                  — CLI entry point

src/
  extraction/web_extractor.py      — fetches article HTML, extracts body text
  preprocessing/
    spacy_processor.py             — SpacyProcessor + ProcessedArticle
    filters.py                     — filter_shamima_mentions (relevance filter)
  sentiment/lexicons/
    sentiment_analyzer.py          — LexiconScorer + SentimentScores
  bias/
    feature_builder.py             — FeatureBuilder + ArticleFeatures
    topic_clusterer.py             — TopicClusterer: k-means on feature matrix
  comparison/outlet_comparator.py  — outlet-level summary statistics
  pipeline/
    config.py                      — PipelineConfig (all output paths)
    news_pipeline.py               — orchestrates the full pipeline

tests/                             — unittest-based test suite
data/raw/                          — source CSV with article metadata
data/intermediate/                 — pipeline stage outputs (CSV checkpoints)
```

## Current state — active refactoring

The pipeline is functional end-to-end. A series of refactoring tasks are
in progress to fix code smells, remove dead code, separate preprocessing
per lexicon, and improve testability. Each task is defined in a numbered
prompt file under `codex_prompts/`.

**Prompts must be executed in order (01 through 10).** Some prompts depend
on changes made by earlier prompts.

## Key design decisions

### ProcessedArticle is the shared data object

`ProcessedArticle` (in `src/preprocessing/spacy_processor.py`) is a `@dataclass`
holding:
- `article_id: str` — SHA-256 hash of the article URL
- `raw_text: str` — original article body, unmodified
- `doc: spacy.tokens.Doc` — full spaCy Doc (tokens, POS, deps, NER)

Computed `@property` values (derived from `doc`, not stored):
- `vader_text -> str` — whitespace-collapsed raw text for VADER. No
  lowercasing, no tokenisation, no stop-word removal.
- `sentiwordnet_tokens -> list[tuple[str, str, str]]` — `(lemma, wn_pos, synset_name)`
  tuples. POS mapped from spaCy to WordNet. First-sense fallback when no
  synsets match. Stop words, punctuation, non-alpha filtered out.
- `nrc_tokens -> list[str]` — lowercased lemmas, stop words removed,
  alpha only.
- `lemmas -> list[str]` — same as `nrc_tokens` (kept for TF-IDF / clustering).
- `tokens -> list[str]` — lowercased token text, punctuation excluded.

`ProcessedArticle` is created once per article by `SpacyProcessor` and passed
to every downstream stage. No stage should call `nlp()` again.

### Each lexicon scorer receives only the preprocessing it needs

| Scorer         | Input property          | Preprocessing applied                                                    |
|----------------|-------------------------|--------------------------------------------------------------------------|
| VADER          | `vader_text`            | Whitespace collapse only. No lowercasing, tokenisation, or stop removal. |
| SentiWordNet   | `sentiwordnet_tokens`   | Tokenised, lemmatised, POS-tagged, first-sense WSD, lowercased, stops removed. |
| NRC            | `nrc_tokens`            | Tokenised, lemmatised, lowercased, stop words removed.                   |
| TF-IDF         | `lemmas`                | Same as nrc_tokens.                                                      |

### FeatureBuilder produces the feature matrix for clustering

`FeatureBuilder.build(articles)` returns a `scipy.sparse.csr_matrix` combining:
- TF-IDF features (up to 300 unigrams/bigrams from lemmatised text)
- 6 scaled linguistic features (adj_rate, adv_rate, modal_rate,
  attribution_rate, passive_rate, ner_rate)

This matrix is the input to `TopicClusterer`. The builder also exposes
`feature_names` for inspecting which TF-IDF terms and linguistic features
matter most per cluster.

### PipelineConfig centralises paths

All output paths live in a `PipelineConfig` dataclass in
`src/pipeline/config.py`. `NewsPipeline.__init__` takes a `PipelineConfig`.
No path strings are hardcoded in `news_pipeline.py`.

### SentimentScores is a typed return value

`SentimentScores` (in `sentiment_analyzer.py`) is a dataclass. The pipeline
flattens it with `dataclasses.asdict()` — no manual per-field mapping.

### CSV checkpoints are for debugging, not data transfer

The pipeline writes CSVs at each stage for inspection. But in-memory objects
are the primary data transfer between stages in `pipeline.run()`.

### article_id is assigned once at ingestion

`build_master_csv` creates the `article_id` column (SHA-256 of URL).
No downstream stage recomputes it. `_ensure_article_id` is removed.

### No abstract base classes unless needed

Use concrete classes and simple composition.

## Scope boundaries — what Codex must NOT do

- **Never modify files outside the task scope.** Each prompt specifies exactly
  which files to create or modify. Do not touch anything else.
- **Never add dependencies** not already listed in the project (spaCy, pandas,
  scikit-learn, NLTK, nrclex, scipy, numpy).
- **Never add logging frameworks, config systems, CLI arguments, or YAML files.**
- **Never add abstract base classes, protocols, or generics** unless the prompt
  explicitly requests them.
- **Never use pytest fixtures or parametrize.** Use `unittest.TestCase` only.
- **Never create `__init__.py` changes** unless the prompt explicitly says to.
- **Never add type: ignore comments** unless strictly necessary for spaCy stubs.
- **Never execute code at module level** (no bare function calls, no
  `nlp = spacy.load(...)` at module scope).
- **Never call `nlp()` on text that has already been processed.** Read from
  `ProcessedArticle.doc` instead.
- **Never write unit tests.** The human will write tests later. Do not create
  test files or test methods unless the prompt explicitly asks.
- **Never delete or rename test files** that already exist.

## Coding conventions

### Style
- Python 3.11+. Type hints on all public method signatures.
- Imports: standard library first, blank line, third-party, blank line, local.
- Docstrings on public classes and methods.
- No logging framework — `print()` sparingly for pipeline progress only.

### Naming
- Classes: `PascalCase`. Methods/variables: `snake_case`.
- Private: single underscore prefix. Constants: `UPPER_SNAKE_CASE`.
- Files: `snake_case.py`.

### Error handling
- `ValueError` for invalid input. Messages must name the specific problem.
- Do not catch and silence exceptions without a recovery strategy.

### Dependencies
- spaCy (`en_core_web_sm`) — tokenization, POS, NER, dependency parse.
- pandas — data transport.
- scikit-learn — TF-IDF, KMeans, StandardScaler, silhouette_score.
- NLTK — VADER, SentiWordNet, WordNet.
- nrclex — NRC emotion lexicon.
- numpy, scipy — numeric operations, sparse matrices.

## Code smells to avoid

- **Redundant spaCy calls.** Read from `ProcessedArticle.doc`, never call
  `nlp()` again.
- **Column-name coupling.** Pass `ProcessedArticle` objects, not DataFrames
  with hardcoded column names.
- **Module-level side effects.** No code at import time.
- **God objects.** Pipeline orchestrates; business logic lives in its own module.
- **Per-field mapping.** Use `dataclasses.asdict()` instead of mapping each
  field manually.
- **Per-call instantiation.** Expensive objects (NRCLex, SentimentIntensityAnalyzer)
  are created in `__init__`, not per article.

## Files to read before modifying a module

| Changing...                        | Read first                                         |
|------------------------------------|----------------------------------------------------|
| `src/preprocessing/spacy_processor.py` | `src/sentiment/lexicons/sentiment_analyzer.py`  |
| `src/sentiment/lexicons/sentiment_analyzer.py` | `src/preprocessing/spacy_processor.py`   |
| `src/pipeline/news_pipeline.py`    | `src/pipeline/config.py`, `src/bias/topic_clusterer.py` |
| `src/pipeline/config.py`          | `src/pipeline/news_pipeline.py`                    |
| `scripts/ingestion/build_master_csv.py` | `src/pipeline/news_pipeline.py`               |
| `src/preprocessing/filters.py`    | `src/preprocessing/article_preprocessor.py`        |
| any test file                      | the source file it tests + existing test patterns  |

## Prompt execution order

```
codex_prompts/
  01_move_filter_delete_dead_preprocessor.md
  02_assign_article_id_at_ingestion.md
  03_extract_pipeline_config.md
  04_add_vader_text_property.md
  05_add_sentiwordnet_tokens_with_pos.md
  06_add_nrc_tokens_property.md
  07_update_lexicon_scorer.md
  08_flatten_score_sentiment_with_asdict.md
  09_cache_nrclex_instance.md
  10_fix_run_pipeline_missing_import.md
```

Execute in order. After each prompt, verify the change compiles and
the existing tests still pass before proceeding.
