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
  ingestion/build_master_csv.py    — raw CSV → cleaned master CSV
  run_pipeline.py                  — CLI entry point

src/
  extraction/web_extractor.py      — fetches article HTML, extracts body text
  preprocessing/
    spacy_processor.py             — SpacyProcessor + ProcessedArticle
    article_preprocessor.py        — legacy preprocessor + filter_shamima_mentions
  sentiment/lexicons/
    sentiment_analyzer.py          — LexiconScorer + SentimentScores
  bias/
    feature_builder.py             — FeatureBuilder + ArticleFeatures
    topic_clusterer.py             — TopicClusterer: k-means on feature matrix
  comparison/outlet_comparator.py  — outlet-level summary statistics
  pipeline/news_pipeline.py        — orchestrates the full pipeline

tests/                             — unittest-based test suite
data/raw/                          — source CSV with article metadata
data/intermediate/                 — pipeline stage outputs (CSV checkpoints)
```

## Current state — what exists and what is being changed

The pipeline is functional end-to-end. The sentiment scoring, preprocessing,
extraction, and comparison stages are complete and stable.

**Active refactor: replacing the logistic regression classifier with k-means
clustering.**

The old `src/bias/bias_classifier.py` used supervised logistic regression with
outlet as a proxy label. This is being replaced by `src/bias/topic_clusterer.py`
which uses unsupervised k-means clustering on the same feature matrix that
`FeatureBuilder` already produces. The goal: group articles by writing style
and word choice, then examine which outlets land in which clusters.

Files being changed:
1. **Delete** `src/bias/bias_classifier.py`
2. **Create** `src/bias/topic_clusterer.py`
3. **Modify** `src/pipeline/news_pipeline.py` — replace `run_bias_classifier`
   with `run_clustering`, update imports, update `run()` method

Files that must NOT change:
- `src/bias/feature_builder.py` — stays exactly as-is
- `src/preprocessing/spacy_processor.py` — stays exactly as-is
- `src/sentiment/` — stays exactly as-is
- `src/extraction/` — stays exactly as-is
- `src/comparison/` — stays exactly as-is
- `scripts/` — stays exactly as-is

## Key design decisions

### ProcessedArticle is the shared data object

`ProcessedArticle` (in `src/preprocessing/spacy_processor.py`) is a `@dataclass`
holding:
- `article_id: str` — SHA-256 hash of the article URL
- `raw_text: str` — original article body, unmodified
- `doc: spacy.tokens.Doc` — full spaCy Doc (tokens, POS, deps, NER)

Computed `@property` values (derived from `doc`, not stored):
- `minimal_text -> str` — `raw_text.strip()` (used by VADER)
- `lemmas -> list[str]` — lowercased lemmas, excluding stopwords, punctuation,
  and non-alphabetic tokens (used by TF-IDF, NRC, SentiWordNet)
- `tokens -> list[str]` — lowercased token text, excluding punctuation only

`ProcessedArticle` is created once per article by `SpacyProcessor` and passed
to every downstream stage. No stage should call `nlp()` again.

### FeatureBuilder produces the feature matrix for clustering

`FeatureBuilder.build(articles)` returns a `scipy.sparse.csr_matrix` combining:
- TF-IDF features (up to 300 unigrams/bigrams from lemmatised text)
- 6 scaled linguistic features (adj_rate, adv_rate, modal_rate,
  attribution_rate, passive_rate, ner_rate)

This matrix is the input to `TopicClusterer`. The builder also exposes
`feature_names` for inspecting which TF-IDF terms and linguistic features
matter most per cluster.

### TopicClusterer design constraints

- Input: the `csr_matrix` from `FeatureBuilder.build()`, plus a list of
  `article_id` strings and a list of `news_outlet` strings (for labelling
  output, not for training — k-means is unsupervised)
- Must run k-means for a range of k values (2–8) and select best k by
  silhouette score
- Output: a `ClusteringResult` dataclass containing:
  - `assignments: pd.DataFrame` — columns: article_id, news_outlet, cluster
  - `top_terms: dict[int, list[str]]` — top 10 TF-IDF/linguistic feature
    names per cluster by mean weight
  - `silhouette_score: float` — silhouette score for the chosen k
  - `k: int` — the chosen number of clusters
- The pipeline writes `assignments` to CSV at
  `data/intermediate/cluster_assignments.csv`
- The pipeline writes `top_terms` to CSV at
  `data/intermediate/cluster_top_terms.csv`

### SentimentScores is a typed return value

`SentimentScores` (in `sentiment_analyzer.py`) is a dataclass with fields:
`vader: float`, `sentiwordnet: float`, `nrc: float`.

### CSV checkpoints are for debugging, not data transfer

The pipeline writes CSVs at each stage for inspection. But in-memory objects
are the primary data transfer between stages in `pipeline.run()`.

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

## Coding conventions

### Style
- Python 3.11+. Type hints on all public method signatures.
- Imports: standard library first, blank line, third-party, blank line, local.
- Docstrings on public classes and methods.
- No logging framework — `print()` sparingly for pipeline progress only.

### Testing
- Framework: `unittest.TestCase`. Not pytest.
- File naming: `tests/test_<module_name>.py`.
- Method naming: `test_<method_or_behaviour>_<expected_outcome>`.
- Use test doubles (fakes) to avoid loading spaCy/NLTK in unit tests.
- Run: `python -m pytest tests/ -v` from project root.

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

## Files to read before modifying a module

| Changing...                      | Read first                                       |
|----------------------------------|--------------------------------------------------|
| `src/bias/topic_clusterer.py`    | `src/bias/feature_builder.py`                    |
| `src/pipeline/news_pipeline.py`  | `src/bias/topic_clusterer.py`, `feature_builder.py` |
| any test file                    | the source file it tests + existing test patterns |
