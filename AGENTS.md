# AGENTS.md

## Project overview

A Python NLP pipeline analysing UK news coverage of Shamima Begum across five
outlets (BBC, Guardian, Daily Mail, The Sun, The Independent). The pipeline
extracts article text from the web, preprocesses it with spaCy, scores
sentiment with three lexicon-based tools (VADER, NRC, SentiWordNet), and will
include a logistic regression bias classifier using TF-IDF and linguistic
features. The dataset is small (88 articles) — this is a research prototype,
not a production system.

## Architecture

```
scripts/
  ingestion/build_master_csv.py    — raw CSV → cleaned master CSV
  run_pipeline.py                  — CLI entry point

src/
  extraction/web_extractor.py      — fetches article HTML, extracts body text
  preprocessing/
    spacy_processor.py             — SpacyProcessor + ProcessedArticle (MUST CREATE)
    article_preprocessor.py        — legacy preprocessor + filter_shamima_mentions
  sentiment/lexicons/
    sentiment_analyzer.py          — LexiconScorer + SentimentScores
  bias/
    feature_builder.py             — (planned) FeatureBuilder + ArticleFeatures
    bias_classifier.py             — (planned) logistic regression wrapper
  comparison/outlet_comparator.py  — outlet-level summary statistics
  pipeline/news_pipeline.py        — orchestrates the full pipeline

tests/                             — unittest-based test suite
data/raw/                          — source CSV with article metadata
data/intermediate/                 — pipeline stage outputs (CSV checkpoints)
```

## Current state — what exists and what is missing

The pipeline (`news_pipeline.py`) and sentiment analyzer (`sentiment_analyzer.py`)
have already been rewritten to use `ProcessedArticle` and `SentimentScores`.
However, the file they import from does not exist yet:

- `src/preprocessing/spacy_processor.py` — **DOES NOT EXIST**. Must be created
  first. Both `news_pipeline.py` and `sentiment_analyzer.py` import
  `SpacyProcessor` and `ProcessedArticle` from this module. The pipeline will
  not run until this file is created.

The remaining implementation sequence is:
1. Create `spacy_processor.py` (unblocks everything)
2. Create `feature_builder.py` (the main new feature)
3. Create `bias_classifier.py` (consumes features)
4. Wire bias classifier into the pipeline

## Key design decisions

### ProcessedArticle is the shared data object

`ProcessedArticle` (to be created in `src/preprocessing/spacy_processor.py`)
is a `@dataclass` holding:
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

### SentimentScores is a typed return value

`SentimentScores` (in `sentiment_analyzer.py`) is a dataclass with fields:
`vader: float`, `sentiwordnet: float`, `nrc: float`. The `score_dataframe`
backward-compat wrapper maps these to DataFrame column names with `_score`
suffixes.

### CSV checkpoints are for debugging, not data transfer

The pipeline writes CSVs at each stage for inspection. But in-memory
`ProcessedArticle` objects are the primary data transfer between stages in
`pipeline.run()`.

### No abstract base classes unless needed

Use concrete classes and simple composition. Introduce an ABC only when there
are 3+ interchangeable implementations. Private methods inside a class are fine.

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
  The project pattern: `FakeToken` + `FakeNlp`/`FakeDoc` classes.
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
- scikit-learn — TF-IDF, logistic regression, StandardScaler.
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
- **Defensive checks everywhere.** Validate once at the boundary.

## Files to read before modifying a module

| Changing...                    | Read first                                      |
|-------------------------------|--------------------------------------------------|
| `src/preprocessing/`          | `tests/test_article_preprocessor.py`             |
| `src/sentiment/`              | `src/preprocessing/spacy_processor.py`           |
| `src/bias/`                   | `src/preprocessing/spacy_processor.py`           |
| `src/pipeline/news_pipeline.py` | every module in `src/` it imports               |
| any test file                 | the source file it tests + existing test patterns |
