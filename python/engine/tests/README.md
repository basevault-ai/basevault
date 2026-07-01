# Pipeline tests

Two tiers.

## Core (default)

Top-level files. Fast, deterministic, no real LLM. Gate every PR.

```
pytest tests/
```

| File                          | What                                              |
|-------------------------------|---------------------------------------------------|
| `test_content_extractor.py`   | Parsing, offset math. Mocked LLM.                 |
| `test_ingestor.py`            | Format detection, zip/pdf/image handling.         |
| `test_metadata_extractor.py`  | Parsing, taxonomy filtering.                      |
| `test_splitter.py`            | Chunking, origin-char propagation.                |
| `test_splitter_corpora.py`    | Structural invariants on real corpora. No LLM.    |
| `test_runner.py`              | End-to-end runner with mocked LLM.                |
| `test_tokens.py`              | Tokenizer fallback.                               |

## Quality (`quality/`)

Opt-in LLM-behavior tests. They hit a real provider and assert on
*what the model returned* — expected topics, entity types, emotion/
signal labels, dates. Fails on model drift; not a gate for unrelated
PRs.

```
pytest tests/quality/ -m integration    # quality/integration tests
pytest tests/quality/ -m regression     # per-stage regression suites
```

| File                              | What                                           |
|-----------------------------------|------------------------------------------------|
| `test_content_extractor.py`       | Real-LLM: extract emits expected items.        |
| `test_metadata_extractor.py`      | Real-LLM: topic/people extraction per case.    |
| `test_extract_regression.py`      | Regression cases under `.regression/extract/`. |
| `test_patterns_regression.py`     | Regression cases under `.regression/patterns/`.|

Quality / regression suites that drive an LLM live under
`testing/` (`testing/eval/`, `testing/integration/`); the
`engine/tests/` tree is stub-only after the production gut.

## Why two tiers?

Core failures block merge. Quality failures point at model regressions
that need triage but shouldn't block unrelated code changes. Each LLM
call in a quality test runs in parallel via a module-scope
`ThreadPoolExecutor` (per AGENTS.md), so the whole quality tier
finishes in a handful of minutes.
