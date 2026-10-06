# AGENTS.md

## Project identity

This repository is Keijiro Konishi's public portfolio reimplementation of his
contribution to a team entry for the ACM RecSys Challenge 2026.

The public-facing repository name and README title are **RecSys Challenge 2026**.
The technical focus remains turn-aware conversational music recommendation:

- Turn 1: dialogue-oriented retrieval because no item history exists.
- Turn 2+: Item2Vec/history retrieval, candidate union, and learned reranking.

Do not describe this repository as the complete team submission or claim that all
components of the original team system were authored by Konishi.

## Read first

Before making structural or modeling changes, read:

1. `README.md`
2. `reports/experiment_summary.md`
3. `docs/LOCAL_DEVELOPMENT.md`

## Repository map

- `src/turnaware_recsys/item2vec.py`: relation-aware Item2Vec.
- `src/turnaware_recsys/lexical.py`: compact TF-IDF retriever for the demo.
- `src/turnaware_recsys/candidate_pool.py`: deterministic multi-source union.
- `src/turnaware_recsys/features.py`: source-rank, history, and text features.
- `src/turnaware_recsys/reranker.py`: LightGBM LambdaRank with sklearn fallback.
- `src/turnaware_recsys/evaluation.py`: Hit Rate and nDCG by turn.
- `src/turnaware_recsys/demo.py`: synthetic end-to-end example.
- `examples/synthetic_data.json`: fictional data safe for public distribution.
- `tests/`: unit tests.

## Development rules

- Keep the repository independently runnable; do not import from the former team
  repository using relative filesystem paths.
- Use Python 3.10 or newer.
- Add type hints to public functions and keep feature order deterministic.
- Preserve the explicit `turn_number >= 2` policy in sequential-ranker training unless
  a documented experiment intentionally changes it.
- Prevent data leakage. Candidate models used to train a reranker should generate
  out-of-fold predictions in benchmark-scale experiments.
- Add or update tests for behavior changes.
- Run `pytest -q` and `ruff check src tests` before committing.
- Keep generated models, embeddings, caches, datasets, and prediction dumps out of Git.
- Do not add a license file unless the repository owner explicitly requests one.

## Public-data and attribution constraints

Never commit any of the following:

- Challenge train/dev/blind records or ground-truth files.
- Spotify-derived raw data.
- API-generated embeddings or API credentials.
- Trained model binaries from the team environment.
- Teammates' code without explicit permission and attribution.
- Full prediction/submission JSON files.

Aggregate metrics and original plots derived from aggregate statistics are acceptable.
Clearly label competition-environment metrics separately from synthetic-demo results.

## Verification commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pytest -q
ruff check src tests
python -m turnaware_recsys.demo
```

The demo may use the sklearn fallback when LightGBM is not installed. A perfect score on
the tiny synthetic example is not a competition result and must not be presented as one.

## Git and GitHub

Canonical remote:

```text
https://github.com/kokeik/recsys-challenge-2026
```

- Use small, descriptive commits.
- Inspect `git diff` and `git status` before committing.
- Never commit credentials, private keys, `.env` files, or local absolute paths.
- Do not rewrite published history or force-push unless the owner explicitly requests it.
- Before changing public claims or benchmark numbers, point to the evidence used.

## Suggested next improvements

Prioritize portfolio clarity over adding many experimental models:

1. Add a small OOF candidate-generation example.
2. Add feature-importance visualization from synthetic or sanitized aggregate data.
3. Add a challenge-format adapter without bundling challenge records.
4. Add integration tests for train/rank/evaluate serialization.
5. Add a concise architecture image for social previews.

