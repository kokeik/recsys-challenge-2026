# RecSys Challenge 2026

A two-stage retrieval and learning-to-rank system for conversational music
recommendation. It treats the cold first turn separately from later turns where
previously recommended tracks provide a strong sequential signal.

This repository is a compact, public reimplementation of my contribution to a team
entry for the ACM RecSys Challenge 2026. My scope covered Item2Vec retrieval for turns
2+, multi-source candidate pooling, LightGBM reranking, and turn-wise evaluation.

It does **not** contain the team's full submission, challenge data, blind-set records,
trained artifacts, or API-generated embeddings. The runnable example uses synthetic data.

## Highlights

- Relation-aware Item2Vec using session sequences and track metadata
- Union of Item2Vec, lexical, dense, and artist-history candidate sources
- Per-source rank features and history-aware artist/album features
- LightGBM LambdaRank with a scikit-learn fallback
- Explicit `turn_number >= 2` training policy
- Group-aware and turn-wise Hit Rate / nDCG evaluation
- Out-of-fold candidate generation in the original experiment workflow

## Selected development-set observations

- User-text-enriched Item2Vec: Recall@20 **0.3054**, Recall@100 **0.5050**
- Item2Vec 300 + artist 100 + TF-IDF 200: candidate Recall **0.7407**
- Final turn-aware reranker: Hit@20 **0.3779**, nDCG@20 **0.1880**

These values are aggregate observations from the team development environment. The
synthetic demo documents the pipeline, but is not intended to reproduce the benchmark.

## Run the demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
python -m turnaware_recsys.demo
```

See the [Japanese README](README.md) and the
[experiment report](reports/experiment_summary.md) for the full design rationale.
