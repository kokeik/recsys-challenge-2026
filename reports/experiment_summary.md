# Experiment Summary

## 1. Why turn-aware modeling?

Turn 1 has no previously recommended item. Sequential recommenders therefore receive an
empty history, while dialogue and profile retrievers are at their most important. From
Turn 2 onward, track history becomes a high-signal representation of the user's evolving
taste. Mixing both regimes in one training objective diluted this distinction.

The resulting policy was:

- Turn 1: dialogue-based dense and lexical retrieval
- Turn 2+: sequential and history-based retrieval followed by learned reranking
- Submission: merge the outputs using the turn number

## 2. Relation-aware Item2Vec

The base skip-gram corpus contained ordered track sequences. Additional two-token
sentences connected each track to artist, album, tags, title tokens, release period,
popularity bucket, duration bucket, and ISRC-derived attributes. Dialogue text was also
tested as a training-only relation.

The central result was that text improved the geometry of the item space even when text
vectors were not used at inference. User utterances gave the strongest Recall@20–200;
music thoughts gave the strongest Recall@500.

## 3. Candidate complementarity

Item2Vec and exact artist-history retrieval overlapped substantially, but artist retrieval
still contributed about 4.4 percentage points of unique target coverage. Expanding the
artist source beyond roughly 75–100 items had diminishing returns. TF-IDF supplied a
different signal when track, artist, album, or tag terms appeared explicitly in dialogue.

This motivated source-specific Top-N limits instead of giving every retriever the same
candidate budget.

## 4. Learning to rank

Each candidate retained its origin rather than receiving only one fused score. Features
included source presence and rank, artist/album matches against history, query-to-track
token overlap, turn number, and interactions between retrieval confidence and history
statistics.

The primary ranker was LightGBM LambdaRank. Training candidates were generated
out-of-fold so that the ranker did not learn from unrealistically optimistic in-sample
retriever outputs. Queries whose target was absent from the candidate pool were analyzed
as retrieval failures instead of being treated as ordinary ranking negatives.

## 5. What did not work reliably?

- Adding more weak sources often improved candidate recall but reduced blind-set ranking.
- Raw source scores were less stable across retrievers than normalized rank features.
- One ranker for all turns underused the structural difference between empty and non-empty
  histories.
- Text mixed directly into the Item2Vec inference vector slightly degraded performance.
- Richer graph-style Item2Vec variants did not automatically outperform the focused
  relation design.

## 6. Next improvements

- Calibrate scores per source before cross-source use.
- Learn an explicit mixture-of-experts gate from query specificity and history entropy.
- Evaluate uncertainty-aware candidate budgets.
- Replace hand-built cross features with a compact listwise neural reranker while keeping
  the OOF protocol.

