from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .features import Track

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _relation_tokens(track: Track) -> list[str]:
    tokens = [
        f"artist::{track.artist.casefold()}",
        f"album::{track.album.casefold()}",
        *(f"tag::{tag.casefold()}" for tag in track.tags),
    ]
    tokens.extend(f"title::{token}" for token in TOKEN_PATTERN.findall(track.title.casefold()))
    if track.year is not None:
        tokens.extend((f"year::{track.year}", f"decade::{track.year // 10 * 10}"))
    return [token for token in tokens if not token.endswith("::")]


@dataclass(frozen=True)
class Item2VecConfig:
    vector_size: int = 64
    window: int = 5
    negative: int = 10
    epochs: int = 20
    seed: int = 42


class RelationItem2Vec:
    """Item2Vec enriched with track-to-metadata relation sentences.

    Session sequences teach item co-occurrence. Two-token relation sentences such as
    ``[track_id, artist::name]`` connect cold or infrequent items through metadata.
    """

    def __init__(self, config: Item2VecConfig | None = None) -> None:
        self.config = config or Item2VecConfig()
        self.model = None
        self.track_ids: list[str] = []
        self.track_matrix: np.ndarray | None = None

    def build_sentences(
        self,
        sessions: Iterable[Sequence[str]],
        catalog: Mapping[str, Track],
        text_relations: Mapping[str, Sequence[str]] | None = None,
    ) -> list[list[str]]:
        sentences = [list(sequence) for sequence in sessions if len(sequence) >= 2]
        for track_id, track in catalog.items():
            sentences.extend([[track_id, token] for token in _relation_tokens(track)])
        for track_id, tokens in (text_relations or {}).items():
            sentences.extend([[track_id, f"text::{token.casefold()}"] for token in tokens])
        return sentences

    def fit(
        self,
        sessions: Iterable[Sequence[str]],
        catalog: Mapping[str, Track],
        text_relations: Mapping[str, Sequence[str]] | None = None,
    ) -> RelationItem2Vec:
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the item2vec extra: pip install -e '.[item2vec]'") from exc

        cfg = self.config
        self.model = Word2Vec(
            sentences=self.build_sentences(sessions, catalog, text_relations),
            vector_size=cfg.vector_size,
            window=cfg.window,
            min_count=1,
            workers=1,
            sg=1,
            negative=cfg.negative,
            epochs=cfg.epochs,
            seed=cfg.seed,
        )
        self.track_ids = sorted(track_id for track_id in catalog if track_id in self.model.wv)
        matrix = np.asarray([self.model.wv[track_id] for track_id in self.track_ids])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.track_matrix = matrix / np.maximum(norms, 1e-12)
        return self

    def recommend(
        self, history_track_ids: Sequence[str], *, top_k: int = 100
    ) -> tuple[list[str], list[float]]:
        if self.model is None or self.track_matrix is None:
            raise RuntimeError("fit must be called before recommend")
        vectors = [self.model.wv[item] for item in history_track_ids if item in self.model.wv]
        if not vectors:
            return [], []
        query = np.mean(vectors, axis=0)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        similarities = self.track_matrix @ query
        seen = set(history_track_ids)
        order = np.argsort(-similarities, kind="stable")
        indices = [index for index in order if self.track_ids[index] not in seen][:top_k]
        return (
            [self.track_ids[index] for index in indices],
            [float(similarities[index]) for index in indices],
        )

