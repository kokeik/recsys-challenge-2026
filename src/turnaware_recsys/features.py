from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .candidate_pool import Candidate

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.casefold()))


@dataclass(frozen=True)
class Track:
    track_id: str
    title: str
    artist: str
    album: str = ""
    tags: tuple[str, ...] = ()
    year: int | None = None

    @property
    def search_text(self) -> str:
        return " ".join((self.title, self.artist, self.album, *self.tags))


@dataclass(frozen=True)
class QueryContext:
    session_id: str
    turn_number: int
    query_text: str
    history_track_ids: tuple[str, ...] = ()
    category: str = "unknown"
    specificity: str = "unknown"


@dataclass
class FeatureExtractor:
    catalog: Mapping[str, Track]
    source_names: Sequence[str]
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _history_stats(self, context: QueryContext) -> tuple[dict[str, int], dict[str, int]]:
        artists: dict[str, int] = {}
        albums: dict[str, int] = {}
        for track_id in context.history_track_ids:
            track = self.catalog.get(track_id)
            if track is None:
                continue
            artists[track.artist] = artists.get(track.artist, 0) + 1
            albums[track.album] = albums.get(track.album, 0) + 1
        return artists, albums

    def transform_one(self, candidate: Candidate, context: QueryContext) -> dict[str, float]:
        track = self.catalog[candidate.track_id]
        values = candidate.rank_features(self.source_names)
        history_artists, history_albums = self._history_stats(context)
        history_size = max(len(context.history_track_ids), 1)
        query_tokens = _tokens(context.query_text)
        track_tokens = _tokens(track.search_text)

        values.update(
            {
                "turn_number": float(context.turn_number),
                "history_length": float(len(context.history_track_ids)),
                "same_artist_count": float(history_artists.get(track.artist, 0)),
                "same_artist_ratio": history_artists.get(track.artist, 0) / history_size,
                "same_album_count": float(history_albums.get(track.album, 0)),
                "same_album_ratio": history_albums.get(track.album, 0) / history_size,
                "query_track_overlap": len(query_tokens & track_tokens)
                / max(len(query_tokens), 1),
                "query_track_jaccard": len(query_tokens & track_tokens)
                / max(len(query_tokens | track_tokens), 1),
                "release_year_scaled": 0.0
                if track.year is None
                else (track.year - 1950) / 100.0,
            }
        )
        return values

    def transform(
        self, candidates: Sequence[Candidate], context: QueryContext
    ) -> tuple[np.ndarray, list[str]]:
        rows = [self.transform_one(candidate, context) for candidate in candidates]
        if not rows:
            return np.empty((0, 0), dtype=np.float32), []
        names = sorted(rows[0])
        self.feature_names_ = names
        matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
        return matrix, names

