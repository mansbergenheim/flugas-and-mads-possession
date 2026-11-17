"""The one match format every loader produces and every model consumes.

Coordinates are metres with the origin at the centre spot, x along the pitch and y across it.
Frames are contiguous inside each half; `halves` holds half-open [start, stop) spans. Labels use
the competition codes, so a match from any source can be scored the same way.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = "HADX"
HOME, AWAY = 0, 1


@dataclass
class Match:
    match_id: str
    source: str
    fps: int
    pitch: Tuple[float, float]
    ball: np.ndarray                   # (T, 2) float32, NaN where missing
    players: np.ndarray                # (T, P, 2) float32, NaN where missing
    team: np.ndarray                   # (P,) int8, HOME or AWAY
    halves: List[Tuple[int, int]]
    labels: Optional[np.ndarray] = None  # (T,) int8, or None when there is no ground truth
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])

    def validate(self) -> None:
        """Raise on anything that would silently corrupt a score later."""
        n = self.n_frames
        if self.ball.shape != (n, 2):
            raise ValueError(f"{self.match_id}: ball shape {self.ball.shape}")
        if self.players.ndim != 3 or self.players.shape[0] != n or self.players.shape[2] != 2:
            raise ValueError(f"{self.match_id}: players shape {self.players.shape}")
        if self.team.shape != (self.players.shape[1],):
            raise ValueError(f"{self.match_id}: team shape {self.team.shape}")
        if not set(np.unique(self.team).tolist()) <= {HOME, AWAY}:
            raise ValueError(f"{self.match_id}: team values outside HOME/AWAY")
        if self.labels is not None:
            if self.labels.shape != (n,):
                raise ValueError(f"{self.match_id}: labels shape {self.labels.shape}")
            if self.labels.min() < 0 or self.labels.max() > X:
                raise ValueError(f"{self.match_id}: label codes outside 0..3")
        last = 0
        for start, stop in self.halves:
            if not last <= start < stop <= n:
                raise ValueError(f"{self.match_id}: bad half span {(start, stop)}")
            last = stop

    def with_tracking(self, ball: np.ndarray, players: np.ndarray, team: np.ndarray,
                      **meta) -> "Match":
        """A copy with replaced tracking and the same labels. Used by the degrader."""
        return replace(self, ball=ball, players=players, team=team,
                       meta={**self.meta, **meta})


def save_cache(match: Match, path: str) -> None:
    match.validate()
    np.savez_compressed(
        path, match_id=match.match_id, source=match.source, fps=match.fps,
        pitch=np.array(match.pitch), ball=match.ball, players=match.players, team=match.team,
        halves=np.array(match.halves),
        labels=match.labels if match.labels is not None else np.array([], dtype=np.int8))


def load_cache(path: str) -> Match:
    with np.load(path, allow_pickle=False) as z:
        labels = z["labels"]
        match = Match(
            match_id=str(z["match_id"]), source=str(z["source"]), fps=int(z["fps"]),
            pitch=tuple(float(v) for v in z["pitch"]), ball=z["ball"], players=z["players"],
            team=z["team"], halves=[tuple(int(v) for v in row) for row in z["halves"]],
            labels=labels if labels.size else None)
    match.validate()
    return match
