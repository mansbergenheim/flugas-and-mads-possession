"""Reference predictors. NearestPlayer re-implements the organisers' published baseline.

Its value here is calibration: it can run on degraded open matches and be compared with the scores
the organisers published for the real match.
"""

import numpy as np

from .match import A, D, H, Match


def nearest_team(match: Match, chunk: int = 20000) -> np.ndarray:
    """Team of the tracked player nearest the ball. -1 where there is no answer."""
    out = np.full(match.n_frames, -1, dtype=np.int8)
    for lo in range(0, match.n_frames, chunk):
        hi = min(lo + chunk, match.n_frames)
        delta = match.players[lo:hi] - match.ball[lo:hi, None, :]
        dist = np.einsum("tpd,tpd->tp", delta, delta)
        any_player = np.isfinite(dist).any(axis=1)
        dist[~np.isfinite(dist)] = np.inf
        nearest = np.argmin(dist, axis=1)
        out[lo:hi] = np.where(any_player, match.team[nearest], -1)
    return out


def majority(codes: np.ndarray, window: int) -> np.ndarray:
    """Centred majority vote over +/- window frames. D where nobody was seen, ties to home."""
    if window <= 0:
        return np.where(codes == 0, H, np.where(codes == 1, A, D)).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    n_home, n_away = home[high] - home[low], away[high] - away[low]
    out = np.full(len(codes), D, dtype=np.int8)
    seen = (n_home + n_away) > 0
    out[seen] = np.where(n_home[seen] >= n_away[seen], H, A)
    return out


class NearestPlayer:
    """The organisers' baseline: nearest player, then a majority vote."""

    def __init__(self, window: int = 50):
        self.window = window
        self.name = f"nearest_w{window}"

    def fit(self, matches) -> None:
        pass

    def predict(self, match: Match) -> np.ndarray:
        return majority(nearest_team(match), self.window)
