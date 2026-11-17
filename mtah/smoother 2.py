"""A second model that reads the first model's probabilities over time. Part of the v6 candidate.

The decoder treats every sample's probabilities as fresh evidence, but neighbouring samples share most
of their inputs. This model sees the shape of the evidence over a few seconds: a real short possession
is a sharp, tall bump, a stray touch is a low one, a drift is neither. It is trained on probabilities
the first model gave for open matches it had not seen, and hands sharper probabilities to the same
decoder.

Built to rebuild identically on any chip. The first model's probabilities are rounded to four decimals
before use, which absorbs last-bit differences between maths libraries, and everything after that is
+, -, *, / and comparisons, which round the same way everywhere.
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .config import SPELL_WEIGHT
from .match import A, D, H, X

OFFSETS = tuple(range(-10, 11)) + (-40, -30, -25, -20, -15, -12, 12, 15, 20, 25, 30, 40)
WINDOWS = (5, 15, 40)
DECIMALS = 4


@dataclass(frozen=True)
class SmootherConfig:
    max_iter: int = 150
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    seed: int = 0


def _shift(x: np.ndarray, by: int) -> np.ndarray:
    """x moved in time by `by` samples (positive looks ahead), NaN past the ends."""
    out = np.full_like(x, np.nan)
    if by > 0:
        out[:-by] = x[by:]
    elif by < 0:
        out[-by:] = x[:by]
    else:
        out[:] = x
    return out


def _mean(x: np.ndarray, before: int, after: int) -> np.ndarray:
    """Mean over [t-before, t+after], shortened at the ends of the half."""
    total = np.concatenate(([0.0], np.cumsum(x)))
    index = np.arange(len(x))
    lo = np.maximum(index - before, 0)
    hi = np.minimum(index + after + 1, len(x))
    return (total[hi] - total[lo]) / (hi - lo)


def evidence(prob: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Inputs for the smoother, [n, k] float64. Team evidence is P(H) - P(A), dead evidence is P(D).

    Columns: for each offset, team then dead; then for each window, team before, team after, dead
    before, dead after. Nothing crosses a half.
    """
    rounded = np.round(np.asarray(prob, dtype=np.float64), DECIMALS)
    blocks = []
    for h in np.unique(half):
        p = rounded[half == h]
        team, dead = p[:, H] - p[:, A], p[:, D]
        columns = []
        for k in OFFSETS:
            columns += [_shift(team, k), _shift(dead, k)]
        for w in WINDOWS:
            columns += [_mean(team, w, 0), _mean(team, 0, w), _mean(dead, w, 0), _mean(dead, 0, w)]
        blocks.append(np.column_stack(columns))
    return np.vstack(blocks)


def _team_columns() -> np.ndarray:
    index = list(range(0, 2 * len(OFFSETS), 2))
    base = 2 * len(OFFSETS)
    for j in range(len(WINDOWS)):
        index += [base + 4 * j, base + 4 * j + 1]
    return np.array(index)


def swap(values: np.ndarray) -> np.ndarray:
    """The same inputs with home and away exchanged: team evidence changes sign."""
    out = values.copy()
    out[:, _team_columns()] *= -1.0
    return out


def sample_weights(labels: np.ndarray, frames: np.ndarray, fps: int = 25,
                   strength: float = SPELL_WEIGHT) -> np.ndarray:
    """The first model's weighting: samples in short possessions count for more."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    sizes = np.diff(np.concatenate(([0], edges, [len(labels)])))
    return (np.repeat(sizes, sizes)[frames] / fps).clip(1.0, 60.0) ** -strength


class Smoother:
    def __init__(self, config: SmootherConfig = SmootherConfig()):
        self.config = config
        self.model = None

    def fit(self, sets: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """sets: (prob [n, 3], half [n], label at each sample [n], weight [n]) per open match."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        ws: List[np.ndarray] = []
        for prob, half, labels, weights in sets:
            values = evidence(prob, half)
            keep = labels != X
            mirrored = labels[keep].copy()
            mirrored[labels[keep] == H] = A
            mirrored[labels[keep] == A] = H
            xs += [values[keep], swap(values[keep])]
            ys += [labels[keep], mirrored]
            ws += [weights[keep], weights[keep]]
        c = self.config
        self.model = HistGradientBoostingClassifier(
            max_iter=c.max_iter, learning_rate=c.learning_rate, max_leaf_nodes=c.max_leaf_nodes,
            min_samples_leaf=c.min_samples_leaf, l2_regularization=c.l2_regularization,
            early_stopping=False, random_state=c.seed)
        self.model.fit(np.vstack(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))

    def probabilities(self, prob: np.ndarray, half: np.ndarray) -> np.ndarray:
        values = evidence(prob, half)
        direct = self.model.predict_proba(values)
        mirrored = self.model.predict_proba(swap(values))[:, (A, H, D)]
        return (direct + mirrored) / 2.0
