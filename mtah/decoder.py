"""Turn per-sample probabilities into whole possessions.

The metric punishes flicker hard, so the decoder chooses the single H/A/D path that best explains
the evidence while paying a cost for every change. Different changes cost different amounts: going
straight from one team to the other is the most expensive, since a stray touch must not count.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .match import A, D, H


@dataclass(frozen=True)
class DecoderConfig:
    temperature: float = 1.0   # scales evidence against change costs
    turnover: float = 12.0     # H<->A
    to_dead: float = 8.0       # H or A -> D
    from_dead: float = 6.0     # D -> H or A


def transition_costs(cfg: DecoderConfig) -> np.ndarray:
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = cfg.turnover
    cost[H, D] = cost[A, D] = cfg.to_dead
    cost[D, H] = cost[D, A] = cfg.from_dead
    return cost


def viterbi(log_p: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Best path through log_p [n, 3] with cost[from, to] charged per change."""
    n = len(log_p)
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_p[0].copy()
    for t in range(1, n):
        options = score[:, None] - cost
        best = options.argmax(axis=0)
        back[t] = best
        score = options[best, (0, 1, 2)] + log_p[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_samples(prob: np.ndarray, half: np.ndarray, cfg: DecoderConfig,
                   log_prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Decode each half separately. log_prior [n, 3] adds outside evidence such as Smart Tagging."""
    log_p = cfg.temperature * np.log(np.clip(prob, 1e-6, 1.0))
    if log_prior is not None:
        log_p = log_p + log_prior
    cost = transition_costs(cfg)
    out = np.empty(len(prob), dtype=np.int8)
    for h in np.unique(half):
        idx = half == h
        out[idx] = viterbi(log_p[idx], cost)
    return out


def expand(path: np.ndarray, frames: np.ndarray, half: np.ndarray, halves, n_frames: int,
           default: int = D) -> np.ndarray:
    """Sample labels back to every frame. A boundary falls midway between two samples."""
    out = np.full(n_frames, default, dtype=np.int8)
    for h, (start, stop) in enumerate(halves):
        idx = half == h
        if not idx.any():
            continue
        f, p = frames[idx], path[idx]
        edges = np.concatenate(([start], (f[1:] + f[:-1] + 1) // 2, [stop]))
        out[start:stop] = np.repeat(p, np.diff(edges))
    return out
