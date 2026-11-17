"""Make clean tracking from other matches look like the competition's automatic tracking.

Clean TRACAB or Metrica data has the ball on nearly every live frame and 22 stable identities. The
competition ATD has the ball on 59% of graded frames, lost in long blackouts, about 20.5 objects,
and slots that do not follow a person. A model validated on clean data tells you nothing.

Ball blackouts are drawn from the real gap and run lengths in degrade_stats.json, which comes from
the organisers' public baseline_raw.csv. scripts/compare_tracking.py checks the rest against the real
tracking in words, using no answers: the ball and players are cleaner and tracks longer than first
estimated, so DegradeConfig and random_config were moved towards that.

How blackouts relate to stoppages:
  - Metrica's samples leave the ball blank on nearly every dead frame, so "no ball" means "dead"
    there, which a model must not learn. fill_dead_ball puts a plausible ball back first. The filled
    ball wanders, because a straight line would itself reveal a stoppage.
  - In real tracking the ball may be lost more during stoppages, but only the ground truth could say
    how much. So dead_hide_prob is a neutral 0.5 at test time and anywhere from 0 to 1 in training,
    and scripts/stress_test.py shows the score at both extremes.
  - gap_keep thins the drawn blackouts so the visible share matches the 59.3% of graded frames that
    DATA.md publishes, which is a property of the tracking: 0.7 gives 0.590 on the 10 open matches.
    Stoppage blackouts never line up with the true boundary, or they would give the answer away.

No setting here is fitted to any score. An earlier version fitted two of them to the published
baseline scores, which come from the private ground truth, so that was removed. See git history.
"""

import json
import math
import os
import zlib
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .match import AWAY, D, HOME, Match

STATS_PATH = os.path.join(os.path.dirname(__file__), "degrade_stats.json")


@dataclass(frozen=True)
class DegradeConfig:
    drop_ball: bool = True
    fill_dead_ball: bool = True
    ball_noise_m: float = 0.5        # estimate
    player_noise_m: float = 0.15     # estimate; the real players are smoother than 0.3 gave
    noise_rho: float = 0.98          # frame-to-frame correlation of the noise
    tracklet_mean_s: float = 62.0    # estimate; the real mean track is longer than 60 s
    gap_mean_s: float = 4.0          # estimate; with 62 s keeps about 20.6 of 22 objects, as published
    wrong_team_rate: float = 0.01    # estimate; share of tracklets given to the other team
    dead_hide_prob: float = 0.5      # chance a whole stoppage has no ball; neutral, unknowable
    gap_keep: float = 0.7            # share of drawn blackouts applied; matches DATA.md's visible ball
    dead_edge_early_s: float = 3.0   # a stoppage blackout may start this long before the stoppage,
    dead_edge_late_s: float = 6.0    # or this long after it, and the mirror image at its end
    filled_ball_wander_m: float = 3.0
    ball_false_rate: float = 0.0     # share of visible ball time replaced by false detections
    slots_per_team: int = 23         # 46 slots in the tactical ATD


# Draws are written so every computer gets the same bits. numpy on Apple chips computes
# rng.uniform(low, high) as one fused multiply-add, which can differ in the last bit from a chip that
# rounds twice; math.fma gives the fused result everywhere. Python's math.exp and math.log avoid
# numpy's own versions on some Intel and AMD chips. On the build Mac these give exactly the same
# numbers as before (checked on 1.2 million draws, and by scripts/repro_check.py).
def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return math.fma(high - low, rng.random(), low)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return math.exp(_uniform(rng, math.log(low), math.log(high)))


def random_config(rng: np.random.Generator) -> DegradeConfig:
    """A tracking quality drawn from a wide range, for training.

    A model trained at one noise level falls apart at another, so training covers the range. The
    clean end matters as much as the bad end: scripts/compare_tracking.py showed the real tracking
    is cleaner than an earlier, narrower range, which only measured tracking, never any answers.
    Log-uniform draws give the clean end as many copies as the bad end.
    """
    return DegradeConfig(
        ball_noise_m=_log_uniform(rng, 0.05, 3.0),
        ball_false_rate=_uniform(rng, 0.0, 0.15) if rng.random() < 0.5 else 0.0,
        player_noise_m=_log_uniform(rng, 0.03, 2.5),
        noise_rho=1.0 - _log_uniform(rng, 0.005, 0.5),
        tracklet_mean_s=_log_uniform(rng, 10.0, 120.0),
        gap_mean_s=_uniform(rng, 2.0, 8.0),
        wrong_team_rate=_uniform(rng, 0.0, 0.1),
        gap_keep=_uniform(rng, 0.4, 1.0),
        dead_hide_prob=_uniform(rng, 0.0, 1.0),
    )


def load_stats(path: Optional[str] = None) -> dict:
    with open(path or STATS_PATH) as handle:
        return json.load(handle)


def _rng(match_id: str, seed: int, salt: str) -> np.random.Generator:
    """Stable across runs and machines, unlike Python's hash()."""
    return np.random.default_rng([zlib.crc32(f"{match_id}/{salt}".encode()), seed])


def _ar1(rng, n: int, sd: float, rho: float, dims: int = 2) -> np.ndarray:
    """Correlated noise via a first-order filter, in vectorised blocks."""
    from scipy.signal import lfilter
    white = rng.normal(0.0, sd * np.sqrt(1.0 - rho * rho), (n, dims))  # rho * rho: pow can differ by chip
    return lfilter([1.0], [1.0, -rho], white, axis=0).astype(np.float32)


def ball_mask(n: int, gaps: np.ndarray, runs: np.ndarray, rng, keep: float = 1.0) -> np.ndarray:
    """True where the ball is visible, alternating real-length visible runs and blackouts."""
    mask = np.ones(n, dtype=bool)
    t = int(rng.integers(0, max(int(np.median(runs)), 1)))
    while t < n:
        gap = int(rng.choice(gaps))
        if rng.random() < keep:
            mask[t:t + gap] = False
        t += gap + int(rng.choice(runs))
    return mask


def fill_ball(ball: np.ndarray) -> np.ndarray:
    """Linear interpolation across every gap, used before masking."""
    out = ball.copy()
    idx = np.arange(len(out))
    seen = np.isfinite(out[:, 0])
    if not seen.any():
        return out
    for axis in (0, 1):
        out[:, axis] = np.interp(idx, idx[seen], out[seen, axis])
    return out


def _contiguous(valid: np.ndarray):
    edges = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())


def fragment_players(match: Match, cfg: DegradeConfig, rng):
    """Cut each player into tracklets, leave gaps, and scatter them over anonymous slots."""
    n = match.n_frames
    fps = match.fps
    slots = cfg.slots_per_team
    out = np.full((n, 2 * slots, 2), np.nan, dtype=np.float32)
    free_at = np.zeros(2 * slots, dtype=np.int64)

    pieces = []
    for p in range(match.players.shape[1]):
        for start, stop in _contiguous(np.isfinite(match.players[:, p, 0])):
            t = start
            while t < stop:
                life = max(int(rng.exponential(cfg.tracklet_mean_s * fps)), fps // 5)
                end = min(t + life, stop)
                pieces.append((t, end, p))
                t = end + max(int(rng.exponential(cfg.gap_mean_s * fps)), 1)
    pieces.sort()

    for start, stop, p in pieces:
        side = int(match.team[p])
        if rng.random() < cfg.wrong_team_rate:
            side = 1 - side
        lo = side * slots
        free = np.flatnonzero(free_at[lo:lo + slots] <= start)
        if not len(free):
            continue  # every slot busy: the object is lost, which also happens in real tracking
        slot = lo + int(rng.choice(free))
        out[start:stop, slot] = match.players[start:stop, p]
        if cfg.player_noise_m:
            out[start:stop, slot] += _ar1(rng, stop - start, cfg.player_noise_m, cfg.noise_rho)
        free_at[slot] = stop
    team = np.array([HOME] * slots + [AWAY] * slots, dtype=np.int8)
    return out, team


def degrade(match: Match, cfg: DegradeConfig = DegradeConfig(), seed: int = 0,
            stats: Optional[dict] = None) -> Match:
    """A new match with degraded tracking and untouched labels. Deterministic for a seed."""
    stats = stats or load_stats()
    rng = _rng(match.match_id, seed, "degrade")

    ball = match.ball
    if cfg.fill_dead_ball:
        blank = ~np.isfinite(ball[:, 0])
        ball = fill_ball(ball)
        if blank.any() and cfg.filled_ball_wander_m:
            # A straight interpolated line would be a fingerprint of a dead ball.
            wander = _ar1(rng, len(ball), cfg.filled_ball_wander_m, 0.999)
            ball[blank] += wander[blank]
    else:
        ball = ball.copy()
    if cfg.ball_noise_m:
        ball = ball + _ar1(rng, len(ball), cfg.ball_noise_m, cfg.noise_rho)
    if cfg.ball_false_rate:
        # False detections: a wrong object tracked as the ball for about a second, near some player.
        anyone = np.nan_to_num(np.nanmean(match.players, axis=1))
        t = 0
        while t < len(ball):
            t += int(rng.exponential(match.fps / cfg.ball_false_rate))
            stop = min(t + int(_uniform(rng, 0.5, 2.0) * match.fps), len(ball))
            if t < len(ball):
                ball[t:stop] = anyone[t:stop] + rng.normal(0, 15.0, 2)
            t = stop
    if cfg.drop_ball:
        gaps = np.asarray(stats["ball_gap_frames"])
        runs = np.asarray(stats["ball_run_frames"])
        visible = np.ones(len(ball), dtype=bool)
        for start, stop in match.halves:
            visible[start:stop] = ball_mask(stop - start, gaps, runs, rng, cfg.gap_keep)
        if cfg.dead_hide_prob and match.labels is not None:
            dead = (match.labels == D).astype(np.int8)
            n = len(ball)
            for start, stop in _contiguous(dead):
                if rng.random() < cfg.dead_hide_prob:
                    # Never aligned exactly: a blackout edge on the true frame would hand a model
                    # the boundary, which real tracking does not do.
                    a = start + int(_uniform(rng, -cfg.dead_edge_early_s, cfg.dead_edge_late_s) * match.fps)
                    b = stop + int(_uniform(rng, -cfg.dead_edge_late_s, cfg.dead_edge_early_s) * match.fps)
                    if b > a:
                        visible[max(a, 0):min(b, n)] = False
        ball[~visible] = np.nan

    players, team = fragment_players(match, cfg, rng)
    return match.with_tracking(ball.astype(np.float32), players, team,
                               degrade=asdict(cfg), degrade_seed=seed)
