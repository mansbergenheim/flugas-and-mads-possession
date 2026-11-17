"""A made-up TV broadcast of a labelled match, for testing the aligner where the truth is known.

It imitates what DATA.md says about the real broadcast: its own timeline with time cut out between
the halves, a camera that shows only the part of the pitch around the ball, and close-ups and
replays in which no pitch transform exists. Close-ups and replays mostly sit in stoppages, and the
wide shot is usually back just before play restarts.

The numbers below are guesses about how a director works, not measurements. They are good enough to
test whether two timelines can be lined up. They say nothing about how much a real director's cuts
tell us about stoppages, so no score measured on a fake broadcast should be believed for that.
"""

import zlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .align import LIVE, NOT_WIDE, REPLAY
from .degrade import _ar1, fill_ball
from .match import D, Match


@dataclass(frozen=True)
class CastConfig:
    lead_s: Tuple[float, float] = (0.0, 10.0)    # play missing at the start and end of each half
    pad_s: Tuple[float, float] = (10.0, 60.0)    # studio shots before, between and after the halves
    view_m: Tuple[float, float] = (18.0, 34.0)   # how far to each side of the ball the camera shows
    seen: float = 0.93                           # chance that a player in view is tracked on a frame
    noise_m: float = 0.4                         # error of the broadcast's own tracking
    stoppage_cut: float = 0.75                   # share of stoppages over 5 s that get a close-up
    replay: float = 0.5                          # share of those followed by a replay
    replay_wide: float = 0.4                     # share of replays shown from the wide camera
    back_early: float = 0.8                      # share in which the wide shot is back before the restart
    live_cut_every_s: float = 45.0               # a close-up during play, about this often


@dataclass
class Cast:
    players: np.ndarray      # [broadcast frames, slots, 2] metres, NaN where not seen
    ball: np.ndarray         # [broadcast frames, 2]
    valid: np.ndarray        # [broadcast frames] a pitch transform exists
    team: np.ndarray         # [slots]
    live: np.ndarray         # the match frame happening at each broadcast frame, -1 in studio shots
    state: np.ndarray        # the truth: NOT_WIDE, LIVE or REPLAY


def fake_broadcast(match: Match, seed: int = 0, cfg: CastConfig = CastConfig(), flip: Tuple[int, int] = (1, 1),
                   swap_teams: bool = False) -> Cast:
    """The match as a TV director might show it. flip turns or mirrors the pitch the broadcast draws."""
    if match.labels is None:
        raise ValueError("the fake broadcast needs labels, to know where the stoppages are")
    rng = np.random.default_rng([zlib.crc32(match.match_id.encode()), seed, 77])
    fps, total = match.fps, match.n_frames

    def frames(low_high) -> int:
        return int(rng.uniform(*low_high) * fps)

    parts = []
    for start, stop in match.halves:
        parts += [np.full(frames(cfg.pad_s), -1), np.arange(start + frames(cfg.lead_s), stop - frames(cfg.lead_s))]
    parts.append(np.full(frames(cfg.pad_s), -1))
    live = np.concatenate(parts).astype(np.int64)
    n = len(live)
    where = np.full(total + 1, -1, dtype=np.int64)          # broadcast frame of each match frame
    where[live[live >= 0]] = np.flatnonzero(live >= 0)
    state = np.where(live >= 0, LIVE, NOT_WIDE).astype(np.uint8)
    source = live.copy()                                    # the match frame on screen, -1 when it is no wide shot

    def show(first: int, last: int, kind: int, replay_from: int = 0) -> None:
        on_air = where[max(first, 0):max(last, 0)]
        on_air = on_air[on_air >= 0]
        state[on_air] = kind
        source[on_air] = np.clip(replay_from + np.arange(len(on_air)) // 2, 0, total - 1) if kind == REPLAY else -1

    dead = np.concatenate(([False], match.labels == D, [False]))
    edges = np.flatnonzero(dead[1:] != dead[:-1])
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < 5 * fps or rng.random() >= cfg.stoppage_cut:
            continue
        back = e - frames((0.5, 2.0)) if rng.random() < cfg.back_early else e + frames((0.5, 3.0))
        first = s + frames((1.0, 2.5))
        last = min(first + frames((2.0, 6.0)), back)
        show(first, last, NOT_WIDE)
        if rng.random() < cfg.replay:
            length, wide, before = frames((4.0, 9.0)), rng.random() < cfg.replay_wide, frames((4.0, 9.0))
            show(last, min(last + length, back), REPLAY if wide else NOT_WIDE, s - before)
    t = int(rng.exponential(cfg.live_cut_every_s) * fps)
    while t < n:
        length = frames((1.5, 4.0))
        cut = np.arange(t, min(t + length, n))
        cut = cut[state[cut] == LIVE]
        state[cut], source[cut] = NOT_WIDE, -1
        t += length + int(rng.exponential(cfg.live_cut_every_s) * fps)

    wide = np.flatnonzero(source >= 0)
    shown = source[wide]
    ball = fill_ball(match.ball)
    centre = np.convolve(np.nan_to_num(ball[:, 0]), np.ones(15) / 15.0, mode="same")
    low, high = cfg.view_m
    reach = (low + high) / 2 + (high - low) / 2 * np.sin(2 * np.pi * wide / (fps * 37.0) + rng.uniform(0, 6.28))
    slots = match.players.shape[1]
    there = match.players[shown]
    in_view = np.abs(there[:, :, 0] - centre[shown][:, None]) < reach[:, None]
    tracked = in_view & (rng.random(in_view.shape) < cfg.seen)
    there = there + _ar1(rng, len(wide), cfg.noise_m, 0.98, dims=slots * 2).reshape(len(wide), slots, 2)
    players = np.full((n, slots, 2), np.nan, dtype=np.float32)
    players[wide] = np.where(tracked[:, :, None], there, np.nan) * np.array(flip, dtype=np.float32)

    ball_out = np.full((n, 2), np.nan, dtype=np.float32)
    ball_seen = (np.abs(ball[shown, 0] - centre[shown]) < reach) & (rng.random(len(wide)) < 0.65)
    ball_out[wide[ball_seen]] = (ball[shown[ball_seen]] + rng.normal(0, 0.3, (int(ball_seen.sum()), 2))) \
        * np.array(flip, dtype=np.float32)
    team = (1 - match.team if swap_teams else match.team).astype(np.int8)
    return Cast(players, ball_out, source >= 0, team, live.astype(np.int32), state)
