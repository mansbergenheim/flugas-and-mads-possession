"""Per-sample features, built to survive long ball blackouts and noisy positions.

Samples are taken every STEP frames (5 Hz). Labels change far less often than that, and a block of
5 frames costs at most 2-3 frames of boundary error, which the metric treats as nearly free.

Four kinds of evidence:
  - Ball, when present: who is near it, how fast it moves, is it off the pitch.
  - Touches, as a sequence: which team played the ball last, before that, and next. A short
    possession is a run of one team's contacts between two runs of the other, and no average over
    a window can show that. Without this the model missed most possessions under 4 s even with the
    ball in plain view.
  - Teams, always present: shape, depth and movement. In a blackout this is what is left. The team
    with the ball stretches and moves forward; the other compacts and tracks back.
  - Memory: who was near the ball when it vanished, and who is near it when it returns. If both
    are the same team, the possession probably continued through the gap.

x is flipped per half so the home team always attacks +x. Everything is computed inside one half.
There are no player identities here, only team membership, because the competition has none.
"""

import warnings
from typing import List, Tuple

import numpy as np

from .match import A, H, Match

STEP = 5
MAX_PLAYER_SPEED = 12.0
WINDOWS_S = (1, 3, 8, 20)
SPEED_SPAN = 3  # samples either side, so speed is measured across 1.2 s
CONTACT_M = 3.0        # a player this close to a slow ball could be playing it
FLIGHT_MPS = 9.0       # above this the ball is travelling, and nearest player says little
MIN_RUN_SAMPLES = 2    # a contact run shorter than this is treated as noise


def attack_sign(match: Match, start: int, stop: int) -> float:
    """+1 if the home team attacks +x in this half, inferred from where each team stands."""
    sample = match.players[start:stop:25, :, 0]
    home = np.nanmean(sample[:, match.team == 0])
    away = np.nanmean(sample[:, match.team == 1])
    return 1.0 if home < away else -1.0


def _length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """sqrt(x^2 + y^2), worked in float64 and rounded back to the inputs' type.

    Used instead of np.hypot, which leaves the rounding to each system's maths library. Square root,
    multiply and add round the same way on every chip. On the build Mac this gives exactly the same
    bits as np.hypot on float32 (checked on 500 million values).
    """
    kind = np.result_type(x, y)
    x64, y64 = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.sqrt(x64 * x64 + y64 * y64).astype(kind)


def _rolling(x: np.ndarray, before: int, after: int) -> np.ndarray:
    """NaN-aware mean over [t-before, t+after]. NaN where the window holds no value."""
    ok = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(x))
    lo = np.maximum(index - before, 0)
    hi = np.minimum(index + after + 1, len(x))
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[hi] - total[lo]) / n, np.nan)


def _fill_forward(x: np.ndarray) -> np.ndarray:
    index = np.where(np.isfinite(x), np.arange(len(x)), -1)
    np.maximum.accumulate(index, out=index)
    return np.where(index >= 0, x[np.maximum(index, 0)], np.nan)


def _fill_backward(x: np.ndarray) -> np.ndarray:
    return _fill_forward(x[::-1])[::-1]


def _since(flag: np.ndarray) -> np.ndarray:
    """Frames since flag was last True. Large before the first True."""
    index = np.where(flag, np.arange(len(flag)), -len(flag))
    return np.arange(len(flag)) - np.maximum.accumulate(index)


def _min_ignore_nan(x: np.ndarray, axis: int) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, np.inf)
    out = filled.min(axis=axis)
    return np.where(np.isfinite(out), out, np.nan)


def _rank(x: np.ndarray) -> np.ndarray:
    """Position of each value within this half, 0 to 1. NaN stays NaN."""
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n > 1:
        order = np.argsort(x[ok], kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(n) / (n - 1)
        out[ok] = ranks
    return out


def _block_mean(x: np.ndarray, step: int, n_blocks: int) -> np.ndarray:
    """NaN-aware mean over consecutive blocks of `step` frames along axis 0."""
    pad = n_blocks * step - len(x)
    if pad > 0:
        x = np.concatenate((x, np.full((pad,) + x.shape[1:], np.nan, dtype=x.dtype)))
    blocks = x[:n_blocks * step].reshape((n_blocks, step) + x.shape[1:])
    ok = np.isfinite(blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(ok, blocks, 0.0).sum(1) / ok.sum(1)).astype(np.float32)


def _shift(x: np.ndarray, by: int) -> np.ndarray:
    """x moved in time by `by` samples, NaN past the ends."""
    out = np.full_like(x, np.nan)
    if by > 0:
        out[:-by] = x[by:]
    else:
        out[-by:] = x[:by]
    return out


def jitter_level(xy: np.ndarray, stride: int = 5) -> float:
    """Typical frame-to-frame wobble in metres, from second differences of each track.

    Reported by scripts/atd_summary.py. Deliberately not a model feature: a per-match constant lets
    trees memorise matches rather than learn from them.
    """
    second = xy[2::stride] - 2 * xy[1:-1:stride][: len(xy[2::stride])] + xy[:-2:stride][: len(xy[2::stride])]
    size = _length(second[..., 0], second[..., 1])
    size = size[np.isfinite(size)]
    return float(np.median(size)) if size.size else 0.0


def _pick(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """values[index], NaN where index falls outside."""
    ok = (index >= 0) & (index < len(values))
    out = np.full(len(index), np.nan, dtype=np.float32)
    out[ok] = values[index[ok]]
    return out


def contact_runs(vote: np.ndarray, near: np.ndarray, speed: np.ndarray):
    """Who was playing the ball, as (first sample, last sample, mean vote, samples) per run.

    A run is an unbroken sequence of contacts by one team, however long the ball was in flight or
    out of sight between them. One or two stray samples in a duel don't start a new run.
    """
    contact = np.isfinite(vote) & (near < CONTACT_M) & ~(speed > FLIGHT_MPS)
    index = np.flatnonzero(contact)
    if not len(index):
        return tuple(np.zeros(0) for _ in range(4))
    side = np.sign(vote[index])
    for _ in range(2):
        cut = np.flatnonzero(side[1:] != side[:-1]) + 1
        first = np.concatenate(([0], cut))
        length = np.diff(np.concatenate((first, [len(side)])))
        keep = np.repeat(length >= MIN_RUN_SAMPLES, length)
        index, side = index[keep], side[keep]
        if not len(index):
            return tuple(np.zeros(0) for _ in range(4))
    cut = np.flatnonzero(side[1:] != side[:-1]) + 1
    first = np.concatenate(([0], cut))
    last = np.concatenate((cut - 1, [len(side) - 1]))
    votes = np.add.reduceat(vote[index].astype(np.float64), first) / (last - first + 1)
    return index[first], index[last], votes, (last - first + 1).astype(np.float64)


def _contact_features(vote, near, speed, dt) -> dict:
    n = len(vote)
    start, stop, votes, size = contact_runs(vote, near, speed)
    now = np.arange(n)
    k = np.searchsorted(start, now, side="right") - 1
    f = {"run_inside": ((k >= 0) & (now <= _pick(stop, k))).astype(np.float32)}
    for name, shift in (("prev2", -2), ("prev", -1), ("cur", 0), ("next", 1), ("next2", 2)):
        f[f"run_{name}_vote"] = _pick(votes, k + shift)
        f[f"run_{name}_n"] = _pick(size, k + shift)
    f["run_since_s"] = (now - _pick(stop, k)) * dt
    f["run_age_s"] = (now - _pick(start, k)) * dt
    f["run_until_s"] = (_pick(start, k + 1) - now) * dt
    return f


def _team_block(prefix: str, xy, vel, speed, pitch) -> dict:
    x, y = xy[..., 0], xy[..., 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        length = pitch[0] / 2
        f = {
            "n": np.isfinite(x).sum(1).astype(np.float32),
            "cx": np.nanmean(x, 1), "cy": np.nanmean(y, 1),
            "sx": np.nanstd(x, 1), "sy": np.nanstd(y, 1),
            "minx": np.nanmin(x, 1), "maxx": np.nanmax(x, 1),
            "q25x": np.nanpercentile(x, 25, axis=1), "q75x": np.nanpercentile(x, 75, axis=1),
            "speed": np.nanmean(speed, 1), "maxspeed": np.nanmax(speed, 1),
            "vx": np.nanmean(vel[..., 0], 1),
            "slow": (speed < 1.0).sum(1).astype(np.float32),
            "fast": (speed > 4.0).sum(1).astype(np.float32),
            "in_att_third": (x > length / 3).sum(1).astype(np.float32),
            "in_def_third": (x < -length / 3).sum(1).astype(np.float32),
        }
    return {f"{prefix}_{k}": v for k, v in f.items()}


# Extra inputs for the v6 candidate. Off by default, and with them off nothing above or below changes.
EXTRA_OFFSETS = (("vote", (1, 2, 3, 5, 8)), ("near", (1, 2, 3, 5, 8)), ("ball_speed", (1, 2, 3, 5, 8)),
                 ("dist_a", (1, 2, 3)), ("dist_b", (1, 2, 3)), ("seen", (1, 3, 8)))
TOUCH_OFFSETS = (1, 2, 3, 5)


def _extra_features(match: Match, start: int, stop: int, step: int, frames: np.ndarray,
                    now: np.ndarray, ball: np.ndarray, f: dict) -> dict:
    """The shape of the evidence over time, relative motion, and touches at the full frame rate.

    The windowed averages above say how much evidence there was, not when. A short possession is a
    sharp bump, so the model also gets the evidence a little earlier and later. It gets whether the
    ball moves with each team's nearest player (a carry) or past him, and the closest approach inside
    each block at 25 Hz, which block-averaged positions blur.

    Only +, -, *, /, sqrt, min, max and sort are used, so every chip rounds the same way.
    """
    fps = match.fps
    dt = step / fps
    home, away = match.team == 0, match.team == 1
    has_ball = np.isfinite(ball[:, 0])
    out = {}
    for name, offsets in EXTRA_OFFSETS:
        base = np.asarray(f[name], dtype=np.float32)
        for k in offsets:
            out[f"{name}_m{k}"] = _shift(base, -k)
            out[f"{name}_p{k}"] = _shift(base, k)

    vel_p = (_shift(now, 1) - _shift(now, -1)) / (2 * dt)
    vel_b = (_shift(ball, 1) - _shift(ball, -1)) / (2 * dt)
    delta = now - ball[:, None, :]
    dist = _length(delta[..., 0], delta[..., 1])
    rows = np.arange(len(frames))
    for tag, mask in (("a", home), ("b", away)):
        d = np.where(np.isfinite(dist[:, mask]), dist[:, mask], np.inf)
        nearest = d.argmin(1)
        vp = vel_p[:, mask][rows, nearest]
        rel = _length(vp[:, 0] - vel_b[:, 0], vp[:, 1] - vel_b[:, 1])
        rel = np.where(np.isfinite(d[rows, nearest]) & has_ball, rel, np.nan).astype(np.float32)
        out[f"rel_{tag}"] = rel
        out[f"rel_{tag}_s"] = _rolling(rel.astype(np.float64), 2, 2)
        second = np.sort(d, axis=1)[:, 1] if d.shape[1] > 1 else np.full(len(d), np.inf)
        out[f"dist2_{tag}"] = np.where(np.isfinite(second) & has_ball, second, np.nan)

    # Frame-rate evidence inside each block.
    n, blocks = stop - start, len(frames)
    players_25, ball_25 = match.players[start:stop], match.ball[start:stop]
    mind = np.full((n, 2), np.inf, dtype=np.float32)
    for lo in range(0, n, 20000):
        hi = min(lo + 20000, n)
        gap = players_25[lo:hi] - ball_25[lo:hi, None, :]
        size = _length(gap[..., 0], gap[..., 1])
        size = np.where(np.isfinite(size), size, np.inf)
        mind[lo:hi, 0] = size[:, home].min(1) if home.any() else np.inf
        mind[lo:hi, 1] = size[:, away].min(1) if away.any() else np.inf
    step_v = np.diff(ball_25, axis=0, prepend=ball_25[:1]) * fps
    speed_25 = _length(step_v[:, 0], step_v[:, 1])
    pad = blocks * step - n
    if pad > 0:
        mind = np.concatenate((mind, np.full((pad, 2), np.inf, dtype=np.float32)))
        speed_25 = np.concatenate((speed_25, np.full(pad, np.nan, dtype=speed_25.dtype)))
    mind = mind[:blocks * step].reshape(blocks, step, 2)
    speed_25 = speed_25[:blocks * step].reshape(blocks, step)

    closest = mind.min(1)
    closest = np.where(np.isfinite(closest), closest, np.nan)
    both = np.isfinite(mind[..., 0]) & np.isfinite(mind[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        home_nearer = (both & (mind[..., 1] > mind[..., 0])).sum(1) / both.sum(1)
    known = np.isfinite(speed_25)
    fastest = np.where(known, speed_25, -np.inf).max(1)
    slowest = np.where(known, speed_25, np.inf).min(1)
    out["ball_fast25"] = np.where(np.isfinite(fastest), fastest, np.nan)
    out["ball_slow25"] = np.where(np.isfinite(slowest), slowest, np.nan)
    out["touch_a"], out["touch_b"] = closest[:, 0], closest[:, 1]
    touch_vote = np.clip(closest[:, 1] - closest[:, 0], -6.0, 6.0)
    out["touch_vote"] = touch_vote
    out["near_frac_a"] = home_nearer
    for k in TOUCH_OFFSETS:
        out[f"touch_vote_m{k}"] = _shift(touch_vote, -k)
        out[f"touch_vote_p{k}"] = _shift(touch_vote, k)

    # Does the ball change direction here? Cosine between its velocity just after and just before.
    after, before = _shift(vel_b, 1).astype(np.float64), _shift(vel_b, -1).astype(np.float64)
    dot = after[:, 0] * before[:, 0] + after[:, 1] * before[:, 1]
    size = (np.sqrt(after[:, 0] * after[:, 0] + after[:, 1] * after[:, 1])
            * np.sqrt(before[:, 0] * before[:, 0] + before[:, 1] * before[:, 1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ball_turn"] = np.where(size > 1.0, dot / size, np.nan)
    return out


def _half_features(match: Match, start: int, stop: int, step: int,
                   extra: bool = False) -> Tuple[dict, np.ndarray]:
    fps = match.fps
    dt = step / fps
    frames = np.arange(start + step // 2, stop, step)
    scale = np.array([attack_sign(match, start, stop), 1.0], dtype=np.float32)

    # Positions are averaged over each block and speeds measured across a full second. Measured
    # frame to frame, a metre of tracking jitter looks like a sprint and hides every stoppage.
    now = _block_mean(match.players[start:stop], step, len(frames)) * scale
    ahead, behind = _shift(now, SPEED_SPAN), _shift(now, -SPEED_SPAN)
    vel = (ahead - behind) / (2 * SPEED_SPAN * dt)
    speed = _length(vel[..., 0], vel[..., 1])
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    vel[~np.isfinite(speed)] = np.nan

    home, away = match.team == 0, match.team == 1
    f = {}
    f.update(_team_block("a", now[:, home], vel[:, home], speed[:, home], match.pitch))
    f.update(_team_block("b", now[:, away], vel[:, away], speed[:, away], match.pitch))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f["all_speed"] = np.nanmean(speed, 1)
    f["all_slow"] = f["a_slow"] + f["b_slow"]
    f["cx_diff"] = f["a_cx"] - f["b_cx"]
    f["vx_diff"] = f["a_vx"] + f["b_vx"]  # both teams move up the pitch together with the ball
    f["sx_diff"] = f["a_sx"] - f["b_sx"]
    f["sy_diff"] = f["a_sy"] - f["b_sy"]
    f["speed_diff"] = f["a_speed"] - f["b_speed"]

    ball = _block_mean(match.ball[start:stop], step, len(frames)) * scale
    f["ball_x"], f["ball_y"] = ball[:, 0], np.abs(ball[:, 1])
    f["ball_speed"] = _length(*(ball - _shift(ball, -1)).T) / dt

    delta = now - ball[:, None, :]
    dist = _length(delta[..., 0], delta[..., 1])
    has_ball = np.isfinite(ball[:, 0])
    f["dist_a"], f["dist_b"] = _min_ignore_nan(dist[:, home], 1), _min_ignore_nan(dist[:, away], 1)
    v = np.clip(f["dist_b"] - f["dist_a"], -6.0, 6.0)
    f["vote"] = v
    f["near_a"] = np.where(has_ball, (dist[:, home] < 5.0).sum(1), np.nan)
    f["near_b"] = np.where(has_ball, (dist[:, away] < 5.0).sum(1), np.nan)
    half_l, half_w = match.pitch[0] / 2, match.pitch[1] / 2
    f["off_pitch"] = np.where(has_ball, (np.abs(ball[:, 0]) > half_l + 0.5) |
                              (np.abs(ball[:, 1]) > half_w + 0.5), np.nan)

    seen = np.isfinite(match.ball[start:stop, 0])
    offsets = frames - start
    f["seen"] = seen[offsets].astype(np.float32)
    f["since_seen_s"] = np.minimum(_since(seen)[offsets] / fps, 120.0)
    f["until_seen_s"] = np.minimum(_since(seen[::-1])[::-1][offsets] / fps, 120.0)

    # Memory across gaps. Votes are smoothed over 1 s first, so one stray frame can't set the memory.
    smooth = _rolling(v, 2, 2)
    smooth = np.where(has_ball, smooth, np.nan)
    f["last_vote"], f["next_vote"] = _fill_forward(smooth), _fill_backward(smooth)
    f["vote_gap_diff"] = f["next_vote"] - f["last_vote"]
    f["last_ball_x"], f["next_ball_x"] = _fill_forward(ball[:, 0]), _fill_backward(ball[:, 0])

    per_s = int(round(1 / dt))
    # Short windows, because a possession can last two seconds.
    travel = (_shift(ball, 1) - _shift(ball, -1)) / (2 * dt)
    f["ball_vx"] = _rolling(travel[:, 0].astype(np.float64), 2, 2)
    f["ball_speed_s"] = _rolling(f["ball_speed"].astype(np.float64), 1, 1)
    f["vote_s"] = smooth
    f["vote_past1"], f["vote_next1"] = _rolling(v, per_s, 0), _rolling(v, 0, per_s)
    close = np.where(has_ball, (f["dist_a"] < 2.0).astype(np.float64) - (f["dist_b"] < 2.0), np.nan)
    f["close_s"] = _rolling(close, 2, 2)
    f["close_c1"] = _rolling(close, per_s, per_s)
    f["near"] = np.fmin(f["dist_a"], f["dist_b"])
    f.update(_contact_features(np.where(has_ball, v, np.nan), f["near"], f["ball_speed_s"], dt))

    for name in ("vote", "seen", "all_speed", "all_slow", "vx_diff", "cx_diff", "ball_speed",
                 "sx_diff", "speed_diff"):
        base = f[name].astype(np.float64)
        for w in WINDOWS_S:
            f[f"{name}_c{w}"] = _rolling(base, w * per_s, w * per_s)
    for name in ("vote", "vx_diff", "all_speed"):
        base = f[name].astype(np.float64)
        f[f"{name}_past3"] = _rolling(base, 3 * per_s, 0)
        f[f"{name}_next3"] = _rolling(base, 0, 3 * per_s)

    # How slow is this moment compared with the rest of the half? A rank doesn't change with the
    # noise level, so stoppages stay visible however jittery the tracking is.
    for name in ("all_speed_c1", "all_speed_c3", "all_speed_c8", "all_speed_past3", "all_speed_next3"):
        f[f"{name}_rank"] = _rank(f[name])
    for name in ("a_maxspeed", "b_maxspeed", "a_speed", "b_speed"):
        f[f"{name}_rank"] = _rank(_rolling(f[name].astype(np.float64), 2 * per_s, 2 * per_s))
    if extra:
        f.update(_extra_features(match, start, stop, step, frames, now, ball, f))
    return f, frames


def build_features(match: Match, step: int = STEP,
                   extra: bool = False) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """(values [n, k] float32, names, frames [n], half index [n]). extra adds the v6 candidate's inputs."""
    blocks, frames, halves = [], [], []
    names = None
    for half, (start, stop) in enumerate(match.halves):
        f, fr = _half_features(match, start, stop, step, extra)
        names = names or sorted(f)
        blocks.append(np.column_stack([np.asarray(f[k], dtype=np.float32) for k in names]))
        frames.append(fr)
        halves.append(np.full(len(fr), half, dtype=np.int8))
    return np.vstack(blocks), names, np.concatenate(frames), np.concatenate(halves)


def swap_teams(match: Match) -> Match:
    """The same match with home and away exchanged, so a model can't learn a home bias."""
    from dataclasses import replace
    labels = None
    if match.labels is not None:
        labels = match.labels.copy()
        labels[match.labels == H] = A
        labels[match.labels == A] = H
    return replace(match, team=(1 - match.team).astype(np.int8), labels=labels)
