"""Line the broadcast recording up with the tactical one, using only the two sets of tracking.

The broadcast has its own frame count, replays, close-ups and time cut out. While it shows the live
wide view, its tracking places the players it sees on the pitch, and the tactical tracking places
everybody on every frame. A broadcast frame therefore belongs to the tactical frame whose players
stand where the broadcast's stand. No video and no sound is read.

How it works:
  1. Both recordings are sampled five times a second. For each tactical sample a coarse map of the
     pitch holds the distance from every 2 m cell to the nearest player.
  2. A few hundred short clips of the broadcast (four samples, a second apart) are each scored
     against every tactical moment by looking their players up in those maps. Each clip votes for
     the shift between the two timelines that fits it best.
  3. The broadcast runs in real time between cuts, so true shifts repeat and wrong ones do not.
     Shifts that many clips agree on become pieces of the timeline, normally one for each half.
  4. Inside each piece the shift is refined to the frame with exact distances, two minutes at a time.
  5. Every broadcast frame is then one of: not a wide shot (it has no pitch transform), the live wide
     shot (its players are where the tactical ones are), a wide shot that is not live (a replay), or
     a wide shot with too few players in it to tell.

Everything is deterministic and uses numpy only. Positions are metres from the centre spot, as in
Match.players, with NaN where a slot is empty.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

NOT_WIDE, LIVE, REPLAY, NOT_COVERED, UNSURE = 0, 1, 2, 3, 4
NO_SHIFT = int(np.iinfo(np.int32).min)
FLIPS = ((1, 1), (-1, -1), (-1, 1), (1, -1))    # the same way round first, then turned, then mirrored


@dataclass(frozen=True)
class AlignConfig:
    step: int = 5               # frames between samples
    cell_m: float = 2.0         # size of a cell in the coarse maps
    margin_m: float = 4.0       # the maps reach this far outside the lines
    cap_m: float = 15.0         # no player counts as further away than this
    clip: int = 4               # samples in a clip
    clip_gap: int = 5           # samples between them: one second
    min_players: int = 5        # players a broadcast sample needs to be used in a clip
    anchors: int = 400          # clips spread over the broadcast
    flip_anchors: int = 60      # clips used to find which way round the broadcast draws the pitch
    vote_tol: int = 2           # samples: clips whose shifts differ by no more than this agree
    min_votes: int = 6          # clips that must agree before a shift is believed
    reach: int = 15             # frames either side of the coarse shift tried in the fine search
    window: int = 3000          # broadcast frames given one fine shift: two minutes
    window_frames: int = 150    # frames of a window used to find it
    min_window_frames: int = 20
    fit_m: float = 2.5          # a frame is live when its players are this close to tactical ones
    trim: float = 0.25          # share of a frame's worst-placed players left out
    smooth: int = 12            # frames either side that vote on whether a frame is live
    carry: int = 250            # frames a verdict is carried into a stretch with too few players to tell


class NoAlignment(ValueError):
    """The two recordings could not be lined up. The message says what was missing, in words."""


@dataclass
class Alignment:
    shift: np.ndarray                      # per broadcast frame: tactical frame = frame + shift, or NO_SHIFT
    state: np.ndarray                      # per broadcast frame: NOT_WIDE, LIVE, REPLAY or UNSURE
    pieces: List[Tuple[int, int, int]]     # (first broadcast frame, frame after the last, typical shift)
    flip: Tuple[int, int]                  # multiply the broadcast's x and y by this to match the tactical pitch
    clips: int                             # clips that voted
    agreed: int                            # of those, how many agreed with a piece
    fit_m: float                           # typical distance between matched players on live frames
    same_teams: Optional[bool]             # the two files give a team the same number; None when unclear
    matrix: Optional[np.ndarray] = None    # 2 x 3: what was done to flipped broadcast positions to undo a skewed pitch


def _trimmed_mean(d: np.ndarray, trim: float) -> np.ndarray:
    """Row means over the best-placed players, leaving out the worst share and the missing. NaN for empty rows."""
    s = np.sort(d, axis=1)                                   # missing players sort to the end
    n = np.isfinite(s).sum(axis=1)
    keep = np.maximum(np.ceil(n * (1.0 - trim)).astype(np.int64), 1)
    total = np.cumsum(np.nan_to_num(s, nan=0.0), axis=1)
    mean = total[np.arange(len(s)), np.minimum(keep, s.shape[1]) - 1] / keep
    return np.where(n > 0, mean, np.nan)


def fit_distance(broadcast: np.ndarray, tactical: np.ndarray, cfg: AlignConfig = AlignConfig()) -> np.ndarray:
    """For each pair of frames, how far the broadcast's players stand from the nearest tactical player, in metres."""
    out = np.full(len(broadcast), np.nan, dtype=np.float32)
    for a in range(0, len(broadcast), 1000):
        b, t = broadcast[a:a + 1000], tactical[a:a + 1000]
        gap = b[:, :, None, :] - t[:, None, :, :]
        d = np.sqrt((gap * gap).sum(axis=-1))
        d = np.where(np.isfinite(d), d, np.inf).min(axis=2)
        d = np.where(np.isfinite(b[:, :, 0]), np.minimum(d, cfg.cap_m), np.nan)
        out[a:a + 1000] = _trimmed_mean(d, cfg.trim)
    return out


def _grid(pitch: Sequence[float], cfg: AlignConfig) -> Tuple[int, int]:
    return (int(math.ceil((pitch[0] + 2 * cfg.margin_m) / cfg.cell_m)),
            int(math.ceil((pitch[1] + 2 * cfg.margin_m) / cfg.cell_m)))


def _cells(points: np.ndarray, pitch: Sequence[float], cfg: AlignConfig) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny = _grid(pitch, cfg)
    ix = np.floor((points[:, 0] + pitch[0] / 2 + cfg.margin_m) / cfg.cell_m).astype(np.int64)
    iy = np.floor((points[:, 1] + pitch[1] / 2 + cfg.margin_m) / cfg.cell_m).astype(np.int64)
    return np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)


def distance_maps(players: np.ndarray, pitch: Sequence[float], cfg: AlignConfig = AlignConfig()) -> np.ndarray:
    """[samples, cells along, cells across] uint8: tenths of a metre from each cell to the nearest player, capped."""
    nx, ny = _grid(pitch, cfg)
    xs = ((np.arange(nx) + 0.5) * cfg.cell_m - pitch[0] / 2 - cfg.margin_m).astype(np.float32)
    ys = ((np.arange(ny) + 0.5) * cfg.cell_m - pitch[1] / 2 - cfg.margin_m).astype(np.float32)
    cx, cy = np.repeat(xs, ny), np.tile(ys, nx)
    out = np.empty((len(players), nx * ny), dtype=np.uint8)
    for a in range(0, len(players), 64):
        p = np.nan_to_num(players[a:a + 64].astype(np.float32), nan=1e6)
        dx = p[:, None, :, 0] - cx[None, :, None]
        dy = p[:, None, :, 1] - cy[None, :, None]
        d = np.sqrt((dx * dx + dy * dy).min(axis=2))
        out[a:a + 64] = np.rint(np.minimum(d, cfg.cap_m) * 10.0).astype(np.uint8)
    return out.reshape(len(players), nx, ny)


def _anchors(usable: np.ndarray, centre: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    """First samples of the clips: usable throughout, spread over the recording, from moving play where possible."""
    span = (cfg.clip - 1) * cfg.clip_gap
    if len(usable) <= span:
        return np.zeros(0, dtype=np.int64)
    ok = usable[:len(usable) - span].copy()
    for m in range(1, cfg.clip):
        ok &= usable[m * cfg.clip_gap:m * cfg.clip_gap + len(ok)]
    starts = np.flatnonzero(ok)
    if not len(starts):
        return starts
    motion = np.abs(centre[starts + span] - centre[starts]).sum(axis=1)
    bins = (starts * cfg.anchors) // len(usable)
    chosen = []
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        # The upper quartile, not the largest: the largest jumps are cuts between two shots.
        target = np.quantile(motion[members], 0.75)
        chosen.append(starts[members[np.abs(motion[members] - target).argmin()]])
    return np.array(chosen, dtype=np.int64)


def _coarse(maps: np.ndarray, samples: np.ndarray, starts: np.ndarray, pitch: Sequence[float],
            cfg: AlignConfig) -> np.ndarray:
    """The tactical sample each clip fits best."""
    n = maps.shape[0] - (cfg.clip - 1) * cfg.clip_gap
    best = np.full(len(starts), -1, dtype=np.int64)
    if n <= 0:
        return best
    for k, i in enumerate(starts):
        total = np.zeros(n, dtype=np.float32)
        for m in range(cfg.clip):
            p = samples[i + m * cfg.clip_gap]
            ix, iy = _cells(p[np.isfinite(p[:, 0])], pitch, cfg)
            d = maps[m * cfg.clip_gap:m * cfg.clip_gap + n][:, ix, iy].astype(np.float32)
            d.sort(axis=1)
            total += d[:, :max(int(math.ceil(d.shape[1] * (1.0 - cfg.trim))), 1)].mean(axis=1)
        best[k] = int(total.argmin())
    return best


def _vote(shifts: np.ndarray, cfg: AlignConfig) -> List[np.ndarray]:
    """Groups of clips whose shifts agree, the biggest group first."""
    left = np.ones(len(shifts), dtype=bool)
    groups = []
    while left.sum() >= cfg.min_votes:
        values, counts = np.unique(shifts[left], return_counts=True)
        near = np.array([counts[np.abs(values - v) <= cfg.vote_tol].sum() for v in values])
        top = values[near.argmax()]
        members = np.flatnonzero(left & (np.abs(shifts - top) <= cfg.vote_tol))
        if len(members) < cfg.min_votes:
            break
        groups.append(members)
        left[members] = False
    return groups


def _runs(starts: np.ndarray, groups: List[np.ndarray]) -> List[List[int]]:
    """[group, first clip sample, last clip sample] for each stretch of the broadcast that one group explains."""
    label = np.full(len(starts), -1, dtype=np.int64)
    for g, members in enumerate(groups):
        label[members] = g
    runs: List[List[int]] = []
    for k in np.argsort(starts):
        if label[k] < 0:
            continue
        if runs and runs[-1][0] == label[k]:
            runs[-1][2] = int(starts[k])
            runs[-1][3] += 1
        else:
            runs.append([int(label[k]), int(starts[k]), int(starts[k]), 1])
    runs = [r for r in runs if r[3] >= 3]                    # two clips agreeing by chance are not a piece
    merged: List[List[int]] = []
    for r in runs:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(r[:3])
    return merged


def _fine(broadcast: np.ndarray, tactical: np.ndarray, usable: np.ndarray, first: int, last: int, coarse: int,
          cfg: AlignConfig) -> Optional[np.ndarray]:
    """For each shift near the coarse one, how well it fits the usable broadcast frames from first to last (metres)."""
    frames = np.flatnonzero(usable[first:last]) + first
    if len(frames) < cfg.min_window_frames:
        return None
    frames = frames[np.linspace(0, len(frames) - 1, min(cfg.window_frames, len(frames))).astype(np.int64)]
    deltas = np.arange(-cfg.reach, cfg.reach + 1)
    target = frames[:, None] + coarse + deltas[None, :]
    inside = (target >= 0) & (target < len(tactical))
    d = fit_distance(np.repeat(broadcast[frames], len(deltas), axis=0),
                     tactical[np.clip(target, 0, len(tactical) - 1).ravel()], cfg).reshape(target.shape)
    d = np.where(inside, d, np.nan)
    if not np.isfinite(d).any(axis=0).all():
        return None
    return np.nanmean(d, axis=0)


def _pairs(broadcast: np.ndarray, tactical: np.ndarray, reach_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Each broadcast player with the tactical player nearest to it, where that one is within reach."""
    gap = broadcast[:, :, None, :] - tactical[:, None, :, :]
    d = np.where(np.isfinite(gap).all(axis=-1), (gap * gap).sum(axis=-1), np.inf)
    nearest = d.argmin(axis=2)
    close = d.min(axis=2) < reach_m * reach_m
    frame, slot = np.nonzero(close)
    return broadcast[frame, slot].astype(np.float64), tactical[frame, nearest[frame, slot]].astype(np.float64)


def _affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """The 2 x 3 matrix that moves src onto dst as well as a stretch, turn and slide can, or None if it is not small."""
    keep = np.ones(len(src), dtype=bool)
    matrix = None
    for limit in (None, 3.0, 2.0):
        if matrix is not None:
            keep = np.sqrt((((src @ matrix[:, :2].T + matrix[:, 2]) - dst) ** 2).sum(axis=1)) < limit
        if keep.sum() < 200:
            return None
        solution = np.linalg.lstsq(np.column_stack((src[keep], np.ones(int(keep.sum())))), dst[keep], rcond=None)[0]
        matrix = np.round(solution.T, 4)                    # rounded so that every computer gets the same matrix
    if np.abs(matrix[:, :2] - np.eye(2)).max() > 0.15 or np.abs(matrix[:, 2]).max() > 8.0:
        return None                                         # not a small correction: something else is wrong
    return matrix


def align(tactical: np.ndarray, broadcast: np.ndarray, valid: np.ndarray, pitch: Sequence[float],
          tactical_team: Optional[np.ndarray] = None, broadcast_team: Optional[np.ndarray] = None,
          cfg: AlignConfig = AlignConfig()) -> Alignment:
    """Where each broadcast frame sits on the tactical timeline, and what it shows.

    tactical [frames, slots, 2] and broadcast [frames, slots, 2] in metres, valid [frames] true where
    the broadcast frame has a pitch transform. Raises NoAlignment when the recordings do not line up.
    """
    valid = np.asarray(valid, dtype=bool)
    if len(valid) != len(broadcast):
        raise ValueError("valid and broadcast have different lengths")
    samples_b = broadcast[::cfg.step].astype(np.float32)
    samples_b[~valid[::cfg.step]] = np.nan
    seen = np.isfinite(samples_b[:, :, 0])
    usable = seen.sum(axis=1) >= cfg.min_players
    centre = np.zeros((len(samples_b), 2), dtype=np.float32)
    count = np.maximum(seen.sum(axis=1), 1)[:, None]
    centre[:] = np.nan_to_num(samples_b, nan=0.0).sum(axis=1) / count
    starts = _anchors(usable, centre, cfg)
    if len(starts) < cfg.min_votes:
        raise NoAlignment("the broadcast has too few wide shots with players in them")

    maps = distance_maps(tactical[::cfg.step], pitch, cfg)
    subset = starts[np.linspace(0, len(starts) - 1, min(cfg.flip_anchors, len(starts))).astype(np.int64)]
    votes = []
    for flip in FLIPS:
        groups = _vote(_coarse(maps, samples_b * np.array(flip, dtype=np.float32), subset, pitch, cfg) - subset, cfg)
        votes.append(sum(len(g) for g in groups[:2]))
    if max(votes) < cfg.min_votes:
        raise NoAlignment("no shift between the two recordings is shared by enough clips")
    flip = FLIPS[int(np.argmax(votes))]
    turn = np.array(flip, dtype=np.float32)

    shifts = _coarse(maps, samples_b * turn, starts, pitch, cfg) - starts
    del maps
    groups = _vote(shifts, cfg)
    runs = _runs(starts, groups)
    if not runs:
        raise NoAlignment("the clips that agree do not form a stretch of the broadcast")
    agreed = int(sum(len(groups[g]) for g in {r[0] for r in runs}))

    turned = broadcast.astype(np.float32) * turn
    turned[~valid] = np.nan
    players_seen = np.isfinite(turned[:, :, 0]).sum(axis=1)
    usable_frames = players_seen >= cfg.min_players
    n = len(turned)

    # Pieces first meet half-way between the last clip of one and the first clip of the next.
    edges = [0] + [(runs[k][2] + runs[k + 1][1]) // 2 * cfg.step for k in range(len(runs) - 1)] + [n]
    coarse = [int(np.median(shifts[groups[r[0]]])) * cfg.step for r in runs]
    deltas = np.arange(-cfg.reach, cfg.reach + 1)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def piece_shifts(k: int):
        """(first frame of each two-minute window, its shift) for piece k, or None if nothing in it can be fitted."""
        firsts = np.arange(edges[k], edges[k + 1], cfg.window)
        curves = [_fine(turned, tactical, usable_frames, int(a), int(min(a + cfg.window, edges[k + 1])), coarse[k], cfg)
                  for a in firsts]
        have = [j for j, c in enumerate(curves) if c is not None]

        def best(windows_used) -> Optional[int]:
            i = int(np.sum([curves[j] for j in windows_used], axis=0).argmin())
            return None if i in (0, len(deltas) - 1) else int(deltas[i])   # still falling at the edge: no answer

        whole = best(have) if have else None
        if whole is None:
            return None
        # The broadcast runs in real time, so a piece has one shift. Fitting it on the whole piece at once
        # cancels the errors a slightly wrong pitch causes, which point one way when play goes left and
        # the other way when it goes right.
        values = np.full(len(firsts), coarse[k] + whole, dtype=np.int64)
        third = len(have) // 3
        if third >= 2:                                      # enough windows to see the two clocks drift apart
            early, late = best(have[:third]), best(have[-third:])
            if early is not None and late is not None and abs(early - late) > 3:
                x0, x1 = float(np.mean(have[:third])), float(np.mean(have[-third:]))
                slope = (late - early) / (x1 - x0)
                values = coarse[k] + np.rint(early + slope * (np.arange(len(firsts)) - x0)).astype(np.int64)
        return firsts, values

    def correct(shift_for_piece) -> None:
        """Measure how the broadcast's pitch is stretched or slid against the tactical one, and undo it."""
        nonlocal turned, matrix
        frames, moved = [], []
        for k in range(len(runs)):
            mine = np.flatnonzero(usable_frames[edges[k]:edges[k + 1]]) + edges[k]
            mine = mine[np.linspace(0, len(mine) - 1, min(1500, len(mine))).astype(np.int64)] if len(mine) else mine
            there = mine + shift_for_piece(k, mine)
            ok = (there >= 0) & (there < len(tactical))
            frames.append(mine[ok])
            moved.append(there[ok])
        frames, moved = np.concatenate(frames), np.concatenate(moved)
        if not len(frames):
            return
        found = _affine(*_pairs(turned[frames], tactical[moved], 5.0))
        if found is None:
            return
        turned = (turned @ found[:, :2].T.astype(np.float32) + found[:, 2].astype(np.float32)).astype(np.float32)
        matrix = np.vstack((found, [0.0, 0.0, 1.0])) @ np.vstack((matrix, [0.0, 0.0, 1.0]))
        matrix = matrix[:2]

    correct(lambda k, frames: np.full(len(frames), coarse[k]))
    windows = [piece_shifts(k) for k in range(len(runs))]

    def shift_of(k: int, frames: np.ndarray) -> np.ndarray:
        firsts, values = windows[k]
        return values[np.clip(np.searchsorted(firsts, frames, side="right") - 1, 0, len(values) - 1)]

    def fits(frames: np.ndarray, moved: np.ndarray) -> np.ndarray:
        """1 where the frame's players fit the tactical ones, 0 where they do not, NaN where it cannot be told."""
        out = np.full(len(frames), np.nan, dtype=np.float32)
        ok = (moved >= 0) & (moved < len(tactical)) & (players_seen[frames] >= 3)
        if ok.any():
            out[ok] = fit_distance(turned[frames[ok]], tactical[moved[ok]], cfg) < cfg.fit_m
        return out

    correct(lambda k, frames: shift_of(k, frames) if windows[k] is not None else np.full(len(frames), coarse[k]))

    # Move each meeting point to where the fit changes from one piece's shift to the next one's.
    for k in range(len(runs) - 1):
        if windows[k] is None or windows[k + 1] is None:
            continue
        frames = np.arange(runs[k][2] * cfg.step, runs[k + 1][1] * cfg.step)
        if not len(frames):
            continue
        gain = np.nan_to_num(fits(frames, frames + shift_of(k, frames))) \
            - np.nan_to_num(fits(frames, frames + shift_of(k + 1, frames)))
        total = np.concatenate(([0.0], np.cumsum(gain)))
        best = np.flatnonzero(total == total.max())          # between the two there is often no wide shot:
        edges[k + 1] = int(frames[0] + (best[0] + best[-1]) // 2)   # split what cannot be told down the middle

    shift = np.full(n, NO_SHIFT, dtype=np.int64)
    fit = np.full(n, np.nan, dtype=np.float32)
    pieces = []
    for k in range(len(runs)):
        if windows[k] is None or edges[k + 1] <= edges[k]:
            continue
        frames = np.arange(edges[k], edges[k + 1])
        moved = frames + shift_of(k, frames)
        inside = (moved >= 0) & (moved < len(tactical))
        shift[frames[inside]] = moved[inside] - frames[inside]
        fit[frames] = fits(frames, moved)
        pieces.append((int(edges[k]), int(edges[k + 1]), int(np.median(shift_of(k, frames)))))
    if not pieces:
        raise NoAlignment("no stretch of the broadcast could be fitted to the frame")

    # Whether a frame is live is decided with its neighbours, since single frames flicker.
    known = np.isfinite(fit).astype(np.float32)
    box = np.ones(2 * cfg.smooth + 1, dtype=np.float32)
    yes = np.convolve(np.nan_to_num(fit), box, mode="same")
    seen_near = np.convolve(known, box, mode="same")
    told = valid & (shift != NO_SHIFT) & (seen_near > 0)
    live = told & (yes * 2 > seen_near)
    state = np.where(~valid, NOT_WIDE, np.where(live, LIVE, np.where(told, REPLAY, UNSURE))).astype(np.uint8)
    # Too few players to tell: take the verdict of the nearest frame of the same unbroken wide shot.
    shot = np.cumsum(np.concatenate(([0], valid[1:] != valid[:-1])))
    judged, unsure = np.flatnonzero(told), np.flatnonzero((state == UNSURE) & (shift != NO_SHIFT))
    if len(judged) and len(unsure):
        after = np.clip(np.searchsorted(judged, unsure), 0, len(judged) - 1)
        before = np.clip(after - 1, 0, len(judged) - 1)
        near = np.where(np.abs(judged[after] - unsure) <= np.abs(unsure - judged[before]), judged[after], judged[before])
        ok = (shot[near] == shot[unsure]) & (np.abs(near - unsure) <= cfg.carry)
        state[unsure[ok]] = state[near[ok]]
    live = state == LIVE

    live_frames = np.flatnonzero(live & (players_seen >= 3))
    sample = live_frames[np.linspace(0, len(live_frames) - 1, min(2000, len(live_frames))).astype(np.int64)] \
        if len(live_frames) else live_frames
    typical = float(np.nanmedian(fit_distance(turned[sample], tactical[sample + shift[sample]], cfg))) \
        if len(sample) else float("nan")
    same = None
    if tactical_team is not None and broadcast_team is not None and len(sample):
        b, t = turned[sample], tactical[sample + shift[sample]]
        gap = b[:, :, None, :] - t[:, None, :, :]
        d = np.where(np.isfinite(gap).all(axis=-1), (gap * gap).sum(axis=-1), np.inf)
        close = np.isfinite(d.min(axis=2)) & (np.sqrt(d.min(axis=2)) < cfg.fit_m)
        match = np.asarray(tactical_team)[d.argmin(axis=2)] == np.asarray(broadcast_team)[None, :]
        if close.sum() >= 50:
            share = float(match[close].mean())
            same = True if share > 0.75 else False if share < 0.25 else None
    return Alignment(np.where(shift == NO_SHIFT, NO_SHIFT, shift).astype(np.int32), state, pieces, flip,
                     len(starts), agreed, typical, same, matrix)


def on_tactical_pitch(alignment: Alignment, points: np.ndarray) -> np.ndarray:
    """Broadcast positions [..., 2] in metres as the tactical recording would place them."""
    turned = points * np.array(alignment.flip, dtype=np.float32)
    if alignment.matrix is None:
        return turned
    return (turned @ alignment.matrix[:, :2].T + alignment.matrix[:, 2]).astype(np.float32)


def tactical_view(alignment: Alignment, n_tactical: int) -> Tuple[np.ndarray, np.ndarray]:
    """For each tactical frame: the broadcast frame showing that moment (-1 if none), and what it shows."""
    frame = np.full(n_tactical, -1, dtype=np.int32)
    state = np.full(n_tactical, NOT_COVERED, dtype=np.uint8)
    known = np.flatnonzero(alignment.shift != NO_SHIFT)
    moved = known + alignment.shift[known].astype(np.int64)
    ok = (moved >= 0) & (moved < n_tactical)
    known, moved = known[ok], moved[ok]
    # A piece is sure of itself between its first and its last live frame. Outside that it only carries
    # its shift on, into studio shots, and must not overrule another piece. Where two frames still claim
    # one tactical frame, the live wide shot wins, then any wide shot.
    sure = np.zeros(len(alignment.shift), dtype=bool)
    for first, last, _ in alignment.pieces:
        live = np.flatnonzero(alignment.state[first:last] == LIVE)
        if len(live):
            sure[first + live[0]:first + live[-1] + 1] = True
    for inside in (False, True):
        for kinds in ((NOT_WIDE,), (REPLAY, UNSURE), (LIVE,)):
            pick = (sure[known] == inside) & np.isin(alignment.state[known], kinds)
            frame[moved[pick]] = known[pick]
            state[moved[pick]] = alignment.state[known[pick]]
    # A shift that changes by a frame between two windows leaves a one-frame hole: fill it from before.
    for _ in range(2):
        hole = np.flatnonzero((state[1:-1] == NOT_COVERED) & (state[:-2] != NOT_COVERED) & (state[2:] != NOT_COVERED)) + 1
        frame[hole], state[hole] = frame[hole - 1], state[hole - 1]
    return frame, state


def _runs_of(mask: np.ndarray) -> List[Tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def director_link(labels: np.ndarray, state: np.ndarray, blocks, fps: int = 25) -> dict:
    """How closely the broadcast director's cutting follows a set of labels on the tactical timeline.

    labels are H/A/D/X codes per tactical frame (an algorithm's own labels, not an answer key); state
    is tactical_view's second output. Only frames inside blocks are used. "Away" means the broadcast is
    on a close-up, a crowd shot or a replay. Returns shares between 0 and 1 (None where nothing counts):

    away_dead, away_live   how often the broadcast is away while the labels say dead, and say live
    cuts_near_dead         of the cuts away that last 2 s or more, the share that begin within 1 s
                           before to 3 s after a frame labelled dead
    chance_near_dead       the same test at every wide-shot frame, which is what cuts at random would score
    stops_with_cut         of the dead spells of 5 s or more that the broadcast covers, the share with 1 s
                           or more away inside them
    back_before_restart    of those with a cut, the share where the live wide shot is back before play
                           restarts
    """
    from .match import D, X                                       # here, to keep align.py free of match at import
    inside = np.zeros(len(labels), dtype=bool)
    for a, b in blocks:
        inside[a:b] = True
    covered = inside & np.isin(state, (LIVE, NOT_WIDE, REPLAY))
    away = covered & np.isin(state, (NOT_WIDE, REPLAY))
    dead = labels == D
    live = (labels != D) & (labels != X)

    def share(part, whole):
        return float(part.sum() / whole.sum()) if whole.any() else None

    out = {"away_dead": share(away & dead, covered & dead), "away_live": share(away & live, covered & live)}
    near = np.zeros(len(labels), dtype=bool)                      # a dead frame from 1 s before to 3 s after
    for s, e in _runs_of(dead):
        near[max(s - 3 * fps, 0):e + fps] = True
    cuts = [s for s, e in _runs_of(away) if e - s >= 2 * fps and s > 0 and state[s - 1] == LIVE]
    out["cuts"] = len(cuts)
    out["cuts_near_dead"] = float(near[cuts].mean()) if cuts else None
    wide = covered & (state == LIVE)
    out["chance_near_dead"] = share(near & wide, wide)
    spells = [(s, e) for s, e in _runs_of(dead & inside) if e - s >= 5 * fps and covered[s:e].mean() >= 0.5]
    with_cut = [(s, e) for s, e in spells if away[s:e].sum() >= fps]
    out["stops"] = len(spells)
    out["stops_with_cut"] = len(with_cut) / len(spells) if spells else None
    back = [state[s + np.flatnonzero(away[s:e])[-1] + 1:e].tolist().count(LIVE) > 0 for s, e in with_cut]
    out["back_before_restart"] = float(np.mean(back)) if back else None
    return out
