"""Listen for the referee's whistle in two soundtracks of one match and see whether they agree.

The tactical recording and the TV broadcast both carry sound. A whistle is a single moment, so once
the broadcast is lined up with the tactical timeline (align.py), a real whistle should be heard in
both at the same moment, while most false alarms come from things only one of them carries (the
commentators are on the TV sound). This module does the counting; scripts/check_whistles.py reads the
files and prints the result in words.

Nothing here uses an answer key. Stoppages come from a set of labels made by an algorithm (the
candidate): a stoppage starts where play is labelled live and then dead for 2 s or more, and quiet
play is live play with no dead label within 10 s. The detector is narrowband.py with its frozen
settings; each half's pitch is learned on the other half's stoppages, as when it was tested.

Safety (false alarms matter far more than misses): one soundtrack is never trusted alone. A whistle
is "safe" only when both soundtracks hear it within TOGETHER_S, in a half where both have a usable
line at the same pitch; the report also gives how many such pairs chance alone would produce on this
match, and how many safe whistles sit close enough to a candidate stoppage for the decoder to use.

Times: a soundtrack's spectrum frame j is at j * DT seconds from its first audio sample. offset is
where video frame 0 sits on that clock, in seconds, so frame k of that video is at k / fps + offset.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import narrowband as nb
from .match import A, D, H

MIN_STOP_S = 2.0
QUIET_MARGIN_S = 10.0
NEAR = (-1.0, 4.0)            # a detection "at" a stoppage starts between 1 s before and 4 s after it
TOGETHER_S = 0.3              # two soundtracks hear the same whistle when their starts are this close
MOVE_S = 2.0                  # layer three: a whistle may only move a candidate stoppage start this far
SPEECH = (1500.0, 3000.0)     # band whose rise and fall shows whether the soundtracks carry the same sound
MAX_LAG_S = 2.0


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def stoppages(labels: np.ndarray, blocks, fps: int = 25) -> List[np.ndarray]:
    """Per block: the frames where play goes from live to dead for MIN_STOP_S or more."""
    out = []
    for a, b in blocks:
        dead = labels[a:b] == D
        starts = [a + s for s, e in _runs(dead) if e - s >= MIN_STOP_S * fps and s > 0
                  and labels[a + s - 1] in (H, A)]
        out.append(np.array(starts, dtype=np.int64))
    return out


def quiet_frames(labels: np.ndarray, blocks, fps: int = 25) -> np.ndarray:
    """Live frames inside the blocks with no dead frame within QUIET_MARGIN_S."""
    dead = (labels == D).astype(np.int64)
    reach = int(QUIET_MARGIN_S * fps)
    near_dead = np.convolve(dead, np.ones(2 * reach + 1, dtype=np.int64), mode="same") > 0
    inside = np.zeros(len(labels), dtype=bool)
    for a, b in blocks:
        inside[a:b] = True
    return inside & np.isin(labels, (H, A)) & ~near_dead


class Track:
    """One soundtrack and how its spectrum frames sit on the tactical timeline."""

    def __init__(self, name: str, S: np.ndarray, to_tactical: np.ndarray, from_tactical: np.ndarray, fps: int = 25):
        self.name, self.S, self.fps = name, S, fps
        self.to_tactical = to_tactical          # per spectrum frame: tactical frame, or -1
        self.from_tactical = from_tactical      # per tactical frame: seconds on this soundtrack's clock, or nan

    @classmethod
    def tactical(cls, name: str, S: np.ndarray, offset: float, n_tactical: int, fps: int = 25) -> "Track":
        k = np.round((np.arange(len(S)) * nb.DT - offset) * fps).astype(np.int64)
        k[(k < 0) | (k >= n_tactical)] = -1
        return cls(name, S, k, np.arange(n_tactical) / fps + offset, fps)

    @classmethod
    def broadcast(cls, name: str, S: np.ndarray, offset: float, shift: np.ndarray, no_shift: int,
                  n_tactical: int, fps: int = 25) -> "Track":
        """shift: per broadcast frame, tactical frame = frame + shift (no_shift where not lined up)."""
        f = np.round((np.arange(len(S)) * nb.DT - offset) * fps).astype(np.int64)
        ok = (f >= 0) & (f < len(shift))
        k = np.full(len(S), -1, dtype=np.int64)
        s = shift[np.clip(f, 0, len(shift) - 1)].astype(np.int64)
        ok &= s != no_shift
        k[ok] = f[ok] + s[ok]
        k[(k < 0) | (k >= n_tactical)] = -1
        back = np.full(n_tactical, np.nan)
        frames = np.flatnonzero(shift != no_shift)
        moved = frames + shift[frames].astype(np.int64)
        good = (moved >= 0) & (moved < n_tactical)
        back[moved[good]] = frames[good] / fps + offset
        return cls(name, S, k, back, fps)

    def delayed(self, lag: float) -> "Track":
        """The same soundtrack, taken to run lag seconds late against the tactical timeline."""
        n = int(round(lag / nb.DT))
        to = np.full(len(self.to_tactical), -1, dtype=np.int64)
        if n >= 0:
            to[n:] = self.to_tactical[:len(to) - n]
        else:
            to[:n] = self.to_tactical[-n:]
        return Track(self.name, self.S, to, self.from_tactical + n * nb.DT, self.fps)

    def seconds(self, tactical_frames: np.ndarray) -> np.ndarray:
        t = self.from_tactical[tactical_frames]
        return t[np.isfinite(t)]

    def in_block(self, block) -> np.ndarray:
        return (self.to_tactical >= block[0]) & (self.to_tactical < block[1])


def listen(track: Track, stops: Sequence[np.ndarray], quiet: np.ndarray, blocks,
           settings: nb.Settings = nb.Settings()) -> dict:
    """Learn each half's pitch on the other half, then find whistles. Times out are tactical seconds."""
    fps = track.fps
    quiet_here = np.zeros(len(track.S), dtype=bool)
    mapped = track.to_tactical >= 0
    quiet_here[mapped] = quiet[track.to_tactical[mapped]]
    halves = []
    for h, block in enumerate(blocks):
        other = (h + 1) % len(blocks)
        learn_quiet = quiet_here & track.in_block(blocks[other])
        pitch, line = nb.learn_line(track.S, track.seconds(stops[other]), learn_quiet)
        found = []
        if np.isfinite(pitch):
            inside = track.in_block(block)
            for e in nb.find(nb.contrast(track.S, pitch), settings):
                j = int(round(e["start"] / nb.DT))
                if j < len(inside) and inside[j]:
                    found.append(track.to_tactical[j] / fps)
        halves.append({"pitch": pitch, "line": line, "starts": np.array(found)})
    on = all(h["line"] >= settings.min_line_db for h in halves)
    same = len(halves) == 2 and nb.same_pitch(halves[0]["pitch"], halves[1]["pitch"])
    covered = np.zeros(len(quiet), dtype=bool)
    covered[track.to_tactical[mapped]] = True
    return {"name": track.name, "halves": halves, "on": on, "same_pitch_both_halves": same,
            "starts": np.concatenate([h["starts"] for h in halves]) if halves else np.zeros(0),
            "covered": covered}


def at_stoppages(starts: np.ndarray, stop_seconds: np.ndarray) -> float:
    """Share of the stoppages with a detection starting from NEAR[0] to NEAR[1] seconds around them."""
    if len(stop_seconds) == 0:
        return float("nan")
    s = np.sort(np.asarray(starts, dtype=np.float64))
    if len(s) == 0:
        return 0.0
    lo = np.searchsorted(s, stop_seconds + NEAR[0], side="left")
    hi = np.searchsorted(s, stop_seconds + NEAR[1], side="right")
    return float(np.mean(hi > lo))


def at_stoppages_by_chance(starts: np.ndarray, stop_seconds: np.ndarray) -> float:
    if len(stop_seconds) == 0:
        return float("nan")
    return float(np.mean([at_stoppages(np.asarray(starts) + s, stop_seconds) for s in nb.CHANCE_SHIFTS]))


def per_quiet_minute(starts: np.ndarray, quiet: np.ndarray, fps: int = 25) -> float:
    minutes = quiet.sum() / fps / 60
    if minutes == 0:
        return float("nan")
    k = np.round(np.asarray(starts) * fps).astype(np.int64)
    k = k[(k >= 0) & (k < len(quiet))]
    return float(quiet[k].sum() / minutes)


def merge(a: np.ndarray, b: np.ndarray, tolerance: float = TOGETHER_S) -> np.ndarray:
    """Starts heard in either: all of a, plus those of b not within tolerance of one in a."""
    b = np.asarray(b, dtype=np.float64)
    return np.sort(np.concatenate([np.asarray(a, dtype=np.float64), b[~nb.coincide(b, a, tolerance)]]))


def envelope(track: Track, band: Tuple[float, float] = SPEECH, fps: int = 25) -> np.ndarray:
    """Per tactical frame (nan where not heard): log power in the band, with its 5 s running median taken off."""
    from scipy.ndimage import median_filter
    cols = (nb.FB >= band[0]) & (nb.FB <= band[1])
    level = np.log10(np.asarray(track.S[:, cols], dtype=np.float64).sum(1) + 1e-20)
    level = level - median_filter(level, size=int(round(5.0 / nb.DT)) | 1)
    out = np.full(len(track.from_tactical), np.nan)
    ok = track.to_tactical >= 0
    out[track.to_tactical[ok]] = level[ok]
    return out


def sound_link(a: Track, b: Track, blocks, fps: int = 25) -> dict:
    """How closely the two soundtracks rise and fall together, and at which lag of b (seconds) they do so best.

    A high correlation means largely the same sound (for example the same stadium feed); a low one
    means different microphones, or one of them carrying the commentators and the other not."""
    ea, eb = envelope(a, fps=fps), envelope(b, fps=fps)
    inside = np.zeros(len(ea), dtype=bool)
    for lo, hi in blocks:
        inside[lo:hi] = True
    step = max(1, int(round(0.04 * fps)))                       # about every 40 ms
    lags = np.arange(-int(MAX_LAG_S * fps), int(MAX_LAG_S * fps) + 1, step)
    best = (float("nan"), 0.0)
    scores = []
    idx = np.flatnonzero(inside)
    for lag in lags:
        j = idx + lag
        j = j[(j >= 0) & (j < len(eb))]
        i = j - lag
        x, y = ea[i], eb[j]
        ok = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 100 else float("nan")
        scores.append(r)
        if np.isfinite(r) and not (r <= best[0]):
            best = (r, lag / fps)
    scores = np.array(scores)
    at_zero = scores[len(scores) // 2] if len(scores) else float("nan")
    return {"best_r": best[0], "lag_s": best[1], "r_at_zero": float(at_zero),
            "median_r": float(np.nanmedian(scores)) if np.isfinite(scores).any() else float("nan")}


def check(labels: np.ndarray, blocks, tactical: Optional[Track], broadcast: Optional[Track],
          settings: nb.Settings = nb.Settings(), fps: int = 25) -> Dict:
    """Everything scripts/check_whistles.py reports, as numbers. Either soundtrack may be missing."""
    stops = stoppages(labels, blocks, fps)
    quiet = quiet_frames(labels, blocks, fps)
    all_stops = np.concatenate(stops) if stops else np.zeros(0, dtype=np.int64)
    out: Dict = {"stoppages": int(len(all_stops)), "quiet_minutes": float(quiet.sum() / fps / 60), "tracks": {}}
    heard = {}
    link = None
    if tactical is not None and broadcast is not None:
        link = sound_link(tactical, broadcast, blocks, fps)
        out["link"] = link
        # A clear peak away from zero means the broadcast's sound runs early or late against its picture;
        # the broadcast's sound is then moved by that lag before listening, decided by the sound alone.
        if np.isfinite(link["best_r"]) and link["best_r"] >= 0.3 and link["best_r"] - link["median_r"] >= 0.15:
            out["lag_used_s"] = link["lag_s"]
        else:
            out["lag_used_s"] = 0.0
        if out["lag_used_s"]:
            broadcast = broadcast.delayed(out["lag_used_s"])
    for track in (tactical, broadcast):
        if track is None:
            continue
        got = listen(track, stops, quiet, blocks, settings)
        stop_s = all_stops[got["covered"][all_stops]] / fps
        got["at_stops"] = at_stoppages(got["starts"], stop_s)
        got["at_stops_chance"] = at_stoppages_by_chance(got["starts"], stop_s)
        got["quiet_per_min"] = per_quiet_minute(got["starts"], quiet & got["covered"], fps)
        got["count"] = int(len(got["starts"]))
        out["tracks"][track.name] = got
        heard[track.name] = got
    if tactical is not None and broadcast is not None:
        t, b = heard[tactical.name], heard[broadcast.name]
        both_cover = t["covered"] & b["covered"]
        k = np.round(t["starts"] * fps).astype(np.int64)
        t_in = t["starts"][both_cover[np.clip(k, 0, len(both_cover) - 1)]]
        k = np.round(b["starts"] * fps).astype(np.int64)
        b_in = b["starts"][both_cover[np.clip(k, 0, len(both_cover) - 1)]]
        pair = {"same_pitch": [nb.same_pitch(x["pitch"], y["pitch"]) for x, y in zip(t["halves"], b["halves"])],
                "together": float(nb.coincide(t_in, b_in).mean()) if len(t_in) else float("nan"),
                "together_chance": nb.chance_coincide(t_in, b_in) if len(t_in) else float("nan"),
                "together_from_broadcast": float(nb.coincide(b_in, t_in).mean()) if len(b_in) else float("nan")}
        stop_s = all_stops[both_cover[all_stops]] / fps
        quiet_both = quiet & both_cover
        both = t_in[nb.coincide(t_in, b_in)]
        # Layer two, the safe whistles: heard in both, in a half where both soundtracks have a usable line
        # and learned the same pitch. How many such pairs chance alone would give is measured on the
        # match itself, by sliding one soundtrack's detections 30 s to 5 min.
        usable = [t["halves"][h]["line"] >= settings.min_line_db and b["halves"][h]["line"] >= settings.min_line_db
                  and pair["same_pitch"][h] for h in range(len(blocks))]
        in_usable = np.zeros(len(both), dtype=bool)
        for h, (lo, hi) in enumerate(blocks):
            if usable[h]:
                in_usable |= (both * fps >= lo) & (both * fps < hi)
        safe = both[in_usable]
        pair["usable_halves"] = usable
        pair["chance_pairs"] = (pair["together_chance"] * len(t_in)) if len(t_in) else 0.0
        # Layer three, previewed: a safe whistle can only act if a candidate stoppage starts within MOVE_S
        # of it (from MOVE_S before to MOVE_S after); the rest would be ignored.
        starts_s = np.sort(all_stops / fps)
        lo = np.searchsorted(starts_s, safe - MOVE_S, side="left")
        hi = np.searchsorted(starts_s, safe + MOVE_S, side="right")
        acts = hi > lo
        pair["safe"] = {"count": int(len(safe)), "would_act": int(acts.sum()),
                        "would_act_chance": float(np.mean([np.mean(np.searchsorted(starts_s, safe + d + MOVE_S, side="right")
                                                                   > np.searchsorted(starts_s, safe + d - MOVE_S, side="left"))
                                                           for d in nb.CHANCE_SHIFTS])) if len(safe) else float("nan"),
                        "quiet_per_min": per_quiet_minute(safe, quiet_both, fps)}
        either = merge(t_in, b_in)
        for name, starts in (("both", both), ("either", either)):
            pair[name] = {"count": int(len(starts)), "at_stops": at_stoppages(starts, stop_s),
                          "at_stops_chance": at_stoppages_by_chance(starts, stop_s),
                          "quiet_per_min": per_quiet_minute(starts, quiet_both, fps)}
        out["pair"] = pair
        out["events"] = {"tactical": t["starts"], "broadcast": b["starts"], "both": both, "either": either, "safe": safe}
    else:
        out["events"] = {name: got["starts"] for name, got in heard.items()}
    return out
