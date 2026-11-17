"""Hear the referee's whistle by learning its pitch first, then listening only there.

The plain detector (whistle.py) takes the strongest tonal peak at each moment. On TV sound the
commentators usually beat a distant whistle, so it does little better than chance. But a referee's
whistle is a very narrow, steady tone at one pitch that stays the same all match. Averaged over the
moments play stops, voices and crowd blur out and the whistle stands up as a narrow line. So:

1. learn the line: the spectrum from 1 s before to 3 s after each stoppage, against quiet play;
   the frequency between 2 and 5 kHz that rises most above its own surroundings is the pitch, and
   how far it rises ("line", in dB) says whether there is a usable whistle at all;
2. listen there: a whistle is where the power within 1.5% of that pitch stands far above the ring
   from 4% to 12% away, for long enough.

The settings below are the first of three safety layers against false alarms (16 Nov 2026). They
were chosen on 21 SoccerNet LaLiga matches by a rule fixed beforehand (experiments/whistle_narrowband.py,
runs/whistle_narrowband.txt): at most 0.1 detections a minute in quiet play, at least 60% of
detections near a labelled stoppage, then the most fouls and offsides found. On the 20 matches held
back they gave: detector on for 15 of 20, 2.5 fouls/offsides found a match, 60% of detections near a
stoppage against 33% by chance, 0.07 detections a minute in quiet play. That is far from safe on its
own, which is why one soundtrack is never used alone: layer two (whistle_check.py) keeps a whistle
only when both of the match's soundtracks hear it at the same moment and pitch, and layer three (the
decoder) lets it only move the start of a stoppage the candidate already has. The settings were never
chosen on the competition match and must not be changed by looking at it.

numpy and scipy only; the same sound gives the same answer on any machine.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter

SR = 48000
WIN, HOP = 2048, 1200                 # 43 ms windows every 25 ms
DT = HOP / SR
FREQS = np.fft.rfftfreq(WIN, 1 / SR)
BAND = (FREQS >= 1500) & (FREQS <= 6000)
FB = FREQS[BAND]                      # the frequencies kept, Hz
SEARCH = (2000.0, 5000.0)             # where the pitch is looked for
CENTRE = 0.015                        # the whistle band: within 1.5% of the pitch
RING = (0.04, 0.12)                   # the background: from 4% to 12% away
MEDIAN_BINS = 41                      # how widely "its own surroundings" reaches when learning the line
BEFORE_S, AFTER_S = 1.0, 3.0          # the stretch around a stoppage that is averaged


@dataclass(frozen=True)
class Settings:
    threshold_db: float = 24.0        # whistle band above the ring
    min_s: float = 0.15               # for at least this long (one unbroken piece)
    min_line_db: float = 2.0          # below this the learned line is not trusted: detector off
    join_s: float = 0.5               # pieces this close are one whistle (a warbling whistle breaks up)

    @property
    def min_frames(self) -> int:
        return int(round(self.min_s / DT))


class Spectrum:
    """Builds the power spectrum [frames, len(FB)] from mono or stereo chunks, whatever their sizes."""

    def __init__(self):
        self._rest = np.zeros(0, dtype=np.float64)
        self._parts: List[np.ndarray] = []
        self._window = np.hanning(WIN)

    def push(self, chunk: np.ndarray) -> None:
        x = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
        x = np.concatenate([self._rest, x.astype(np.float64)])
        n = (len(x) - WIN) // HOP + 1
        if n <= 0:
            self._rest = x
            return
        frames = np.lib.stride_tricks.as_strided(x, (n, WIN), (x.strides[0] * HOP, x.strides[0]))
        power = np.abs(np.fft.rfft(frames * self._window, axis=1)) ** 2
        self._parts.append(power[:, BAND].astype(np.float32))
        self._rest = x[n * HOP:].copy()

    def result(self) -> np.ndarray:
        if not self._parts:
            return np.zeros((0, len(FB)), dtype=np.float32)
        return np.concatenate(self._parts)


def spectrum(chunks: Iterable[np.ndarray]) -> np.ndarray:
    built = Spectrum()
    for chunk in chunks:
        built.push(chunk)
    return built.result()


def frames_at(times: Sequence[float], n: int) -> np.ndarray:
    """Spectrum frames from BEFORE_S before to AFTER_S after each time (seconds), inside 0..n."""
    if len(times) == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.concatenate([np.arange(int((t - BEFORE_S) / DT), int((t + AFTER_S) / DT)) for t in times])
    return idx[(idx >= 0) & (idx < n)]


def learn_line(S: np.ndarray, stops: Sequence[float], quiet: np.ndarray) -> Tuple[float, float]:
    """(pitch in Hz, line in dB) from the stoppage times (seconds) and a mask of quiet frames.

    Returns (nan, -inf) when there is nothing to learn from."""
    idx = frames_at(stops, len(S))
    quiet_idx = np.flatnonzero(quiet[: len(S)])
    if len(idx) == 0 or len(quiet_idx) == 0:
        return float("nan"), float("-inf")
    ratio = 10 * np.log10((np.asarray(S[idx], dtype=np.float64).mean(0) + 1e-20)
                          / (np.asarray(S[quiet_idx], dtype=np.float64).mean(0) + 1e-20))
    line = ratio - median_filter(ratio, size=MEDIAN_BINS)
    inside = (FB >= SEARCH[0]) & (FB <= SEARCH[1])
    k = int(np.argmax(np.where(inside, line, -np.inf)))
    return float(FB[k]), float(line[k])


def contrast(S: np.ndarray, pitch: float) -> np.ndarray:
    """Per frame, dB of the strongest bin within CENTRE of the pitch over the ring's median."""
    away = np.abs(FB / pitch - 1)
    centre = away <= CENTRE
    ring = (away >= RING[0]) & (away <= RING[1])
    top = np.asarray(S[:, centre], dtype=np.float64).max(1)
    floor = np.median(np.asarray(S[:, ring], dtype=np.float64), axis=1)
    return (10 * np.log10((top + 1e-20) / (floor + 1e-20))).astype(np.float32)


def find(c: np.ndarray, settings: Settings = Settings()) -> List[dict]:
    """Whistles in a contrast series: dicts with start and duration in seconds."""
    on = c >= settings.threshold_db
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
    gap = int(round(settings.join_s / DT))
    joined: List[List[int]] = []                  # [start, end, longest unbroken piece]
    for a, b in zip(edges[::2].tolist(), edges[1::2].tolist()):
        if joined and a - joined[-1][1] <= gap:
            joined[-1][1] = b
            joined[-1][2] = max(joined[-1][2], b - a)
        else:
            joined.append([a, b, b - a])
    # A whistle needs one unbroken piece of at least min_s; joining only stops it being counted twice.
    out = [{"start": a * DT, "duration": (b - a) * DT} for a, b, longest in joined if longest >= settings.min_frames]
    return out


def same_pitch(a: float, b: float, within: float = CENTRE) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a / b - 1) <= within)


def coincide(a: Sequence[float], b: Sequence[float], tolerance: float = 0.3) -> np.ndarray:
    """For each time in a, whether some time in b lies within tolerance seconds."""
    a, b = np.asarray(a, dtype=np.float64), np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), dtype=np.int64)
    nearest = np.minimum(np.abs(b[i] - a), np.abs(b[np.maximum(i - 1, 0)] - a))
    return nearest <= tolerance


CHANCE_SHIFTS = tuple(float(s) for s in np.concatenate([-np.arange(30, 330, 30), np.arange(30, 330, 30)]))


def chance_coincide(a: Sequence[float], b: Sequence[float], tolerance: float = 0.3,
                    shifts: Sequence[float] = CHANCE_SHIFTS) -> float:
    """The share of a that would coincide with b if b were slid by 30 s to 5 min: what chance gives."""
    if len(a) == 0:
        return float("nan")
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean([coincide(a, b + s, tolerance).mean() for s in shifts]))
