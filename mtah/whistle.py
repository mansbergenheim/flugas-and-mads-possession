"""Find the referee's whistle in a match's sound. Plain numpy, no learning, deterministic.

What it looks for. A referee's whistle is one sharp, steady tone between about 1.5 and 6 kHz,
usually with an overtone at twice the pitch, lasting from a tenth of a second to a second or two,
and the same instrument all match. Fans whistling is the main trap: thousands of mouths make a
smeared chorus with no single clean peak, and one person's wolf whistle slides in pitch. A
commentator's voice has many weak peaks a couple of hundred hertz apart, never one dominant one.

How. The sound is cut into 50 ms frames every 10 ms. For each frame and each of two mixes, "mid"
(left plus right) and "side" (left minus right, which cancels whatever is mixed dead centre, as
commentary usually is), it measures within 1.5 to 6 kHz:

    pitch     the strongest frequency, to a fraction of a bin
    tonality  how far that peak stands above the band's median, in dB
    share     how much of the band's energy sits in that one peak
    overtone  how far the band around twice the pitch stands above its own median, in dB
    isolation how far the peak stands above the strongest other peak more than 6% of the pitch away
              (at least 60 Hz), in dB; a pea whistle's warble stays inside that
              (a voice's harmonics, or a chorus of fans, leave several peaks of similar height)

Frames that are tonal enough are chained while the pitch holds steady, the chains become events,
and events that wander in pitch are dropped. Every setting is chosen on other recordings and on
made-up sound, never on the competition match: the event thresholds on one RoboCup recording
(experiments/whistle_robocup.py), checked on a second. Measuring against a slowly tracked
background instead of the raw spectrum was tried there and lost, so it is not used.

Long recordings are fed in chunks (FeatureStream), so a 45-minute half never has to sit in memory
as audio. The result does not depend on how the sound was chunked.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

WINDOW_S = 0.05
HOP_S = 0.01
BAND = (1500.0, 6000.0)
MODES = ("mid", "side")
FIELDS = ("pitch", "tonality", "share", "overtone", "level", "isolation")
ISOLATION_WIDTH = 0.06  # other peaks closer than this (relative, at least 60 Hz) belong to the same one


@dataclass(frozen=True)
class EventRules:
    tonality_db: float = 22.0      # a frame counts as tonal above this
    min_share: float = 0.25        # and with at least this share of the band in its peak
    min_isolation_db: float = 3.0  # and standing this far above any other peak
    step: float = 0.04             # largest distance from the chain's recent pitch (a pea whistle warbles)
    gap_frames: int = 5            # chains this close, at the same pitch, are one event
    same_pitch: float = 0.03       # "the same pitch" for joining chains
    min_s: float = 0.08            # shortest event
    max_s: float = 4.0             # longest event
    max_spread: float = 0.03       # events whose pitch wanders more than this (relative) are dropped
    min_pitch: float = 0.0         # events lower than this (Hz) are dropped


def frame_sizes(sr: int):
    return int(round(WINDOW_S * sr)), int(round(HOP_S * sr))


def mixes(stereo: np.ndarray) -> Dict[str, np.ndarray]:
    """'mid' and 'side' from [n, channels] sound. A single channel has no side, so it is silent."""
    stereo = np.asarray(stereo, dtype=np.float64)
    if stereo.ndim == 1:
        stereo = stereo[:, None]
    if stereo.shape[1] == 1:
        return {"mid": stereo[:, 0], "side": np.zeros(len(stereo))}
    return {"mid": 0.5 * (stereo[:, 0] + stereo[:, 1]), "side": 0.5 * (stereo[:, 0] - stereo[:, 1])}


def _features(signal: np.ndarray, sr: int) -> np.ndarray:
    """[frames, len(FIELDS)] for whole frames of one mix."""
    win, hop = frame_sizes(sr)
    if len(signal) < win:
        return np.zeros((0, len(FIELDS)))
    frames = np.lib.stride_tricks.sliding_window_view(signal, win)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) ** 2 + 1e-20
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    lo, hi = np.searchsorted(freqs, BAND)
    hi = min(hi, len(freqs) - 1)
    band = power[:, lo:hi]
    k = band.argmax(axis=1)
    rows = np.arange(len(band))
    left = np.log(band[rows, np.clip(k - 1, 0, band.shape[1] - 1)])
    mid = np.log(band[rows, k])
    right = np.log(band[rows, np.clip(k + 1, 0, band.shape[1] - 1)])
    curve = left - 2 * mid + right
    shift = np.where(np.abs(curve) > 1e-12, 0.5 * (left - right) / np.where(curve == 0, 1, curve), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    step = freqs[1] - freqs[0]
    pitch = freqs[lo] + (k + shift) * step
    median = np.median(band, axis=1)
    tonality = 10 * np.log10(band[rows, k] / median)
    columns = np.arange(band.shape[1])[None, :]
    share = np.where(np.abs(columns - k[:, None]) <= 1, band, 0.0).sum(axis=1) / band.sum(axis=1)
    level = 10 * np.log10(band.sum(axis=1))
    width = np.ceil(np.maximum(60.0, ISOLATION_WIDTH * pitch) / step)
    near = np.abs(columns - k[:, None]) <= width[:, None]
    isolation = 10 * np.log10(band[rows, k] / np.where(near, 0.0, band).max(axis=1).clip(1e-20))
    overtone = np.full(len(band), np.nan)
    for r in range(len(band)):
        f2 = 2 * pitch[r]
        if f2 * 1.25 >= freqs[-1]:
            continue
        a, b = np.searchsorted(freqs, (f2 * 0.98, f2 * 1.02 + step))
        c, d = np.searchsorted(freqs, (f2 * 0.75, f2 * 1.25))
        overtone[r] = 10 * np.log10(power[r, a:b].max() / np.median(power[r, c:d]))
    out = np.stack([pitch, tonality, share, overtone, level, isolation], axis=1)
    return np.round(out, 4)           # so tiny floating-point differences between computers vanish


class FeatureStream:
    """Feed sound in chunks of any size; collect per-frame features for both mixes."""

    def __init__(self, sr: int):
        self.sr = sr
        self.win, self.hop = frame_sizes(sr)
        self.buffer = np.zeros((0, 2))
        self.parts: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
        self.channels: Optional[int] = None

    def push(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if self.channels is None:
            self.channels = chunk.shape[1]
        if chunk.shape[1] == 1:
            chunk = np.repeat(chunk, 2, axis=1)       # one channel: the side mix is silent
        self.buffer = np.concatenate([self.buffer, chunk[:, :2]])
        n = (len(self.buffer) - self.win) // self.hop + 1 if len(self.buffer) >= self.win else 0
        if n <= 0:
            return
        used = (n - 1) * self.hop + self.win
        for mode, signal in mixes(self.buffer[:used]).items():
            self.parts[mode].append(_features(signal, self.sr))
        self.buffer = self.buffer[n * self.hop:]

    def result(self) -> Dict[str, np.ndarray]:
        return {m: (np.concatenate(p) if p else np.zeros((0, len(FIELDS)))) for m, p in self.parts.items()}


def features(sound: np.ndarray, sr: int, chunk_s: float = 60.0) -> Dict[str, np.ndarray]:
    stream = FeatureStream(sr)
    size = int(chunk_s * sr)
    for start in range(0, len(sound), size):
        stream.push(sound[start:start + size])
    return stream.result()


def frame_time(index, sr: int):
    """Centre of frame `index`, in seconds from the start of the sound."""
    win, hop = frame_sizes(sr)
    return (np.asarray(index) * hop + win / 2.0) / sr


def find_events(feat: np.ndarray, sr: int, rules: EventRules = EventRules()) -> List[dict]:
    """Whistle-like events in one mix's features."""
    pitch, tonality, share, isolation = feat[:, 0], feat[:, 1], feat[:, 2], feat[:, 5]
    tonal = np.flatnonzero((tonality >= rules.tonality_db) & (share >= rules.min_share)
                           & (isolation >= rules.min_isolation_db))
    chains: List[List[int]] = []
    for t in tonal:
        if (chains and t == chains[-1][-1] + 1
                and abs(pitch[t] / np.median(pitch[chains[-1][-10:]]) - 1) <= rules.step):
            chains[-1].append(int(t))
        else:
            chains.append([int(t)])
    merged: List[List[int]] = []
    for chain in chains:
        if merged:
            last = merged[-1]
            if (chain[0] - last[-1] - 1 <= rules.gap_frames
                    and abs(np.median(pitch[chain]) / np.median(pitch[last]) - 1) <= rules.same_pitch):
                last.extend(range(last[-1] + 1, chain[0]))
                last.extend(chain)
                continue
        merged.append(list(chain))
    win_s, hop = (n / sr for n in frame_sizes(sr))
    events = []
    for chain in merged:
        idx = np.array(chain)
        good = idx[(tonality[idx] >= rules.tonality_db) & (isolation[idx] >= rules.min_isolation_db)]
        duration = max(len(idx) * hop - (win_s - hop), hop)     # frames overlap the sound by a window
        if duration < rules.min_s or duration > rules.max_s or len(good) < 2:
            continue
        p = pitch[good]
        centre = float(np.median(p))
        spread = float((np.percentile(p, 75) - np.percentile(p, 25)) / centre)
        if spread > rules.max_spread or centre < rules.min_pitch:
            continue
        overtone = feat[good, 3]
        events.append({
            "start": float(frame_time(idx[0], sr) - hop / 2), "duration": round(duration, 3), "pitch": round(centre, 1),
            "spread": round(spread, 4), "tonality": round(float(tonality[good].mean()), 2),
            "share": round(float(share[good].mean()), 3), "isolation": round(float(isolation[good].mean()), 2),
            "overtone": round(float(np.nanmean(overtone)), 2) if np.isfinite(overtone).any() else None,
            "warble": round(float(np.std(p / centre - 1)), 4),
        })
    return events


def combined_events(feats: Dict[str, np.ndarray], sr: int, rules: EventRules = EventRules()) -> List[dict]:
    """Events from both mixes. Where the two overlap in time, the more tonal one is kept.

    The side mix hears what is off-centre over the commentary; a whistle mixed dead centre vanishes
    from it, and the mid mix still has that one."""
    return merge_mixes({m: find_events(feats[m], sr, rules) for m in MODES})


def merge_mixes(by_mix: Dict[str, List[dict]]) -> List[dict]:
    """One list from each mix's events; where two overlap in time the more tonal one is kept."""
    pool = sorted(((e, m) for m, events in by_mix.items() for e in events), key=lambda x: x[0]["start"])
    out: List[dict] = []
    for event, mode in pool:
        event = dict(event, mix=mode)
        if out and event["start"] < out[-1]["start"] + out[-1]["duration"]:
            if event["tonality"] > out[-1]["tonality"]:
                out[-1] = event
            continue
        out.append(event)
    return out


def one_voice(events: Sequence[dict], tolerance: float = 0.03):
    """(the pitch most events share, the share of events within `tolerance` of it). (None, 0) if none."""
    if not events:
        return None, 0.0
    pitches = np.array([e["pitch"] for e in events])
    counts = [int((np.abs(pitches / p - 1) <= tolerance).sum()) for p in pitches]
    best = int(np.argmax(counts))
    return float(pitches[best]), counts[best] / len(pitches)


def pitch_groups(events: Sequence[dict], tolerance: float = 0.03, min_size: int = 3) -> List[dict]:
    """Events grouped by pitch, biggest group first: {"pitch", "members" (indices into events)}.

    Repeatedly take the pitch that has the most events within `tolerance` of it, make those a
    group, and carry on with the rest. Ties go to the lower pitch, so the result is deterministic."""
    left = list(range(len(events)))
    pitches = np.array([e["pitch"] for e in events], dtype=float)
    groups: List[dict] = []
    while left:
        sub = pitches[left]
        near = np.abs(sub[None, :] / sub[:, None] - 1) <= tolerance
        counts = near.sum(axis=1)
        best = max(range(len(left)), key=lambda i: (counts[i], -sub[i]))
        if counts[best] < min_size:
            break
        members = [left[j] for j in np.flatnonzero(near[best])]
        groups.append({"pitch": float(np.median(pitches[members])), "members": members})
        left = [i for i in left if i not in set(members)]
    return groups


def near_stops(starts: np.ndarray, stops: np.ndarray, before: float = 1.0, after: float = 3.0) -> np.ndarray:
    """For each event start, is there a stoppage start from `before` s earlier to `after` s later?

    A referee whistles as play stops, so a stoppage the tracking finds tends to begin from a moment
    before the whistle (the decision) to a few seconds after it (the players slowing down)."""
    starts, stops = np.asarray(starts, dtype=float), np.sort(np.asarray(stops, dtype=float))
    if not len(stops) or not len(starts):
        return np.zeros(len(starts), dtype=bool)
    lo = np.searchsorted(stops, starts - before)
    hi = np.searchsorted(stops, starts + after, side="right")
    return hi > lo


def referee_whistles(events: Sequence[dict], stops: Optional[Sequence[float]] = None, tolerance: float = 0.03,
                     candidates: int = 3, min_size: int = 5):
    """Keep only the events at the referee's pitch. Returns (kept events, report).

    One referee blows one whistle all match, so the referee's whistles form a tight group at one
    pitch, while false alarms (voices, horns, jingles, fans) are spread over many. Without `stops`
    the biggest group is taken. With `stops`, stoppage start times in seconds found from the
    tracking, the group among the `candidates` biggest whose events most often fall near a
    stoppage start is taken instead, which guards against a recurring non-referee sound (a
    broadcast jingle at one pitch, say) that happens to be more frequent. No person chooses."""
    groups = [g for g in pitch_groups(events, tolerance, min_size)][:candidates]
    report = {"groups": [{"pitch": round(g["pitch"], 1), "size": len(g["members"])} for g in groups],
              "chosen": None, "how": None, "kept": 0, "of": len(events)}
    if not groups:
        report["how"] = "no pitch is shared by enough events"
        return [], report
    if stops is not None and len(groups) > 1:
        starts = np.array([e["start"] for e in events])
        link = [float(near_stops(starts[g["members"]], np.asarray(stops)).mean()) for g in groups]
        for g, share in zip(report["groups"], link):
            g["near_stops"] = round(share, 3)
        pick = int(np.argmax(link))
        report["how"] = "the group most often near a stoppage start"
    else:
        pick = 0
        report["how"] = "the biggest group"
    chosen = groups[pick]
    kept = [events[i] for i in sorted(chosen["members"])]
    report.update(chosen=round(chosen["pitch"], 1), kept=len(kept))
    return kept, report
