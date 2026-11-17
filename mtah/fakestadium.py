"""Made-up stadium sound with known referee whistles, for testing mtah/whistle.py.

Not a model of any real match. It checks that the detector's code does what it should and lets us
compare ideas (side mix, full bandwidth) under conditions we control; it cannot say how well the
detector works on real broadcasts, because every sound here follows rules written by us. That is
what other matches with real sound are for.

A scene holds, each optional:
  crowd       pink noise, partly different in the two channels, slowly swelling, with roars
  commentary  a voice-like sound with harmonics and formants, mixed almost dead centre
  whistles    one referee's whistle all scene: a pea whistle (fast warble) or a pealess one
              (two close tones beating), with an overtone, short or long blasts, off-centre with
              a small delay between channels and a few echoes
  fans        bursts of a whistling chorus (many wobbling tones) and single sliding wolf whistles
  blips       other steady tones at random pitches (horns, phones, a robot's jingle, a held note):
              the kind of sound the detector mistakes for a whistle on real recordings
  stings      a recurring broadcast sound at one fixed pitch, partly a few seconds after whistles
              (as a replay jingle would be), partly at random
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Scene:
    sr: int = 48000
    seconds: float = 120.0
    whistle_db: float = -15.0       # whistle level against the crowd, whole band
    commentary_db: float = 6.0      # commentary against the crowd; None for none
    fans: bool = True
    whistles_per_minute: float = 6.0
    blips_per_minute: float = 0.0
    sting_pitch: Optional[float] = None
    sting_after_whistle: float = 0.0     # share of whistles followed by the sting 3 to 6 s later
    sting_per_minute: float = 0.0        # and this many more at random
    seed: int = 0


@dataclass
class Made:
    sound: np.ndarray                       # [n, 2]
    onsets: List[float] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)
    lengths: List[float] = field(default_factory=list)
    pitch: float = 0.0
    blips: List[float] = field(default_factory=list)
    stings: List[float] = field(default_factory=list)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def _pink(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=float)
    f[0] = 1.0
    return np.fft.irfft(spec / np.sqrt(f), n)


def _crowd(scene, rng):
    n = int(scene.sr * scene.seconds)
    common = _pink(n, rng)
    out = np.stack([0.6 * common + 0.8 * _pink(n, rng), 0.6 * common + 0.8 * _pink(n, rng)], axis=1)
    t = np.arange(n) / scene.sr
    swell = 1.0 + 0.3 * np.sin(2 * np.pi * t / rng.uniform(15, 40) + rng.uniform(0, 6))
    for _ in range(int(scene.seconds / 30) + 1):
        at, length = rng.uniform(0, scene.seconds), rng.uniform(1, 3)
        swell += 1.5 * np.exp(-0.5 * ((t - at) / (length / 3)) ** 2)
    return out * swell[:, None]


def _voice(scene, rng):
    n = int(scene.sr * scene.seconds)
    out = np.zeros(n)
    t0 = 0.0
    formants = ((500, 200), (1500, 300), (2500, 400))
    while t0 < scene.seconds - 0.5:
        length = rng.uniform(0.15, 0.35)
        a, b = int(t0 * scene.sr), int((t0 + length) * scene.sr)
        tt = np.arange(b - a) / scene.sr
        env = np.hanning(b - a)
        if rng.random() < 0.75:
            f0 = rng.uniform(100, 200) * (1 + rng.uniform(-0.1, 0.1) * tt / length)
            phase = 2 * np.pi * np.cumsum(f0) / scene.sr
            for h in range(1, int(5000 / f0.mean())):
                f = h * f0.mean()
                gain = sum(np.exp(-0.5 * ((f - c) / w) ** 2) for c, w in formants) + 0.01
                out[a:b] += gain * np.sin(h * phase) * env
        else:
            noise = np.fft.rfft(rng.standard_normal(b - a))
            freqs = np.fft.rfftfreq(b - a, 1 / scene.sr)
            noise[(freqs < 3000) | (freqs > 8000)] = 0
            out[a:b] += 0.5 * np.fft.irfft(noise, b - a) * env
        t0 += length + (rng.uniform(0.3, 1.2) if rng.random() < 0.15 else rng.uniform(0.0, 0.08))
    return out


def _echo(x, sr, rng):
    out = x.copy()
    for _ in range(3):
        d = int(rng.uniform(0.03, 0.12) * sr)
        out[d:] += 10 ** (rng.uniform(-15, -8) / 20) * x[:-d]
    return out


def _whistle_tone(sr, length, pitch, pea, rng):
    n = int(length * sr)
    t = np.arange(n) / sr
    if pea:
        rate, depth = rng.uniform(25, 40), rng.uniform(0.015, 0.035)
        freq = pitch * (1 + depth * np.sin(2 * np.pi * rate * t))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        tone = np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    else:
        tone = sum(np.sin(2 * np.pi * pitch * k * t) + 0.25 * np.sin(4 * np.pi * pitch * k * t) for k in (1.0, 1.012))
    attack, release = int(0.015 * sr), int(0.03 * sr)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return tone * env


def _fans(scene, rng, level):
    n = int(scene.sr * scene.seconds)
    out = np.zeros((n, 2))
    for _ in range(int(scene.seconds / 40) + 1):
        at, length = rng.uniform(0, scene.seconds - 6), rng.uniform(2, 6)
        a = int(at * scene.sr)
        m = int(length * scene.sr)
        t = np.arange(m) / scene.sr
        burst = np.zeros((m, 2))
        for _ in range(int(rng.integers(30, 200))):
            f = rng.uniform(1000, 3200) * (1 + 0.04 * rng.uniform(-1, 1) * np.sin(2 * np.pi * rng.uniform(4, 7) * t
                                                                                 + rng.uniform(0, 6)))
            f *= 1 + rng.uniform(-0.05, 0.05) * t / length
            tone = np.sin(2 * np.pi * np.cumsum(f) / scene.sr) * np.hanning(m)
            pan = rng.uniform(0, 1)
            burst += np.stack([tone * (1 - pan), tone * pan], axis=1)
        out[a:a + m] += burst / _rms(burst) * level
    for _ in range(int(scene.seconds / 30) + 1):
        at = rng.uniform(0, scene.seconds - 1)
        m = int(0.6 * scene.sr)
        t = np.arange(m) / scene.sr
        f = 1500 + 1500 * np.sin(np.pi * t / 0.6)
        tone = np.sin(2 * np.pi * np.cumsum(f) / scene.sr) * np.hanning(m)
        pan = rng.uniform(0, 1)
        a = int(at * scene.sr)
        out[a:a + m] += np.stack([tone * (1 - pan), tone * pan], axis=1) / _rms(tone) * level * 2
    return out


def make(scene: Scene) -> Made:
    rng = np.random.default_rng(scene.seed)
    crowd = _crowd(scene, rng)
    level = _rms(crowd)
    sound = crowd.copy()
    if scene.commentary_db is not None:
        voice = _echo(_voice(scene, rng), scene.sr, rng)
        voice *= level * 10 ** (scene.commentary_db / 20) / _rms(voice)
        sound += np.stack([voice, 0.93 * voice], axis=1)      # almost, not exactly, centred
    pitch = float(rng.uniform(2400, 3800))
    pea = bool(rng.random() < 0.5)
    made = Made(sound=sound, pitch=pitch)
    t = rng.uniform(2, 8)
    while t < scene.seconds - 3:
        kind = "long" if rng.random() < 0.3 else "short"
        length = rng.uniform(0.7, 1.6) if kind == "long" else rng.uniform(0.12, 0.45)
        tone = _echo(_whistle_tone(scene.sr, length + 0.2, pitch, pea, rng), scene.sr, rng)
        tone *= level * 10 ** (scene.whistle_db / 20) / _rms(tone[: int(length * scene.sr)])
        pan = rng.uniform(0.0, 1.0)
        delay = int(rng.uniform(0.0, 0.0015) * scene.sr)
        a = int(t * scene.sr)
        left = tone * np.sqrt(1 - pan)
        right = np.concatenate([np.zeros(delay), tone * np.sqrt(pan)])[: len(tone)]
        sound[a:a + len(tone), 0] += left
        sound[a:a + len(tone), 1] += right
        made.onsets.append(round(float(t), 3))
        made.kinds.append(kind)
        made.lengths.append(round(float(length), 3))
        t += length + rng.exponential(60.0 / scene.whistles_per_minute) + 1.0
    if scene.fans:
        sound += _fans(scene, rng, level * 10 ** (-6 / 20))
    loud = level * 10 ** (scene.whistle_db / 20)

    def put(at, tone, pan):
        a = int(at * scene.sr)
        piece = tone[: max(0, len(sound) - a)]
        sound[a:a + len(piece), 0] += piece * np.sqrt(1 - pan)
        sound[a:a + len(piece), 1] += piece * np.sqrt(pan)

    for _ in range(rng.poisson(scene.blips_per_minute * scene.seconds / 60)):
        at, length, f = rng.uniform(0, scene.seconds - 1), rng.uniform(0.1, 0.8), np.exp(rng.uniform(np.log(1500), np.log(6000)))
        t = np.arange(int(length * scene.sr)) / scene.sr
        tone = np.sin(2 * np.pi * f * t) * np.hanning(len(t)) ** 0.3
        put(at, tone / _rms(tone) * loud * 10 ** (rng.uniform(-5, 5) / 20), rng.uniform(0, 1))
        made.blips.append(round(float(at), 3))
    if scene.sting_pitch:
        times = [o + rng.uniform(3, 6) for o in made.onsets if rng.random() < scene.sting_after_whistle]
        times += list(rng.uniform(0, scene.seconds - 1, rng.poisson(scene.sting_per_minute * scene.seconds / 60)))
        t = np.arange(int(0.35 * scene.sr)) / scene.sr
        sting = np.sin(2 * np.pi * scene.sting_pitch * t) * np.hanning(len(t)) ** 0.3
        for at in sorted(times):
            if at < scene.seconds - 0.5:
                put(at, sting / _rms(sting) * loud, 0.5)
                made.stings.append(round(float(at), 3))
    made.sound = sound / (np.abs(sound).max() + 1e-9) * 0.9
    return made


def second_microphone(made: Made, scene: Scene, whistle_db: float = -20.0, commentary_db: Optional[float] = None,
                      seed: int = 1) -> np.ndarray:
    """Another recording of the same moments: the same whistles (times, lengths, pitch), its own crowd,
    by default no commentators, and the whistles at their own level. [n, 2], like Made.sound."""
    other = Scene(sr=scene.sr, seconds=scene.seconds, whistle_db=whistle_db, commentary_db=commentary_db,
                  fans=scene.fans, whistles_per_minute=0.0, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    crowd = _crowd(other, rng)
    level = _rms(crowd)
    sound = crowd.copy()
    if commentary_db is not None:
        voice = _echo(_voice(other, rng), scene.sr, rng)
        voice *= level * 10 ** (commentary_db / 20) / _rms(voice)
        sound += np.stack([voice, 0.93 * voice], axis=1)
    pea = bool(rng.random() < 0.5)
    for onset, length in zip(made.onsets, made.lengths):
        tone = _echo(_whistle_tone(scene.sr, length + 0.2, made.pitch, pea, rng), scene.sr, rng)
        tone *= level * 10 ** (whistle_db / 20) / _rms(tone[: int(length * scene.sr)])
        a = int(onset * scene.sr)
        piece = tone[: max(0, len(sound) - a)]
        sound[a:a + len(piece)] += piece[:, None] * np.sqrt(0.5)
    if scene.fans:
        sound += _fans(other, rng, level * 10 ** (-6 / 20))
    return sound / (np.abs(sound).max() + 1e-9) * 0.9
