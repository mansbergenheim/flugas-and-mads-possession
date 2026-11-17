"""The whistle detector on made-up stadium sound (mtah/fakestadium.py). No real sound involved.

This checks the code and compares two ideas under conditions we wrote ourselves, so it flatters
anything that matches our own assumptions (for instance, the made-up commentary is mixed almost
dead centre, which is exactly what the side mix exploits). Real quality is measured on other
matches with real sound, never here and never on the competition match.

For each condition, three scenes of two minutes each:
  found      share of the true whistles with a detected event starting within 0.2 s
  false      detected events per minute that are not inside any true whistle
  timing     median distance from the true start, for the ones found
  one voice  share of detected events at the most common pitch (the referee should dominate)
compared for the mid mix, the side mix and both together, and for full bandwidth against sound first reduced to
16 kHz, as the earlier whistle scripts did.

    python3 experiments/whistle_synthetic.py      # about a minute
"""

import os
import sys

import numpy as np
from scipy.signal import resample_poly

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah.fakestadium import Scene, make  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from mtah.whistle import combined_events, features, find_events, one_voice  # noqa: E402

SEEDS = (11, 12, 13)
SECONDS = 120.0
OUT = os.path.join("runs", "whistle_synthetic.json")
CONDITIONS = (
    ("broadcast: whistle -5 dB, commentary, fans", dict(whistle_db=-5.0)),
    ("broadcast: whistle -10 dB, commentary, fans", dict(whistle_db=-10.0)),
    ("broadcast: whistle -15 dB, commentary, fans", dict(whistle_db=-15.0)),
    ("broadcast without fans: whistle -10 dB", dict(whistle_db=-10.0, fans=False)),
    ("pitch-side only: whistle -10 dB, no commentary", dict(whistle_db=-10.0, commentary_db=None)),
)


def score(made, events):
    found, timing = 0, []
    for onset in made.onsets:
        gaps = [abs(e["start"] - onset) for e in events]
        if gaps and min(gaps) < 0.2:
            found += 1
            timing.append(min(gaps))
    inside = [any(o - 0.2 <= e["start"] <= o + l for o, l in zip(made.onsets, made.lengths)) for e in events]
    return found, len(made.onsets), sum(not x for x in inside), timing


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(seeds=list(SEEDS), seconds=SECONDS)
    print(f"  {'':<48}{'mix':>6}{'rate':>8}{'found':>8}{'false/min':>11}{'timing':>9}{'one voice':>11}")
    record["rows"] = {}
    for label, how in CONDITIONS:
        for rate in (48000, 16000):
            for mode in ("mid", "side", "both"):
                found = total = false = 0
                timing, voice = [], []
                for seed in SEEDS:
                    made = make(Scene(seconds=SECONDS, seed=seed, **how))
                    sound = made.sound if rate == 48000 else resample_poly(made.sound, 1, 3, axis=0)
                    feats = features(sound, rate)
                    events = combined_events(feats, rate) if mode == "both" else find_events(feats[mode], rate)
                    f, t, x, tm = score(made, events)
                    found, total, false, timing = found + f, total + t, false + x, timing + tm
                    voice.append(one_voice(events)[1])
                cells = {"found": found / total, "false_per_min": false / (len(SEEDS) * SECONDS / 60),
                         "timing_s": float(np.median(timing)) if timing else None, "one_voice": float(np.mean(voice))}
                record["rows"][f"{label} | {mode} | {rate}"] = cells
                tm = f"{cells['timing_s']:.2f} s" if timing else "-"
                print(f"  {label:<48}{mode:>6}{rate // 1000:>6}k{cells['found']:>8.0%}{cells['false_per_min']:>11.1f}"
                      f"{tm:>9}{cells['one_voice']:>11.0%}", flush=True)
    save_manifest(OUT, record)
