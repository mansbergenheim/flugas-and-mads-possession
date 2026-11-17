"""The whistle detector on real whistles: two RoboCup recordings from TU Dortmund. Not football broadcasts.

The recordings (experiments/whistle_robocup_labels.json says where they come from and how the
whistles were marked) hold real referee whistles among hall noise, speech, robot sounds and a
loud crowd. One recording is for trying settings ("tune"), the other only for checking them
("check"), so a setting that merely fits the first one shows up as a drop on the second.

    found    share of the marked whistles with a detection starting within 0.4 s
    false    detections not within 0.4 s of a marked whistle (nor of an unsure mark), per minute

Needs the two WAV files in ~/whistle-data/tudortmund (or --folder):
    https://data.tu-dortmund.de/api/access/datafile/9729   Testing_20221104_091937_2.wav
    https://data.tu-dortmund.de/api/access/datafile/9892   Testing_20221104_101430_4_cut.wav

    python3 experiments/whistle_robocup.py
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import whistle  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402

LABELS = os.path.join(os.path.dirname(__file__), "whistle_robocup_labels.json")
TOLERANCE = 0.4
OUT = os.path.join("runs", "whistle_robocup.json")


def score(events, whistles, unsure, seconds):
    starts = np.array([e["start"] for e in events])
    found = sum(bool(len(starts)) and np.abs(starts - w).min() <= TOLERANCE for w in whistles)
    marks = np.array(list(whistles) + list(unsure))
    false = sum(np.abs(marks - s).min() > TOLERANCE for s in starts) if len(marks) else len(starts)
    return found, len(whistles), false, seconds


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    x /= np.abs(x).max() + 1e-12
    return sr, x[:, :2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default=os.path.expanduser("~/whistle-data/tudortmund"))
    args = parser.parse_args()
    os.chdir(ROOT)
    with open(LABELS) as handle:
        marks = json.load(handle)["files"]
    record = manifest(tolerance=TOLERANCE, rules=vars(whistle.EventRules()))
    record["rows"] = {}
    print(f"  {'recording':<10}{'mix':>6}{'found':>12}{'false/min':>11}{'one voice':>11}")
    for name, info in marks.items():
        sr, sound = load(os.path.join(args.folder, name))
        feats = whistle.features(sound, sr)
        for mode in ("mid", "side", "both"):
            events = (whistle.combined_events(feats, sr) if mode == "both" else whistle.find_events(feats[mode], sr))
            found, total, false, seconds = score(events, info["whistles"], info["unsure"], len(sound) / sr)
            voice = whistle.one_voice(events)[1]
            record["rows"][f"{info['role']} | {mode}"] = {"found": found, "of": total, "false": false,
                                                          "false_per_min": 60 * false / seconds, "one_voice": voice}
            print(f"  {info['role']:<10}{mode:>6}{found:>6} of {total:<3}{60 * false / seconds:>11.1f}{voice:>11.0%}")
    save_manifest(OUT, record)
