"""How well does the whistle detector work on real broadcast sound? SoccerNet matches only.

Needs the matches fetched by scripts/fetch_soccernet_audio.py (default folder ~/soccernet). Never
run on the competition match: its settings are chosen here, on other matches.

For each mix (mid, side, both together) it prints, over all fetched halves, the measures explained
in mtah/whistle_eval.py: how many fouls, offsides and kick-offs get a detection in their window
against what chance would give, false alarms per minute in quiet play, how often ball-outs get one,
the typical delay after a foul, and how many detections share the most common pitch ("one voice").

Features are computed once per half and kept beside the sound (whistle_features.npz), so trying
other event rules is quick.

    python3 experiments/whistle_soccernet.py [--dest ~/soccernet]
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import audio  # noqa: E402
from mtah import whistle  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from mtah.whistle_eval import combine, judge_half, read_labels  # noqa: E402

OUT = os.path.join("runs", "whistle_soccernet.json")
VERSION = f"{whistle.WINDOW_S}-{whistle.HOP_S}-{whistle.BAND}-{whistle.FIELDS}"


def half_features(path: str) -> dict:
    cache = path.replace("_audio.flac", "_whistle_features.npz")
    if os.path.exists(cache):
        with np.load(cache) as z:
            if str(z["version"]) == VERSION:
                return {k: z[k] for k in z.files}
    stream = whistle.FeatureStream(audio.SR)
    samples, mono = 0, True
    for chunk in audio.chunks(path):
        stream.push(chunk)
        samples += len(chunk)
        mono = mono and bool(np.array_equal(chunk[:, 0], chunk[:, 1]))
    out = dict(stream.result(), seconds=np.array(samples / audio.SR), mono=np.array(mono), version=np.array(VERSION))
    np.savez_compressed(cache, **out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=os.path.expanduser("~/soccernet"))
    args = parser.parse_args()
    os.chdir(ROOT)
    halves = sorted(glob.glob(os.path.join(args.dest, "**", "*_audio.flac"), recursive=True))
    if not halves:
        sys.exit(f"no fetched sound under {args.dest}: run scripts/fetch_soccernet_audio.py first")
    rules = whistle.EventRules()
    parts = {m: [] for m in ("mid", "side", "both")}
    voices = {m: [] for m in parts}
    mono_halves = 0
    for path in halves:
        folder, half = os.path.dirname(path), int(os.path.basename(path)[0])
        labels_path = os.path.join(folder, "Labels-v2.json")
        if not os.path.exists(labels_path):
            continue
        feats = half_features(path)
        mono_halves += bool(feats["mono"])
        labels = read_labels(labels_path)[half]
        for mode in parts:
            events = (whistle.combined_events(feats, audio.SR, rules) if mode == "both"
                      else whistle.find_events(feats[mode], audio.SR, rules))
            parts[mode].append(judge_half(events, labels, float(feats["seconds"])))
            voices[mode].append(whistle.one_voice(events)[1])
        print(f"  read {os.path.relpath(path, args.dest)}", flush=True)
    record = manifest(halves=len(parts["mid"]), rules=vars(rules), version=VERSION)
    print(f"\n{len(parts['mid'])} halves from {len(set(os.path.dirname(h) for h in halves))} matches"
          f" ({mono_halves} with one-channel sound, where the side mix is silent)")
    names = ("Foul", "Offside", "Kick-off", "Ball out of play")
    print(f"  {'mix':<6}" + "".join(f"{n:>22}" for n in names)
          + f"{'quiet/min':>11}{'delay':>8}{'one voice':>11}")
    record["mixes"] = {}
    for mode, items in parts.items():
        total = combine(items)
        cells = "".join((f"{total['share'][n]:.0%} (chance {total['chance'][n]:.0%})"
                         if total["share"].get(n) is not None else "none labelled").rjust(22) for n in names)
        delay = f"{total['median_timing']:.1f} s" if total["median_timing"] is not None else "-"
        print(f"  {mode:<6}{cells}{total['quiet_per_min']:>11.2f}{delay:>8}{np.mean(voices[mode]):>11.0%}")
        record["mixes"][mode] = {k: total[k] for k in ("share", "chance", "quiet_per_min", "median_timing", "labels")}
        record["mixes"][mode]["one_voice"] = float(np.mean(voices[mode]))
    save_manifest(OUT, record)
