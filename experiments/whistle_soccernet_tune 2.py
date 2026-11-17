"""Choose the whistle detector's event settings on SoccerNet matches, and check them on others.

The fetched matches (scripts/fetch_soccernet_audio.py) are split once, by name: the first half of
them choose the settings ("tune"), the second half only check them ("check"). The settings are
the event rules in mtah/whistle.py (shortest event, tonality, isolation, lowest pitch) and which
mix is used. Measured as in mtah/whistle_eval.py, with fouls and offsides counted together:

  found    share of fouls and offsides with a detection in their window
  chance   the same share if the detections fell at random
  quiet    detections per minute of quiet play, mostly false alarms

A match has perhaps 30 to 40 whistles; the decoder copes with about 10 false ones a match, which
is roughly 0.15 a minute of quiet play. So for each limit on quiet detections the best "found"
under it is shown, on the tune matches and then the same setting on the check matches, without
and with the referee-pitch filter (mtah/whistle.referee_whistles, biggest pitch group per half).

    python3 experiments/whistle_soccernet_tune.py [--dest ~/soccernet]
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import audio, whistle  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from mtah.whistle_eval import combine, judge_half, read_labels  # noqa: E402
from whistle_soccernet import half_features  # noqa: E402

OUT = os.path.join("runs", "whistle_soccernet_tune.json")
GRID = dict(min_s=(0.08, 0.15, 0.25, 0.35), tonality_db=(22.0, 28.0, 34.0), min_isolation_db=(3.0, 6.0, 9.0),
            min_pitch=(0.0, 2000.0, 2500.0))
MIXES = ("mid", "side", "both")
LIMITS = (0.15, 0.3, 0.6, 1.2)


def halves_of(folders):
    out = []
    for folder in folders:
        labels = read_labels(os.path.join(folder, "Labels-v2.json"))
        for half in (1, 2):
            path = os.path.join(folder, f"{half}_audio.flac")
            if os.path.exists(path):
                out.append((half_features(path), labels[half]))
    return out


CACHE = {}


def events_for(index, feats, rules, mix):
    """Frame-level thresholds need a new pass; length and pitch are filters on the events after it."""
    key = (index, rules.tonality_db, rules.min_isolation_db)
    if key not in CACHE:
        loose = whistle.EventRules(tonality_db=rules.tonality_db, min_isolation_db=rules.min_isolation_db,
                                   min_s=min(GRID["min_s"]), min_pitch=0.0)
        CACHE[key] = {m: whistle.find_events(feats[m], audio.SR, loose) for m in whistle.MODES}
    keep = lambda e: e["duration"] >= rules.min_s and e["pitch"] >= rules.min_pitch  # noqa: E731
    by_mix = {m: [e for e in ev if keep(e)] for m, ev in CACHE[key].items()}
    return whistle.merge_mixes(by_mix) if mix == "both" else by_mix[mix]


def measure(halves, rules, mix, referee, offset=0):
    parts = []
    for i, (feats, labels) in enumerate(halves):
        events = events_for(offset + i, feats, rules, mix)
        if referee:
            events = whistle.referee_whistles(events)[0]
        parts.append(judge_half(events, labels, float(feats["seconds"])))
    total = combine(parts)
    hit = sum(total["labels"].get(k, [0, 0])[0] for k in ("Foul", "Offside"))
    n = sum(total["labels"].get(k, [0, 0])[1] for k in ("Foul", "Offside"))
    return {"found": hit / max(n, 1), "chance": total["chance"].get("Foul", 0.0), "quiet": total["quiet_per_min"],
            "kickoff": total["share"].get("Kick-off"), "ball_out": total["share"].get("Ball out of play"),
            "delay": total["median_timing"], "n": n}


def line(tag, r):
    ko = "-" if r["kickoff"] is None else f"{r['kickoff']:.0%}"
    out = "-" if r["ball_out"] is None else f"{r['ball_out']:.0%}"
    return (f"{tag:<7}{r['found']:>7.0%}{r['chance']:>8.0%}{r['quiet']:>8.2f}{ko:>9}{out:>9}"
            f"{(r['delay'] if r['delay'] is not None else float('nan')):>7.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=os.path.expanduser("~/soccernet"))
    args = parser.parse_args()
    os.chdir(ROOT)
    folders = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(args.dest, "**", "2_audio.flac"), recursive=True)
                      if os.path.exists(os.path.join(os.path.dirname(p), "1_audio.flac"))})
    if len(folders) < 4:
        sys.exit("fetch at least four matches first")
    cut = len(folders) // 2
    tune, check = halves_of(folders[:cut]), halves_of(folders[cut:])
    print(f"{cut} matches to choose settings, {len(folders) - cut} to check them")
    results = []
    names = list(GRID)
    for values in itertools.product(*GRID.values()):
        rules = whistle.EventRules(**dict(zip(names, values)))
        for mix in MIXES:
            for referee in (False, True):
                results.append((rules, mix, referee, measure(tune, rules, mix, referee)))
    record = manifest(grid={k: list(v) for k, v in GRID.items()}, tune=[os.path.basename(f) for f in folders[:cut]],
                      check=[os.path.basename(f) for f in folders[cut:]])
    record["best"] = []
    head = f"{'':<7}{'found':>7}{'chance':>8}{'quiet':>8}{'kick-off':>9}{'ball out':>9}{'delay':>7}"
    for referee in (False, True):
        print(f"\n{'with' if referee else 'without'} the referee-pitch filter")
        for limit in LIMITS:
            fits = [r for r in results if r[2] == referee and r[3]["quiet"] <= limit]
            if not fits:
                print(f"  quiet at most {limit}/min: no setting gets there")
                continue
            rules, mix, _, got = max(fits, key=lambda r: (r[3]["found"] - r[3]["chance"], -r[3]["quiet"]))
            held = measure(check, rules, mix, referee, offset=10_000)
            setting = (f"mix {mix}, shortest {rules.min_s} s, tonality {rules.tonality_db:g} dB, "
                       f"isolation {rules.min_isolation_db:g} dB, lowest pitch {rules.min_pitch:g} Hz")
            print(f"  quiet at most {limit}/min: {setting}\n    {head}\n    {line('tune', got)}\n    {line('check', held)}")
            record["best"].append({"referee": referee, "limit": limit, "mix": mix, "rules": vars(rules),
                                   "tune": got, "check": held})
    save_manifest(OUT, record)
