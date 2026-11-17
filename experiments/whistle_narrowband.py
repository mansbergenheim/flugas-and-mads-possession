"""A whistle detector that first learns the referee's pitch, then listens only there. SoccerNet matches only.

Why. On broadcast sound the plain detector (mtah/whistle.py) looks for the strongest tonal peak in
each moment, and the commentators' voices usually beat a distant whistle; on 12 SoccerNet LaLiga
matches it found fouls barely more often than chance. But averaging the spectrum over all fouls
of a match against quiet play shows, in some matches, a narrow line at one pitch: the referee's
whistle. So:

The detector itself is mtah/narrowband.py; this script chooses and checks its settings.

1. learn the line: the spectrum from 1 s before to 3 s after each stoppage, against quiet play;
   the frequency between 2 and 5 kHz that rises most above its own surroundings is the referee's
   pitch, and how far it rises says whether there is a usable whistle at all ("line" dB);
2. listen there: a whistle is where the power within 1.5% of that pitch stands far above the ring
   from 4% to 12% away, for long enough.

Each half's pitch is learned on the OTHER half's fouls, so the half being scored never helps to
find its own whistles. On the competition match the stoppages would come from the candidate's
labels instead of SoccerNet's fouls. Settings are chosen on the even-numbered matches and checked
on the odd-numbered ones. Matches whose learned line is too weak are left out, as the detector
would be switched off for them. Settings are chosen precision-first: see the rule above run().

Spectra are cached beside the sound (_narrowband_spec.npy) and only one match is held in memory at a
time; what is kept per half is the learned pitch, the line strength and the contrast series.

    python3 experiments/whistle_narrowband.py      # needs scripts/fetch_soccernet_audio.py first
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import audio  # noqa: E402
from mtah import narrowband as nb  # noqa: E402
from mtah.whistle_eval import combine, judge_half, read_labels  # noqa: E402

DT = nb.DT
JOIN_S = nb.Settings().join_s
OUT = os.path.join(ROOT, "runs", "whistle_narrowband.txt")


def spectro(path):
    cache = path.replace("_audio.flac", "_narrowband_spec.npy")
    if not os.path.exists(cache):
        np.save(cache, nb.spectrum(audio.chunks(path, chunk_s=60.0)))
    return np.load(cache, mmap_mode="r")


learn_line, contrast = nb.learn_line, nb.contrast


def events(c, thr, min_frames=4):
    return nb.find(c, nb.Settings(threshold_db=thr, min_s=min_frames * DT, join_s=JOIN_S))


def prepare(folder):
    """Per half: pitch and line learned on the other half's fouls, and this half's contrast there."""
    labels = read_labels(folder + "/Labels-v2.json")
    spectra, quiet, stops = {}, {}, {}
    for h in (1, 2):
        S = spectro(f"{folder}/{h}_audio.flac")
        t = np.arange(len(S)) * DT
        q = np.ones(len(S), bool)
        for x in labels[h]:
            q &= np.abs(t - x["time"]) > 10
        spectra[h], quiet[h] = S, q
        stops[h] = [x["time"] for x in labels[h] if x["label"] in ("Foul", "Offside")]
    halves = []
    for h in (1, 2):
        other = 3 - h
        pitch, height = learn_line(spectra[other], stops[other], quiet[other])
        S = spectra[h]
        halves.append(dict(pitch=pitch, height=height, c=contrast(S, pitch), labels=labels[h], seconds=len(S) * DT))
    return halves


def switched_on(halves, gate, agree):
    """Use the detector on a match only if both halves show a line, and (if agree is set) at nearly the same pitch."""
    if min(h["height"] for h in halves) < gate:
        return False
    return agree is None or abs(halves[0]["pitch"] / halves[1]["pitch"] - 1) <= agree


# The rule for choosing settings (16 Nov 2026). False alarms matter far more than misses. A first rule
# (at most 0.05 quiet detections a minute and 90% of detections near a labelled stoppage) left a
# detector that hears under one whistle a match: on TV sound alone no setting that hears a useful
# number of whistles is right more than about two times in three (see the table this prints). So one
# soundtrack is never used alone. This setting is the first of three safety layers: each soundtrack
# must give at most 0.1 detections a minute in quiet play and have at least 60% of its detections near
# a labelled stoppage, and among those the setting finding the most fouls and offsides per match wins.
# Layer two: a whistle counts only when both soundtracks hear it at the same moment and pitch
# (mtah/whistle_check.py). Layer three: it may only move the start of a stoppage the candidate already
# has, never create one (the decoder, later).
MAX_QUIET_PER_MIN = 0.10
MIN_PRECISION = 0.60


def run(subset, thr, min_frames, gate, agree=None):
    tot, per = [], []
    for name, halves in subset:
        if not switched_on(halves, gate, agree):
            per.append((name, None))
            continue
        parts = [judge_half(events(h["c"], thr, min_frames), h["labels"], h["seconds"]) for h in halves]
        c = combine(parts)
        tot += parts
        hit = sum(c["labels"].get(k, [0, 0])[0] for k in ("Foul", "Offside"))
        n = sum(c["labels"].get(k, [0, 0])[1] for k in ("Foul", "Offside"))
        prec = "-" if c["precision"] is None else f"{c['precision']:.0%}"
        per.append((name, f"{hit}/{n} found (chance {c['chance'].get('Foul', 0):.0%}), {c['events']} detections, "
                          f"{prec} near a stoppage, {c['unexplained_per_hour']:.1f} unexplained/h, quiet {c['quiet_per_min']:.2f}/min"))
    if not tot:
        return None, per
    c = combine(tot)
    hit = sum(c["labels"].get(k, [0, 0])[0] for k in ("Foul", "Offside"))
    n = sum(c["labels"].get(k, [0, 0])[1] for k in ("Foul", "Offside"))
    matches = sum(p is not None for _, p in per)
    return dict(found=hit / n, chance=c["chance"].get("Foul", 0), quiet=c["quiet_per_min"], n=n, matches=matches,
                hits_per_match=hit / len(subset), detections=c["events"],
                precision=c["precision"] if c["precision"] is not None else float("nan"),
                precision_chance=c["precision_chance"] if c["precision_chance"] is not None else float("nan"),
                unexplained=c["unexplained_per_hour"]), per


def describe(r, n_subset):
    return (f"found {r['found']:.0%} of fouls/offsides (chance {r['chance']:.0%}), {r['hits_per_match']:.1f} a match;"
            f" {r['precision']:.0%} of detections near a stoppage (chance {r['precision_chance']:.0%});"
            f" {r['unexplained']:.1f} unexplained/h; quiet {r['quiet']:.3f}/min; {r['matches']} of {n_subset} matches on")


if __name__ == "__main__":
    folders = sorted({os.path.dirname(p) for p in glob.glob(os.path.expanduser("~/soccernet/**/1_audio.flac"), recursive=True)
                      if all(os.path.exists(os.path.join(os.path.dirname(p), f"{h}_audio.flac"))
                             and os.path.getsize(os.path.join(os.path.dirname(p), f"{h}_audio.flac")) > 50e6 for h in (1, 2))})
    data = []
    for i, folder in enumerate(folders):
        data.append((os.path.basename(folder)[:40], prepare(folder)))
        print(f"  [{i + 1}/{len(folders)}] {os.path.basename(folder)}", file=sys.stderr, flush=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731
    heights = np.array([min(h["height"] for h in halves) for _, halves in data])
    say(f"{len(data)} SoccerNet matches with the sound of both halves")
    say("weaker half's learned line: " + ", ".join(f"{(heights >= g).sum()} at {g:g} dB or more" for g in (2, 3, 4, 5, 6)))
    tune, check = data[0::2], data[1::2]
    rows = []
    for thr in (16.0, 20.0, 24.0, 28.0, 32.0):
        for mf in (6, 10, 16, 24):
            for gate in (2.0, 3.0, 4.0, 5.0):
                r, _ = run(tune, thr, mf, gate)
                if r:
                    rows.append((thr, mf, gate, r))
    say()
    say("all 41 matches, every setting with line >= 2 dB: detections and hits per match, share of detections near a")
    say("labelled stoppage (precision) against chance, unexplained detections per hour, quiet detections per minute")
    say(f"  {'setting':<18}{'det/match':>10}{'hits/match':>11}{'precision':>10}{'chance':>7}{'unexpl/h':>9}{'quiet/min':>10}")
    for thr in (14.0, 16.0, 20.0, 24.0, 28.0):
        for mf in (6, 10, 16, 24):
            r, _ = run(data, thr, mf, 2.0)
            if r:
                say(f"  {thr:g} dB {mf * DT:.2f} s{'':<6}{r['detections'] / len(data):>10.1f}{r['hits_per_match']:>11.1f}"
                    f"{r['precision']:>10.0%}{r['precision_chance']:>7.0%}{r['unexplained']:>9.1f}{r['quiet']:>10.3f}")
    safe = [row for row in rows if row[3]["quiet"] <= MAX_QUIET_PER_MIN and row[3]["precision"] >= MIN_PRECISION]
    say()
    say(f"rule: on the tune matches at most {MAX_QUIET_PER_MIN} detections a minute in quiet play and at least"
        f" {MIN_PRECISION:.0%} of detections near a labelled stoppage; then the most fouls/offsides found per match")
    say(f"{len(safe)} of {len(rows)} settings pass on the tune matches (even-numbered)")
    if safe:
        safe.sort(key=lambda row: (row[3]["hits_per_match"], -row[3]["quiet"]), reverse=True)
        pick = safe[0]
    else:
        say("none passes: falling back to the setting with the fewest quiet detections that still finds 3x chance")
        ok = [row for row in rows if row[3]["found"] >= 3 * max(row[3]["chance"], 1e-9)] or rows
        ok.sort(key=lambda row: (row[3]["quiet"], -row[3]["hits_per_match"]))
        pick = ok[0]
    for thr, mf, gate, r in (safe[:8] if safe else [pick]):
        say(f"  {thr:g} dB, at least {mf * DT:.2f} s, line >= {gate:g} dB: {describe(r, len(tune))}")
    thr, mf, gate, _ = pick
    say()
    say(f"chosen: threshold {thr:g} dB, one unbroken piece of at least {mf * DT:.2f} s, line >= {gate:g} dB"
        f" (pieces within {JOIN_S:g} s joined)")
    for tag, subset in (("tune", tune), ("check", check)):
        r, per = run(subset, thr, mf, gate)
        say(f"{tag}: " + (describe(r, len(subset)) if r else "no match had a clear line"))
        for name, text in per:
            say(f"    {name}: {text or 'no clear whistle line, left out'}")
    with open(OUT, "w") as handle:
        handle.write("\n".join(lines) + "\n")
