"""Is the referee's whistle heard in the match's two soundtracks, and do they agree? Run this yourself.

    python3 scripts/check_whistles.py

Reads, all already on this computer outside the workspace:

    <data folder>/tactical/BAR-ATH_tactical.mp4             its sound
    <data folder>/broadcast/BAR-ATH_broadcast.mp4           its sound (or --broadcast for a sound file)
    <data folder>/alignment/broadcast_alignment.npz         from scripts/align_broadcast.py
    <data folder>/candidate_check/candidate_trees_tcn.csv   from scripts/check_candidate.py (or --labels)

and writes <data folder>/whistle/whistle_events.npz (about the real match: it stays on this computer).

Safety first: false alarms matter far more than missed whistles, so one soundtrack is never trusted
alone. A whistle counts only when both soundtracks hear it at the same moment and pitch, and the
script measures how many such agreements chance alone would give on this match. Any later use may
only move the start of a stoppage the candidate already has, never create one.

What it does. The whistle detector (mtah/narrowband.py) learns the referee's pitch from the moments the
candidate says play stopped, each half from the other half, then listens only at that pitch. Its
settings were fixed on 41 SoccerNet matches and are not changed here. It does this for each
soundtrack, puts the broadcast's detections on the tactical timeline through the alignment, and counts
how often the two hear a whistle at the same moment. Details: mtah/whistle_check.py.

It uses no answer key and nothing from the public five minutes, and changes nothing in the
submission. The candidate's labels are not the truth, so "near a stoppage" means near one the
candidate found. It prints words only, so the printout can be pasted anywhere. Decoding the sound
takes a few minutes the first time; the spectra are kept in <data folder>/whistle/ for later runs.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import audio  # noqa: E402
from mtah import narrowband as nb  # noqa: E402
from mtah import submission as sub  # noqa: E402
from mtah import whistle_check as wc  # noqa: E402
from mtah.align import NO_SHIFT  # noqa: E402
from mtah.loaders import gated  # noqa: E402

SOUND_FILES = ("*.wav", "*.flac", "*.m4a", "*.aac", "*.mp3", "*.ogg")
VIDEO_FILES = ("*.mp4", "*.mov", "*.mkv")


def find_source(folder, source, explicit):
    """The video or sound file for one source, and the places looked if none is found.

    Order: the path given; the standard name; the only video, else the only sound file, in
    <data>/<source>; the only video or sound file anywhere in the data folder with the source's
    name in it (tactical, broadcast)."""
    if explicit:
        return explicit, [explicit]
    standard = gated.asset("video", folder, source)
    if os.path.exists(standard):
        return standard, [standard]
    looked = [standard]
    inside = os.path.join(folder, source)
    for patterns in (VIDEO_FILES, SOUND_FILES):
        found = sorted(p for pattern in patterns for p in glob.glob(os.path.join(inside, pattern)))
        if len(found) == 1:
            return found[0], looked
    looked.append(os.path.join(inside, "(one video or sound file)"))
    for patterns in (VIDEO_FILES, SOUND_FILES):
        found = sorted(p for pattern in patterns for p in glob.glob(os.path.join(folder, "**", pattern), recursive=True)
                       if source in os.path.basename(p).lower())
        if len(found) == 1:
            return found[0], looked
    looked.append(os.path.join(folder, f"**/*{source}* (exactly one video or sound file)"))
    return None, looked


def share_words(value):
    if value is None or not np.isfinite(value):
        return "none to count"
    for low, words in ((0.75, "most"), (0.5, "more than half"), (0.25, "between a quarter and half"),
                       (0.1, "between a tenth and a quarter"), (0.03, "a few")):
        if value >= low:
            return words
    return "hardly any"


def against_chance(share, chance):
    if share is None or chance is None or not (np.isfinite(share) and np.isfinite(chance)):
        return "cannot be told"
    if chance <= 0:
        return "far more than chance" if share > 0 else "no more than chance"
    ratio = share / chance
    return ("ten times chance or more" if ratio >= 10 else "three to ten times chance" if ratio >= 3 else
            "up to three times chance" if ratio >= 1.5 else "about what chance gives")


def rate_words(per_min):
    if per_min is None or not np.isfinite(per_min):
        return "cannot be told"
    if per_min < 0.1:
        return "fewer than one every 10 minutes"
    if per_min < 0.25:
        return "one every 4 to 10 minutes"
    if per_min < 1:
        return "one every 1 to 4 minutes"
    return "more than one a minute"


def count_words(n):
    return ("none" if n == 0 else "a handful (under 5)" if n < 5 else "some (5 to 19)" if n < 20 else
            "dozens (20 to 49)" if n < 50 else "many (50 or more)")


def chance_words(expected):
    if expected < 0.5:
        return "under one"
    if expected < 2:
        return "one or two"
    return "several (agreements here may be chance: not safe)"


def line_words(halves, settings):
    low = min(h["line"] for h in halves)
    if low >= 5:
        return "a clear whistle line in both halves"
    if low >= settings.min_line_db:
        return "a weak whistle line in both halves"
    return "no usable whistle line (detector off for this soundtrack)"


def cached_spectrum(path, cache_dir, name):
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, f"{name}_spectrum.npy")
    stamp = {"file": os.path.abspath(path), "size": os.path.getsize(path), "mtime": int(os.path.getmtime(path)),
             "win": nb.WIN, "hop": nb.HOP, "bins": len(nb.FB)}
    meta = target.replace(".npy", ".json")
    if os.path.exists(target) and os.path.exists(meta):
        with open(meta) as handle:
            if json.load(handle) == stamp:
                return np.load(target, mmap_mode="r"), True
    S = nb.spectrum(audio.chunks(path, chunk_s=60.0))
    np.save(target, S)
    with open(meta, "w") as handle:
        json.dump(stamp, handle)
    return np.load(target, mmap_mode="r"), False


def soundtrack(path, n_frames, label):
    """(offset, words) for one file: where video frame 0 sits on the sound's clock, and a line to print."""
    info = audio.describe(path)
    if not info["audio"]:
        return None, f"{label}: the file has no sound"
    starts = audio.first_times(path) if info["video"] else {"video": None, "audio": 0.0}
    if info["video"] and starts["video"] is not None and starts["audio"] is not None:
        offset = starts["video"] - starts["audio"]
        timing = ("sound and picture start together (within 0.05 s)" if abs(offset) <= 0.05 else
                  f"the picture starts {abs(offset):.2f} s {'after' if offset > 0 else 'before'} the sound (allowed for)")
    else:
        offset = 0.0
        timing = "a sound file without pictures: taken to start with the video"
    layout = info["layout"] or "unknown layout"
    return offset, f"{label}: {info['rate'] or '?'} Hz, {layout}; {timing}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    parser.add_argument("--labels", default=None, help="label file; default <data>/candidate_check/candidate_trees_tcn.csv")
    parser.add_argument("--tactical", default=None, help="tactical video or its sound; default the tactical video")
    parser.add_argument("--broadcast", default=None, help="broadcast video or its sound; default the broadcast video")
    args = parser.parse_args()
    started = time.time()
    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    try:
        folder = gated.data_dir(root)
    except RuntimeError as problem:
        sys.exit(f"1 FILES        {problem}")
    tactical_path, tactical_looked = find_source(folder, "tactical", args.tactical)
    broadcast_path, _ = find_source(folder, "broadcast", args.broadcast)
    aligned = os.path.join(folder, "alignment", "broadcast_alignment.npz")
    labels_path = args.labels or os.path.join(folder, "candidate_check", "candidate_trees_tcn.csv")
    if not os.path.exists(labels_path):
        sys.exit(f"1 FILES        label file not found: {labels_path}")
    if tactical_path is None or not os.path.exists(tactical_path):
        sys.exit("1 FILES        the tactical video (or its sound) was not found. Looked at:\n"
                 + "".join(f"               {p}\n" for p in tactical_looked)
                 + "               Point at it with --tactical \"/full/path/to/the tactical video or sound file\"")
    use_broadcast = broadcast_path is not None and os.path.exists(broadcast_path) and os.path.exists(aligned)
    try:
        labels = sub.read_submission(labels_path)
    except (OSError, ValueError) as problem:
        sys.exit(f"1 FILES        the label file could not be read: {problem}")
    n_tac = len(labels)
    print(f"1 FILES        labels read ({os.path.basename(labels_path)})"
          + ("" if use_broadcast else "; broadcast sound or alignment not found, tactical sound only"))

    tac_offset, words = soundtrack(tactical_path, n_tac, "tactical ")
    print(f"2 SOUND        {words}")
    bc_offset = None
    if use_broadcast:
        bc_offset, words = soundtrack(broadcast_path, gated.N_BROADCAST_FRAMES, "broadcast")
        print(f"               {words}")
    if tac_offset is None and bc_offset is None:
        sys.exit("VERDICT        no sound to listen to")

    cache = os.path.join(folder, "whistle")
    tactical = broadcast = None
    if tac_offset is not None:
        S, reused = cached_spectrum(tactical_path, cache, "tactical")
        tactical = wc.Track.tactical("tactical", S, tac_offset, n_tac)
        length = len(S) * nb.DT - (n_tac / gated.FPS + tac_offset)
        print(f"3 LENGTH       tactical sound {'and picture end together (within 1 s)' if abs(length) <= 1 else 'and picture differ in length by more than 1 s: the sound may drift'}")
    if bc_offset is not None:
        try:
            with np.load(aligned) as saved:
                shift = saved["shift"]
        except (OSError, KeyError, ValueError) as problem:
            sys.exit(f"1 FILES        the alignment could not be read: {problem}")
        S, _ = cached_spectrum(broadcast_path, cache, "broadcast")
        broadcast = wc.Track.broadcast("broadcast", S, bc_offset, shift, NO_SHIFT, n_tac)
        length = len(S) * nb.DT - (len(shift) / gated.FPS + bc_offset)
        print(f"               broadcast sound {'and picture end together (within 1 s)' if abs(length) <= 1 else 'and picture differ in length by more than 1 s: the sound may drift'}")

    settings = nb.Settings()
    out = wc.check(labels, sub.GRADED_BLOCKS, tactical, broadcast, settings)
    print(f"4 STOPPAGES    the candidate has {'enough' if out['stoppages'] >= 40 else 'few'} stoppages to learn from")

    if "link" in out:
        r = out["link"]["best_r"]
        same = ("largely the same sound (for example one stadium feed)" if r >= 0.6 else
                "partly the same sound" if r >= 0.3 else
                "different sounds (different microphones, or commentators on only one)")
        print(f"5 SAME SOUND?  the two soundtracks rise and fall like {same}")
        if out["lag_used_s"]:
            lag = out["lag_used_s"]
            print(f"               the broadcast's sound runs about {abs(lag):.1f} s {'late' if lag > 0 else 'early'}"
                  " against the tactical timeline; allowed for")
        else:
            print("               no lag between the soundtracks was measured or allowed for")

    step = 6
    for name, got in out["tracks"].items():
        pitch = ("the pitch could not be learned in every half" if not all(np.isfinite(h["pitch"]) for h in got["halves"])
                 else "the same pitch in both halves" if got["same_pitch_both_halves"] else "a different pitch in each half")
        print(f"{step} {name.upper():<12}{line_words(got['halves'], settings)}; {pitch}")
        print(f"               {share_words(got['at_stops'])} of the candidate's stoppages get a detection"
              f" ({against_chance(got['at_stops'], got['at_stops_chance'])})")
        print(f"               detections in quiet play: {rate_words(got['quiet_per_min'])}")
        step += 1
    if "pair" in out:
        pair = out["pair"]
        same = pair["same_pitch"]
        print(f"{step} TOGETHER     both soundtracks learned the same pitch: "
              + ", ".join(f"half {i + 1} {'yes' if s else 'no'}" for i, s in enumerate(same)))
        print(f"               {share_words(pair['together'])} of the tactical detections are heard in the broadcast"
              f" within {wc.TOGETHER_S} s ({against_chance(pair['together'], pair['together_chance'])})")
        step += 1
        safe = pair["safe"]
        usable = pair["usable_halves"]
        print(f"{step} SAFE         whistles heard in both, in a half where both have a usable line at the same pitch"
              f" ({', '.join(f'half {i + 1} {chr(10003) if u else chr(10007)}' for i, u in enumerate(usable))}):"
              f" {count_words(safe['count'])}")
        print(f"               agreements chance alone would give on this match: {chance_words(pair['chance_pairs'])}")
        if safe["count"]:
            print(f"               {share_words(safe['would_act'] / safe['count'])} of them lie within {wc.MOVE_S:g} s of a"
                  f" candidate stoppage start ({against_chance(safe['would_act'] / safe['count'], safe['would_act_chance'])});"
                  " only those could ever be used")
            print(f"               safe whistles in quiet play: {rate_words(safe['quiet_per_min'])}")
        step += 1

    target = os.path.join(cache, "whistle_events.npz")
    np.savez_compressed(target, settings=json.dumps(vars(settings)),
                        **{f"{k}_seconds": np.asarray(v, dtype=np.float64) for k, v in out["events"].items()},
                        lag_used_s=np.array(out.get("lag_used_s", 0.0)))
    print(f"{step} SAVED        {target}  (about the real match: keep it on this computer)")

    # One soundtrack is never trusted alone: on TV sound alone no setting is right much more than two
    # times in three (runs/whistle_narrowband.txt). Every verdict that says "worth" needs both.
    if "pair" not in out:
        verdict = "only one soundtrack: never trusted alone, so nothing here can be used"
    else:
        pair, safe = out["pair"], out["pair"]["safe"]
        acts = safe["would_act"] / safe["count"] if safe["count"] else 0.0
        if not any(pair["usable_halves"]):
            verdict = "the two soundtracks do not share a usable whistle line: close the whistle idea"
        elif pair["chance_pairs"] >= 1.0:
            verdict = "too many of the agreements could be chance here: not safe enough, close the whistle idea"
        elif safe["count"] < 5:
            verdict = "too few whistles that both soundtracks agree on: not worth building on"
        elif acts < 0.5 or acts < 2 * safe["would_act_chance"]:
            verdict = "the safe whistles seldom sit at the candidate's stoppages: not worth building on"
        else:
            verdict = ("safe whistles found: worth sizing as a decoder rule that may only move the start of a"
                       " stoppage the candidate already has")
    print(f"VERDICT        {verdict}   ({time.time() - started:.0f} s)")
