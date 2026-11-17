"""Line the broadcast up with the tactical recording, on the real match. Run this yourself.

    python3 scripts/align_broadcast.py

It reads five files of the organisers' download from MTAH_DATA_DIR (or ~/mtah-data), and never a video:

    tactical/BAR-ATH_tactical_atd.xml      broadcast/BAR-ATH_broadcast_atd.xml
    tactical/BAR-ATH_tactical_atd.txt      broadcast/BAR-ATH_broadcast_atd.txt
                                           broadcast/BAR-ATH_broadcast_homography.csv

It writes

    <data folder>/alignment/broadcast_alignment.npz

which is about the real match and stays on this computer: for every tactical frame, which broadcast
frame shows that moment and whether the broadcast was on its live wide shot, on a replay, or on
something else. The printout is words only, with no measurement from the match in it, so it can be
pasted anywhere, an AI tool included.

How it works is in mtah/align.py. How well it works on made-up broadcasts of other matches is in
experiments/align_check.py.
"""

import argparse
import os
import sys
import time
from xml.etree.ElementTree import ParseError

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah.align import LIVE, NOT_COVERED, NoAlignment, align, tactical_view  # noqa: E402
from mtah.loaders import gated  # noqa: E402

FIT_BANDS = ((1.0, "tight"), (2.0, "fair"))                # metres between matched players
AGREE_BANDS = ((0.5, "clear"), (0.2, "fair"))              # share of test clips that agreed


def band(value, bands, last, higher_is_better=False):
    for limit, name in bands:
        if (value >= limit) if higher_is_better else (value < limit):
            return name
    return last


def halves_report(found, state, halves, tracked):
    """One line for each half, and whether both line up. Counted over the frames with players on the pitch."""
    lines, good = [], True
    for name, (a, b) in zip(("first", "second"), halves):
        covered = float((state[a:b] != NOT_COVERED)[tracked[a:b]].mean()) if tracked[a:b].any() else 0.0
        pieces = sum(1 for first, last, shift in found.pieces if min(last + shift, b) - max(first + shift, a) > 250)
        if covered >= 0.95 and pieces == 1:
            lines.append(f"the {name} half lines up as one unbroken piece")
        elif covered >= 0.95:
            lines.append(f"the {name} half lines up, in more than one piece (time is cut out of the broadcast inside it)")
        elif covered >= 0.5:
            lines.append(f"the {name} half lines up only in part")
            good = False
        else:
            lines.append(f"the {name} half does not line up")
            good = False
    return lines, good


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    parser.add_argument("--broadcast-frames", type=int, default=gated.N_BROADCAST_FRAMES,
                        help="frames in the broadcast; only a fake bundle needs this")
    args = parser.parse_args()
    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    started = time.time()
    try:
        needed = [gated.asset(name, root, source) for source in ("tactical", "broadcast")
                  for name in ("atd_meta", "atd_raw")] + [gated.asset("homography", root, "broadcast")]
    except RuntimeError as problem:
        sys.exit(f"1 FILES        could not be read: {problem}")
    missing = [path for path in needed if not os.path.exists(path)]
    if missing:
        sys.exit("1 FILES        not found:\n" + "".join(f"               {path}\n" for path in missing)
                 + "               They are in the organisers' download, beside the videos. No video is needed.")
    recordings = {}
    for name, load in (("tactical", lambda: gated.load_atd_match(root)),
                       ("broadcast", lambda: gated.load_broadcast(root, args.broadcast_frames))):
        try:
            recordings[name] = load()
        except (OSError, ValueError, RuntimeError, ParseError) as problem:     # ParseError: a file cut short
            sys.exit(f"1 FILES        the {name} files could not be read: {problem}")
    tactical, broadcast = recordings["tactical"], recordings["broadcast"]
    print("1 FILES        both recordings read", flush=True)

    try:
        found = align(tactical.players, broadcast["players"], broadcast["valid"], tactical.pitch, tactical.team,
                      broadcast["team"])
    except NoAlignment as problem:
        sys.exit(f"2 LINING UP    failed: {problem}\nVERDICT        do not use")
    way = {(1, 1): "the broadcast draws the pitch the same way round as the tactical recording",
           (-1, -1): "the broadcast draws the pitch turned half a circle (handled)",
           (-1, 1): "the broadcast draws the pitch mirrored end to end (handled)",
           (1, -1): "the broadcast draws the pitch mirrored side to side (handled)"}[found.flip]
    print(f"2 WAY ROUND    {way}")
    teams = {True: "the two files number the teams the same way", False: "the two files number the teams the other way round (noted)",
             None: "could not be told from the matched players"}[found.same_teams]
    print(f"3 TEAMS        {teams}")
    frame, state = tactical_view(found, tactical.n_frames)
    tracked = np.isfinite(tactical.players[:, :, 0]).sum(axis=1) >= 10
    lines, good = halves_report(found, state, tactical.halves, tracked)
    for k, line in enumerate(lines):
        print(f"{'4 PIECES' if not k else '':<15}{line}")
    fit = band(found.fit_m, FIT_BANDS, "loose")
    agree = band(found.agreed / max(found.clips, 1), AGREE_BANDS, "weak", higher_is_better=True)
    print(f"5 FIT          {fit}: how close the broadcast's players stand to the tactical ones once lined up")
    print(f"6 AGREEMENT    {agree}: how many of the test clips voted for the same line-up")
    graded = np.concatenate([np.arange(a, b) for a, b in tactical.halves])
    live = float((state[graded] == LIVE).mean())
    words = "most of the time" if live >= 0.75 else "more than half the time" if live >= 0.5 else "less than half the time"
    print(f"7 LIVE VIEW    during the two halves the broadcast is on its live wide shot {words}")

    # How much of the time without a tactical ball the broadcast could fill, in words.
    lost = ~np.isfinite(tactical.ball[graded, 0])
    there = frame[graded]
    has = (state[graded] == LIVE) & (there >= 0)
    has[has] = np.isfinite(broadcast["ball"][there[has], 0])
    filled = float(has[lost].mean()) if lost.any() else 0.0
    words = "more than half the time" if filled >= 0.5 else "between a quarter and half the time" if filled >= 0.25 else \
        "between a tenth and a quarter of the time" if filled >= 0.1 else "less than a tenth of the time"
    print(f"8 BALL         where the tactical recording has lost the ball, the broadcast has it {words}")

    folder = os.path.join(gated.data_dir(root), "alignment")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "broadcast_alignment.npz")
    np.savez_compressed(target, shift=found.shift, broadcast_state=found.state, tactical_to_broadcast=frame,
                        tactical_state=state, flip=np.array(found.flip), matrix=found.matrix,
                        pieces=np.array(found.pieces, dtype=np.int64),
                        same_teams=np.array(-1 if found.same_teams is None else int(found.same_teams)))
    print(f"9 SAVED        {target}  (about the real match: keep it on this computer)")
    verdict = "ready to use" if good and fit != "loose" and agree != "weak" else \
        "usable with care" if good else "do not use"
    print(f"VERDICT        {verdict}   ({time.time() - started:.0f} s)")
