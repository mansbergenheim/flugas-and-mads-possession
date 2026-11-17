#!/usr/bin/env python3
"""How often do our who-has-the-ball mentions and the friend's possession cues agree? Run this yourself.

First give the friend the line-ups read from the broadcast, as the roster his extractor reads
(JHugos files/llm/extract_possession.py --roster), instead of a looked-up squad list:

    python3 scripts/compare_mentions.py --export-roster

That writes <data folder>/lineups/roster_from_broadcast.csv (columns player, team = HOME/AWAY). Once
he has run his extractor with it, compare:

    python3 scripts/compare_mentions.py --ours MENTIONS.csv --theirs HIS_OUTPUT.jsonl

Both files time things in seconds into the broadcast. A claim says which team has the ball:
ours from on_ball / wins_ball (that player's team) and loses_ball (the other team); his from
PLAYER_ON_BALL / POSSESSION_GAINED (that team) and POSSESSION_LOST (the other), when he resolved the
team. Two claims are about the same moment when they are at most --within seconds apart. The
printout is shares only: no names, no times.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah.loaders import gated  # noqa: E402

OTHER = {"home": "away", "away": "home"}


def ours(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["side"] in OTHER and r["role"] in ("on_ball", "wins_ball"):
                out.append((float(r["time_s"]), r["side"]))
            elif r["side"] in OTHER and r["role"] == "loses_ball":
                out.append((float(r["time_s"]), OTHER[r["side"]]))
    return sorted(out)


def theirs(path, eligible_only):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        side = {"HOME": "home", "AWAY": "away"}.get(str(r.get("team", "")).upper())
        if side is None or (eligible_only and not r.get("eligible_for_fusion")):
            continue
        if r.get("type") in ("PLAYER_ON_BALL", "POSSESSION_GAINED"):
            out.append((float(r["start_seconds"]), side))
        elif r.get("type") == "POSSESSION_LOST":
            out.append((float(r["start_seconds"]), OTHER[side]))
    return sorted(out)


def compare(a, b, within):
    """For each claim in a: is there a claim in b nearby, and does the nearest one name the same team?"""
    tb = np.array([t for t, _ in b])
    near = same = 0
    for t, side in a:
        if not len(tb):
            break
        k = int(np.argmin(np.abs(tb - t)))
        if abs(tb[k] - t) <= within:
            near += 1
            same += b[k][1] == side
    return near, same


def pct(x, n):
    return f"{100 * x / n:.0f}%" if n else "n/a"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export-roster", action="store_true")
    ap.add_argument("--ours")
    ap.add_argument("--theirs")
    ap.add_argument("--within", type=float, default=3.0, help="seconds apart for two claims to be one moment")
    ap.add_argument("--all", action="store_true", help="use all his cues, not only those he marked eligible")
    args = ap.parse_args()
    root = os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    if args.export_roster:
        folder = os.path.join(gated.data_dir(root), "lineups")
        data = json.load(open(os.path.join(folder, "lineups.json"), encoding="utf-8"))
        target = os.path.join(folder, "roster_from_broadcast.csv")
        with open(target, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["player", "team"])
            n = 0
            for side in ("home", "away"):
                for group in ("starting", "others"):
                    for p in data[side][group]:
                        w.writerow([p["name"].title(), side.upper()])
                        n += 1
        print(f"wrote {target}: {n} players from the broadcast's line-up graphics (keep it on this computer;"
              " pass it to his extractor with --roster)")
        sys.exit(0)
    if not (args.ours and args.theirs):
        sys.exit("give --ours and --theirs, or --export-roster")
    a, b = ours(args.ours), theirs(args.theirs, not args.all)
    near_a, same_a = compare(a, b, args.within)
    near_b, same_b = compare(b, a, args.within)
    print(f"ours    {len(a)} who-has-the-ball claims; {pct(near_a, len(a))} have one of his within {args.within:.0f} s; "
          f"of those, {pct(same_a, near_a)} name the same team")
    print(f"his     {len(b)} claims{'' if args.all else ' marked eligible'}; {pct(near_b, len(b))} have one of ours within "
          f"{args.within:.0f} s; of those, {pct(same_b, near_b)} name the same team")
    if near_a:
        verdict = ("they agree well" if same_a >= 0.85 * near_a else
                   "they agree more often than not" if same_a >= 0.6 * near_a else
                   "they often disagree: look at the line-ups and the team sides before trusting either")
        print(f"VERDICT {verdict}")
