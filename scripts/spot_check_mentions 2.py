#!/usr/bin/env python3
"""Spot-check the local model's roles for the commentary mentions. Run this yourself, in your own terminal.

    python3 scripts/spot_check_mentions.py                      # 20 random mentions
    python3 scripts/spot_check_mentions.py -n 30 --seed 2

For each mention you see the subtitle line before, the line itself (the name in >> <<) and the line
after, and the model's role. Answer with one key and Enter:

    b  live play, and this player has the ball or is playing it now
    p  live play, but this player is not on the ball (making a run, being marked, waiting)
    n  not live play: statistics, history, a stoppage, the line-up, a substitution
    r  a replay
    ?  can't tell

At the end it prints a table of counts only (the model's role against your answer). That table holds
no names, times or words of the commentary: paste it to Claude. Nothing is saved or sent anywhere.
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_commentary_events import load_segments  # noqa: E402

DATA = os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
ANSWERS = {"b": "on the ball", "p": "live, not on ball", "n": "not live", "r": "replay", "?": "can't tell"}


def show(text, said):
    k = text.lower().find(said.lower())
    return text if k < 0 else f"{text[:k]}>>{text[k:k + len(said)]}<<{text[k + len(said):]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mentions", default=os.path.join(DATA, "commentary", "possession_mentions.csv"))
    ap.add_argument("--srt", default=os.path.join(HERE, "..", "..", "private-output", "commentary", "BAR-ATH_broadcast.srt"))
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.mentions, newline="", encoding="utf-8")))
    segs = load_segments(Path(args.srt)) if os.path.exists(args.srt) else None
    if segs is None:
        print(f"(transcript not found at {args.srt}: showing the line alone; give --srt for context)")
    picked = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    tally = Counter()
    for k, r in enumerate(picked, 1):
        print(f"\n--- {k}/{len(picked)}   {int(float(r['time_s']) // 60)}:{int(float(r['time_s']) % 60):02d} "
              f"into the broadcast   model says: {r['role']}")
        i = int(r["segment"].lstrip("s"))
        if segs:
            if i > 0:
                print(f"    {segs[i - 1]['text']}")
            print(f"  > {show(segs[i]['text'], r['said'])}")
            if i + 1 < len(segs):
                print(f"    {segs[i + 1]['text']}")
        else:
            print(f"  > {show(r['line'], r['said'])}")
        while True:
            a = input("  b / p / n / r / ? > ").strip().lower()
            if a in ANSWERS:
                break
        tally[(r["role"], a)] += 1
    roles = sorted({role for role, _ in tally})
    print("\nPASTE THIS (counts only)")
    print(f"{'model says':<14}" + "".join(f"{ANSWERS[a]:>20}" for a in ANSWERS))
    for role in roles:
        print(f"{role:<14}" + "".join(f"{tally[(role, a)]:>20}" for a in ANSWERS))
    print(f"total checked: {sum(tally.values())} of {len(rows)} mentions")
