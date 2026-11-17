#!/usr/bin/env python3
"""Put commentary events on the match clock. Run this yourself; it prints words and a hash only.

    python3 scripts/commentary_to_match_clock.py EVENTS.csv

EVENTS.csv is the file extract_commentary_events.py wrote: its time_s column is seconds into the
BROADCAST video. This script writes EVENTS_clock.csv, the same rows with four columns added:

    tactical_frame   the frame of the tactical recording that shows that moment
    half             1 or 2 (empty outside the two halves)
    clock_s          seconds played in that half
    match_clock      the clock as a scoreboard shows it: 23:15, 45+02:10, 61:40, 90+03:05

How: scripts/align_broadcast.py has already lined the broadcast up with the tactical recording from
the players' positions (<data folder>/alignment/broadcast_alignment.npz: for each stretch of the
broadcast, tactical frame = broadcast frame + shift). The halves' first frames on the tactical
timeline are published by the organisers (mtah.loaders.gated.HALVES), and a half's first graded
frame is taken as its kick-off. So the clock comes from tracking and published numbers, not from
reading the scoreboard and not from anyone's judgement.

With --replace-time the file is written for reading instead: time_s is replaced by the match clock
(columns half and match_clock come first; the frame and second columns are left out), and the file
is named EVENTS_matchclock.csv.

Rows whose moment the broadcast alignment does not cover (before the match, half-time, time cut out
of the broadcast) keep empty clock columns. The original file is not changed, and the same inputs
always give the same bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah.loaders import gated  # noqa: E402

HALF_BASE_MIN = (0, 45)   # the clock a half starts on
HALF_FULL_MIN = (45, 90)  # past this the scoreboard shows added time: 45+mm:ss, 90+mm:ss


def match_clock(half: int, clock_s: float) -> str:
    total = int(clock_s)  # a scoreboard shows whole seconds, rounded down
    minutes, seconds = HALF_BASE_MIN[half] + total // 60, total % 60
    full = HALF_FULL_MIN[half]
    if minutes >= full:
        return f"{full}+{minutes - full:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def to_tactical(broadcast_frame: int, pieces: np.ndarray):
    """Tactical frame for a broadcast frame, or None where the alignment has no piece."""
    for first, after_last, shift in pieces:
        if first <= broadcast_frame < after_last:
            return int(broadcast_frame + shift)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", type=Path, help="CSV from extract_commentary_events.py")
    ap.add_argument("-o", "--out", type=Path, default=None, help="default: EVENTS_clock.csv beside the input")
    ap.add_argument("--replace-time", action="store_true",
                    help="replace time_s by the match clock (half, match_clock first; no frame or second columns)")
    ap.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    ap.add_argument("--alignment", type=Path, default=None,
                    help="default: <data folder>/alignment/broadcast_alignment.npz")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds to add to time_s first, if the transcript's zero is not the video's first frame "
                         "(for example an audio track that starts late)")
    args = ap.parse_args()
    out = args.out or args.events.with_name(
        args.events.stem + ("_matchclock.csv" if args.replace_time else "_clock.csv"))
    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    alignment = args.alignment or Path(gated.data_dir(root)) / "alignment" / "broadcast_alignment.npz"
    if not alignment.exists():
        sys.exit(f"{alignment} not found: run scripts/align_broadcast.py first")
    pieces = np.load(alignment)["pieces"].reshape(-1, 3)
    pieces = pieces[np.argsort(pieces[:, 0], kind="stable")]

    with args.events.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if "time_s" not in fields:
        sys.exit(f"{args.events} has no time_s column")
    added = ["tactical_frame", "half", "clock_s", "match_clock"]
    if any(name in fields for name in added):
        sys.exit(f"{args.events} already has clock columns; give the original events file")

    placed = in_half = 0
    for row in rows:
        row.update(dict.fromkeys(added, ""))
        frame = to_tactical(int(round((float(row["time_s"]) + args.offset) * gated.FPS)), pieces)
        if frame is None:
            continue
        placed += 1
        row["tactical_frame"] = str(frame)
        for half, (first, after_last) in enumerate(gated.HALVES):
            if first <= frame < after_last:
                in_half += 1
                clock_s = (frame - first) / gated.FPS
                row["half"] = str(half + 1)
                row["clock_s"] = f"{clock_s:.2f}"
                row["match_clock"] = match_clock(half, clock_s)

    k = fields.index("time_s") + 1
    if args.replace_time:
        columns = fields[:k - 1] + ["half", "match_clock"] + fields[k:]
    else:
        columns = fields[:k] + added + fields[k:]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def share(n: int) -> str:
        part = n / max(len(rows), 1)
        return "all" if n == len(rows) else "most" if part >= 0.75 else "more than half" if part >= 0.5 \
            else "less than half" if n else "none"
    print(f"alignment      {len(pieces)} piece(s) of the broadcast lined up with the tactical recording")
    print(f"placed         {share(placed)} of the events fall where the broadcast is lined up")
    print(f"on the clock   {share(in_half)} of the events fall inside one of the two halves")
    print(f"wrote          {out}")
    print(f"sha256         {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
