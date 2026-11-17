#!/usr/bin/env python3
"""Where do the final submission files disagree? Run this yourself; keep its output on this computer.

    python3 scripts/compare_final.py ~/mtah-data/final/without_commentary/Flugas_and_Mads.csv \
        ~/mtah-data/final/with_commentary/Flugas_and_Mads.csv \
        ~/mtah-data/final/with_commentary_strong/Flugas_and_Mads.csv

Prints, for every pair, how many graded frames differ and in how many stretches, then every stretch
where any two files disagree: the half, the match clock, how long, each file's label, and whether it
lies inside the public five minutes.

    ... --score     also prints spellF1 and macroF1 on the public five minutes, with our copy of the
                    scorer (tested against the organisers' results)

The public window is for checking the pipeline, not for choosing: the rules forbid selecting anything
on it. Decide which file to submit first (submission.json), then look, if you want to.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import metrics as m  # noqa: E402
from mtah.loaders.gated import HALVES  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402
from mtah.submission import read_submission  # noqa: E402

FPS = 25
PUBLIC = os.path.join(ROOT, "..", "official-repository", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
NAME = {H: "home", A: "away", D: "dead", X: "X"}


def clock(frame: int) -> str:
    for h, (start, stop) in enumerate(HALVES):
        if start <= frame < stop:
            minutes, seconds = divmod((frame - start) / FPS + 45 * 60 * h, 60)
            return f"half {h + 1}  {int(minutes):02d}:{seconds:04.1f}"
    return "outside the halves"


def stretches(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def read_public(path: str, n: int) -> np.ndarray:
    lookup = {"H": H, "A": A, "D": D}
    with open(path, encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f][1:]
    if len(rows) != n:
        sys.exit(f"the public label file has {len(rows)} rows, the submissions {n}")
    return np.array([lookup.get(r[1].strip().upper(), X) for r in rows], dtype=np.int8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--score", action="store_true", help="also score each file on the public five minutes")
    ap.add_argument("--public", default=PUBLIC, help="the organisers' public label file")
    args = ap.parse_args()
    names = [os.path.basename(os.path.dirname(os.path.abspath(f))) or f for f in args.files]
    codes = [read_submission(f) for f in args.files]
    n = len(codes[0])
    if any(len(c) != n for c in codes):
        sys.exit("the files have different lengths")
    graded = np.zeros(n, dtype=bool)
    for start, stop in HALVES:
        graded[start:stop] = True
    window = np.zeros(n, dtype=bool)
    truth = read_public(args.public, n) if os.path.exists(args.public) else None
    if truth is not None:
        window = truth != X

    print("About the real match: keep this on this computer, don't paste it into an AI tool.\n")
    print("1. PAIRS")
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            diff = graded & (codes[i] != codes[j])
            print(f"   {names[i]} vs {names[j]}: {diff.sum()} frames ({diff.sum() / FPS:.1f} s) in "
                  f"{len(stretches(diff))} stretches; {int((diff & window).sum())} of them in the public window")

    any_diff = graded & np.any([c != codes[0] for c in codes[1:]], axis=0)
    print(f"\n2. WHERE THEY DISAGREE  (labels in the order: {', '.join(names)})")
    for a, b in stretches(any_diff):
        labels = " / ".join("+".join(sorted({NAME[int(v)] for v in c[a:b]})) for c in codes)
        before = NAME[int(codes[0][a - 1])] if a > 0 else "?"
        after = NAME[int(codes[0][b])] if b < n else "?"
        flag = "  PUBLIC WINDOW" if window[a:b].any() else ""
        print(f"   {clock(int(a))}  {(b - a) / FPS:4.1f} s   {labels:<30} (around it: {before} ... {after}){flag}")
    if not any_diff.any():
        print("   nowhere: the files are identical on every graded frame")

    if args.score:
        print("\n3. PUBLIC FIVE MINUTES  (a pipeline check, not for choosing)")
        if truth is None:
            print(f"   the public label file was not found: {args.public}")
        else:
            inside = int((any_diff & window).sum())
            for name, c in zip(names, codes):
                r = m.evaluate(truth, c)
                print(f"   {name:<28} spellF1 {r['spell_f1']:.3f}   macroF1 {r['macro_f1']:.3f}")
            print(f"   the files differ on {inside} frames inside the window"
                  + (", so their scores there can only be the same" if inside == 0 else ""))
