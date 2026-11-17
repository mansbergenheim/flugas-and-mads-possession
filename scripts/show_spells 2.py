"""List what a file of labels says on the public five minutes, as possessions with their times.

Run this yourself, not through an AI tool. It prints what the model says about the real match stretch
by stretch, so its output stays on this computer: do not paste it into Claude or any other online AI.

    python3 scripts/show_spells.py                      # the file scripts/check_candidate.py wrote
    python3 scripts/show_spells.py path/to/labels.csv   # any file in the submission format

Each stretch gets a line: the match clock, the place in the tracking timeline, who has the ball, and
for how long. 75 frames of H at 25 frames a second become "Barcelona, 3.0 s". H is the home team,
Barcelona. A is the away team, Athletic Club. D is the ball out of play.

The public five minutes are the frames the organisers' public label file grades. The match clock is
counted from the first graded frame of the first half, which DATA.md gives as the kickoff.

With --answers the organisers' public labels are shown beside ours, one line for each stretch in which
neither changes, and the lines where they differ are marked. That is for understanding what the labels
mean. It is not for tuning: never change the model because of what these five minutes show.

Watching the match to understand and debug is allowed by the rules. Labelling it by hand, correcting
its tracks by hand, and rules tied to particular frames are not.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from check_candidate import FILE_NAME, PUBLIC, read_public  # noqa: E402
from mtah import submission as sub  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402

FPS = 25
WHO = {H: "Barcelona", A: "Athletic Club", D: "ball out of play", X: "no answer"}
OUT_NAME = "spells_public.txt"
WARNING = "About the real match. Keep this on this computer: do not paste it into an AI tool."


def clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes):02d}:{rest:04.1f}"


def runs(codes: np.ndarray, start: int, stop: int):
    """(first frame, the frame after the last, code) for each unbroken stretch inside start to stop."""
    part = codes[start:stop]
    bounds = np.concatenate(([0], np.flatnonzero(part[1:] != part[:-1]) + 1, [len(part)]))
    return [(start + int(a), start + int(b), int(part[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def summary(stretches, name: str) -> str:
    home = [b - a for a, b, code in stretches if code == H]
    away = [b - a for a, b, code in stretches if code == A]
    dead = [b - a for a, b, code in stretches if code == D]
    return (f"{name}: {len(home) + len(away)} possessions. Barcelona {len(home)} ({sum(home) / FPS:.0f} s), "
            f"Athletic Club {len(away)} ({sum(away) / FPS:.0f} s), ball out of play {len(dead)} times "
            f"({sum(dead) / FPS:.0f} s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    folder = os.path.join(os.path.expanduser(os.environ.get("MTAH_DATA_DIR", "~/mtah-data")), "candidate_check")
    parser.add_argument("labels", nargs="?", default=os.path.join(folder, FILE_NAME),
                        help="a file of labels in the submission format; default: the one check_candidate.py wrote")
    parser.add_argument("--public", default=PUBLIC, help="the organisers' public label file")
    parser.add_argument("--answers", action="store_true", help="show the organisers' public labels beside ours")
    args = parser.parse_args()
    if not os.path.exists(args.labels):
        sys.exit(f"no file at {args.labels}\nRun scripts/check_candidate.py first, or give the path of a labels file.")
    if not os.path.exists(args.public):
        sys.exit(f"no public label file at {args.public}")

    try:
        ours = sub.read_submission(args.labels)
        truth = read_public(args.public, len(ours))
    except (ValueError, IndexError) as error:
        sys.exit(f"could not read the files: {error}")
    graded = np.flatnonzero(truth != X)
    if not len(graded):
        sys.exit("the public label file grades no frame")
    low, high = int(graded[0]), int(graded[-1]) + 1
    first_half = sub.GRADED_BLOCKS[0]
    kickoff = first_half[0] if first_half[0] <= low < first_half[1] else None

    def when(frame: int) -> str:
        return clock((frame - kickoff) / FPS) if kickoff is not None else "  --   "

    lines = [WARNING, "", f"File: {args.labels}", "",
             f"The public window: frames {low} to {high - 1}, {(high - low) / FPS:.0f} seconds."]
    if kickoff is not None:
        lines.append(f"On the match clock: {when(low)} to {when(high)} of the first half, counted from the kickoff "
                     f"at frame {kickoff}.")
    else:
        lines.append("It is not in the first half, so no match clock is shown.")
    lines += [f"In the tracking timeline, and in a video that starts with it: {clock(low / FPS)} to "
              f"{clock(high / FPS)} from the start.", ""]

    mine = runs(ours, low, high)
    if not args.answers:
        lines.append(f"{'match clock':<19}{'tracking time':<15}{'who':<19}{'lasts':>8}")
        for a, b, code in mine:
            lines.append(f"{when(a)} - {when(b):<9}{clock(a / FPS):<15}{WHO[code]:<19}{(b - a) / FPS:>6.1f} s")
        lines += ["", summary(mine, "Ours")]
    else:
        theirs = runs(truth, low, high)
        cuts = sorted({a for a, _, _ in mine} | {a for a, _, _ in theirs} | {high})
        lines.append(f"{'match clock':<19}{'tracking time':<15}{'ours':<19}{'public answer':<19}{'lasts':>8}")
        for a, b in zip(cuts[:-1], cuts[1:]):
            same = ours[a] == truth[a]
            lines.append(f"{when(a)} - {when(b):<9}{clock(a / FPS):<15}{WHO[int(ours[a])]:<19}{WHO[int(truth[a])]:<19}"
                         f"{(b - a) / FPS:>6.1f} s" + ("" if same else "   <-- differ"))
        lines += ["", summary(mine, "Ours"), summary(theirs, "The public answers"),
                  "", "For understanding only. Never change the model because of what these five minutes show."]
    lines += ["", WARNING]

    print("\n".join(lines))
    target = os.path.join(os.path.dirname(os.path.abspath(args.labels)), OUT_NAME)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nAlso saved as {target}")
