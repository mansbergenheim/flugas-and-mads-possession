"""Does the TV director cut away when the candidate says play has stopped? Run this yourself.

    python3 scripts/check_director.py

Needs two files that are already on this computer, outside the workspace:

    <data folder>/alignment/broadcast_alignment.npz        from scripts/align_broadcast.py
    <data folder>/candidate_check/candidate_trees_tcn.csv  from scripts/check_candidate.py

(--labels picks another label file, such as the submission.) It compares the candidate's own labels
with what the broadcast showed at the same moment. It uses no answer key and nothing from the public
five minutes, and it changes nothing. It prints words only, so the printout can be pasted anywhere.

Read it for one thing: is the director's cutting a strong enough sign of stoppages to be worth
building on later? The candidate's labels are not the truth, so a weak link can also mean the
candidate is wrong, and a strong link says nothing about which of the two is right where they differ.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import submission as sub  # noqa: E402
from mtah.align import director_link  # noqa: E402
from mtah.loaders import gated  # noqa: E402

SHARE_WORDS = ((0.75, "most"), (0.5, "more than half"), (0.25, "between a quarter and half"), (0.1, "a few"))


def share_words(value):
    if value is None:
        return "none to count"
    return next((words for low, words in SHARE_WORDS if value >= low), "hardly any")


def times_words(ratio):
    if ratio is None:
        return "cannot be told"
    if ratio >= 4:
        return "far more often (four times or more)"
    if ratio >= 2:
        return "clearly more often (two to four times)"
    if ratio >= 1.3:
        return "a little more often"
    return "about as often or less"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    parser.add_argument("--labels", default=None, help="label file; default <data folder>/candidate_check/candidate_trees_tcn.csv")
    args = parser.parse_args()
    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    started = time.time()
    try:
        folder = gated.data_dir(root)
    except RuntimeError as problem:
        sys.exit(f"1 FILES        {problem}")
    aligned = os.path.join(folder, "alignment", "broadcast_alignment.npz")
    labels_path = args.labels or os.path.join(folder, "candidate_check", "candidate_trees_tcn.csv")
    missing = [p for p in (aligned, labels_path) if not os.path.exists(p)]
    if missing:
        sys.exit("1 FILES        not found:\n" + "".join(f"               {p}\n" for p in missing)
                 + "               Run scripts/align_broadcast.py and scripts/check_candidate.py first.")
    try:
        labels = sub.read_submission(labels_path)
        with np.load(aligned) as saved:
            state = saved["tactical_state"]
    except (OSError, ValueError, KeyError) as problem:
        sys.exit(f"1 FILES        could not be read: {problem}")
    if len(labels) != len(state):
        sys.exit("1 FILES        the label file and the alignment cover different numbers of frames")
    print(f"1 FILES        alignment and labels read ({os.path.basename(labels_path)})")

    found = director_link(labels, state, sub.GRADED_BLOCKS)
    ratio = None
    if found["away_dead"] is not None and found["away_live"]:
        ratio = found["away_dead"] / found["away_live"]
    print(f"2 CUTTING      the broadcast is away from the live wide shot {times_words(ratio)}")
    print("               when the candidate says dead than when it says live")
    print(f"3 CUTS         {share_words(found['cuts_near_dead'])} of the director's cuts away begin close to a moment")
    lift = None
    if found["cuts_near_dead"] is not None and found["chance_near_dead"]:
        lift = found["cuts_near_dead"] / found["chance_near_dead"]
    print(f"               the candidate calls dead, against {share_words(found['chance_near_dead'])} if the cuts fell at random")
    print(f"4 STOPPAGES    {share_words(found['stops_with_cut'])} of the candidate's stoppages of 5 s or more get a cut away")
    print(f"5 RESTARTS     in {share_words(found['back_before_restart'])} of those, the live wide shot is back before"
          " the candidate's restart")

    strong = ratio is not None and ratio >= 2 and lift is not None and lift >= 1.5
    weak = ratio is None or ratio < 1.3
    verdict = ("the director's cuts follow the candidate's stoppages closely: worth building on after Phase 1" if strong
               else "the director's cuts say little about stoppages here: close the broadcast idea" if weak
               else "some link, not a strong one: not worth building on now")
    print(f"VERDICT        {verdict}   ({time.time() - started:.0f} s)")
