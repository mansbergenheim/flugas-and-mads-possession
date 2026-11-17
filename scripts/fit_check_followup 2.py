"""Reference values for the follow-up part of scripts/check_real_match.py, from open matches only.

For each of the 10 open matches, as tested: the held-out predictions in runs/oof/v5 on the standard
test copy, split by what is happening at the ball, and how often the ball turns at a player. Writes
mtah/check_followup.json.

    python3 scripts/fit_check_followup.py      # about 3 minutes
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check_real_match import FOLLOWUP, flight_bends, keeper_figures, turn_figures, why_figures  # noqa: E402
from mtah import harness  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.features import build_features  # noqa: E402
from sweep_decoder import load  # noqa: E402


def box(values):
    return [float(v) for v in np.nanpercentile(np.asarray(values, dtype=np.float64), (0, 25, 75, 100))]


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    stats = load_stats()
    why, turns, bends, keepers = [], [], [], []
    for match_id, fold in sorted(load("v5").items()):
        test = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        if not np.array_equal(test.labels, fold["labels"]):
            raise ValueError(f"{match_id}: the rebuilt test copy differs from the saved fold")
        values, names, frames, half = build_features(harness.without_labels(test))
        if not np.array_equal(frames, fold["frames"]):
            raise ValueError(f"{match_id}: samples differ from the saved fold")
        col = {name: values[:, i] for i, name in enumerate(names)}
        why.append(why_figures(fold["prob"], col, np.ones(len(frames), dtype=bool)))
        turns.append(turn_figures(test))
        bends.append(flight_bends(test))
        k = keeper_figures(col, test.pitch[0], np.ones(len(frames), dtype=bool))
        keepers += [k["home"], k["away"]]
        print(f"{match_id}: turns at a player {turns[-1]['turns_at_player']:.2f}, per minute "
              f"{turns[-1]['turns_per_minute']:.1f}; share at feet "
              f"{why[-1]['share|at one player' + chr(39) + 's feet']:.2f}; other turns {turns[-1]['away_distance_m']:.1f} m "
              f"from a player, one way {turns[-1]['away_one_way']:.2f}; flights {bends[-1]['flights']}, bent "
              f"{bends[-1]['bent_share']:.2f}, one way {bends[-1]['bend_one_way']:.2f}", flush=True)
    with open(FOLLOWUP, "w") as handle:
        json.dump({"made_from": "open matches only: runs/oof/v5 on the 10 standard test copies",
                   "why": {key: box([r[key] for r in why]) for key in why[0]},
                   "turns": {key: box([r[key] for r in turns]) for key in turns[0]},
                   "bends": {key: box([r[key] for r in bends]) for key in bends[0]},
                   "keepers": box(keepers)}, handle, indent=1)
    print(f"written {os.path.relpath(FOLLOWUP)}")
