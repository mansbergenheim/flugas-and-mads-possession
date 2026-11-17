"""Build mtah/degrade_stats.json from the organisers' public baseline_raw.csv.

In that file D means the tracker had no ball, so it shows where the ball was missing without
touching the gated data. No labels are involved.

    python3 scripts/fit_degrader_public.py ../official-repository
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah.degrade import STATS_PATH  # noqa: E402
from mtah.match import D  # noqa: E402
from mtah.submission import GRADED_BLOCKS, read_submission  # noqa: E402


def runs(mask: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "official-repository")
    raw = read_submission(os.path.join(repo, "evaluation", "results", "baseline_raw.csv"))
    gaps, seen = [], []
    for start, stop in GRADED_BLOCKS:
        missing = raw[start:stop] == D
        gaps.extend(runs(missing).tolist())
        seen.extend(runs(~missing).tolist())
    stats = {"source": "official baseline_raw.csv, graded frames, no labels",
             "ball_gap_frames": sorted(int(v) for v in gaps),
             "ball_run_frames": sorted(int(v) for v in seen)}
    with open(STATS_PATH, "w") as handle:
        json.dump(stats, handle)
    g, s = np.array(gaps), np.array(seen)
    print(f"{len(g)} gaps, median {np.median(g) / 25:.1f}s, longest {g.max() / 25:.1f}s; "
          f"{len(s)} visible runs, median {np.median(s) / 25:.1f}s; "
          f"visible share {s.sum() / (s.sum() + g.sum()):.3f}")
