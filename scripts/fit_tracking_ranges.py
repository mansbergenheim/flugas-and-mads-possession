"""Measure the tracking the model trains on, so the real match can be compared with it.

Runs on the degraded copies of the open matches, exactly as they are drawn for training. Writes
mtah/tracking_ranges.json, which scripts/compare_tracking.py reads. Open data only.

    python3 scripts/fit_tracking_ranges.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atd_summary import summarise  # noqa: E402
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import TRAIN_COPIES  # noqa: E402
from mtah.degrade import load_stats  # noqa: E402

PATH = os.path.join(os.path.dirname(__file__), "..", "mtah", "tracking_ranges.json")
SKIP = ("slots", "home_slots")

if __name__ == "__main__":
    rows = [summarise(match) for match in harness.training_matches(ds.available(), 0, TRAIN_COPIES, load_stats())]
    ranges = {key: [float(v) for v in np.percentile([r[key] for r in rows], (0, 25, 75, 100))]
              for key in rows[0] if key not in SKIP}
    with open(PATH, "w") as handle:
        json.dump({"copies": len(rows), "ranges": ranges}, handle, indent=1)
    for key, values in ranges.items():
        print(f"{key:<28}" + "".join(f"{v:10.3f}" for v in values))
