"""Is the real tracking like the tracking the model trained on? Run this yourself, not through an AI tool.

Prints words only, never a measurement, so the result can be repeated to anyone. Writes nothing.

    MTAH_DATA_DIR=~/mtah-data python3 scripts/compare_tracking.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atd_summary import summarise  # noqa: E402
from mtah.loaders import gated  # noqa: E402

RANGES = os.path.join(os.path.dirname(__file__), "..", "mtah", "tracking_ranges.json")
NAMES = {"ball_visible_graded": "Ball visible",
         "ball_jitter_m": "Ball wobble",
         "ball_jumps_per_min": "Ball impossible jumps",
         "ball_near_player_median_m": "Ball distance to nearest player",
         "ball_within_3m_share": "Ball within 3 m of a player",
         "objects_per_graded_frame": "Objects tracked",
         "player_jitter_m": "Player wobble",
         "track_jumps_per_min": "Track impossible jumps",
         "tracklet_mean_s": "Track length, mean",
         "tracklet_median_s": "Track length, median",
         "slot_gap_mean_s": "Gaps between tracks in a slot"}


def position(value, low, lower_quarter, upper_quarter, high):
    """Where a value sits among the training copies, as words."""
    if value < low:
        return "BELOW everything trained on"
    if value > high:
        return "ABOVE everything trained on"
    if value < lower_quarter:
        return "inside, low end"
    return "inside, high end" if value > upper_quarter else "inside, middle"


def report(measured, ranges):
    return [(label, position(measured[key], *ranges[key])) for key, label in NAMES.items()]


if __name__ == "__main__":
    with open(RANGES) as handle:
        ranges = json.load(handle)["ranges"]
    lines = report(summarise(gated.load_atd_match()), ranges)
    for label, verdict in lines:
        print(f"{label:<34} {verdict}")
    outside = sum(verdict.isupper() or verdict.startswith(("BELOW", "ABOVE")) for _, verdict in lines)
    print(f"\n{outside} of {len(lines)} outside what the model trained on")
