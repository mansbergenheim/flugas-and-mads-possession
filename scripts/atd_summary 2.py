"""Run this yourself on the real ATD, in your own terminal, not through an AI tool.

Measures the tracking properties the degrader can only estimate, and writes aggregates to
runs/gated/atd_summary.json. It prints nothing except where the file went. Do not paste the file
into an AI tool unless the organisers have confirmed aggregates are acceptable.

    MTAH_DATA_DIR=~/mtah-data python3 scripts/atd_summary.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah.features import jitter_level  # noqa: E402
from mtah.loaders import gated  # noqa: E402


def run_lengths(valid):
    edges = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def jumps_per_minute(xy, fps, limit_m=2.0):
    """Moves too big for one frame (50 m/s), per minute of tracked time. xy is [T, 2] or [T, P, 2]."""
    step = xy[1:] - xy[:-1]
    size = np.hypot(step[..., 0], step[..., 1])
    known = np.isfinite(size)
    return float((size[known] > limit_m).sum() / max(known.sum() / fps / 60, 1e-9))


def ball_to_players(match, graded):
    """Distance from the visible ball to the nearest tracked object, on graded frames."""
    ball, players = match.ball[graded], match.players[graded]
    keep = np.isfinite(ball[:, 0])
    delta = players[keep] - ball[keep, None, :]
    size = np.hypot(delta[..., 0], delta[..., 1])
    return np.where(np.isfinite(size), size, np.inf).min(axis=1)


def summarise(match):
    graded = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.halves:
        graded[start:stop] = True
    seen = np.isfinite(match.players[..., 0])
    lives, gaps = [], []
    for slot in range(seen.shape[1]):
        s = seen[graded, slot]
        lives.extend(run_lengths(s).tolist())
        inner = np.flatnonzero(s)
        if len(inner):
            gaps.extend(run_lengths(~s[inner[0]:inner[-1] + 1]).tolist())
    lives, gaps = np.array(lives) / match.fps, np.array(gaps) / match.fps
    near = ball_to_players(match, graded)
    return {"ball_jitter_m": jitter_level(match.ball[graded]),
            "ball_jumps_per_min": jumps_per_minute(match.ball[graded], match.fps),
            "ball_near_player_median_m": float(np.median(near)) if len(near) else 0.0,
            "ball_within_3m_share": float((near < 3.0).mean()) if len(near) else 0.0,
            "track_jumps_per_min": jumps_per_minute(match.players[graded], match.fps),
            "ball_visible_graded": float(np.isfinite(match.ball[graded, 0]).mean()),
            "objects_per_graded_frame": float(seen[graded].sum(1).mean()),
            "slots": int(seen.shape[1]), "home_slots": int((match.team == 0).sum()),
            "player_jitter_m": jitter_level(match.players[graded]),
            "tracklet_mean_s": float(lives.mean()) if len(lives) else 0.0,
            "tracklet_median_s": float(np.median(lives)) if len(lives) else 0.0,
            "slot_gap_mean_s": float(gaps.mean()) if len(gaps) else 0.0}


if __name__ == "__main__":
    result = summarise(gated.load_atd_match())
    os.makedirs(os.path.join("runs", "gated"), exist_ok=True)
    path = os.path.join("runs", "gated", "atd_summary.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"written to {path}")
