"""Does the aligner line a broadcast up with the tactical recording? Open data only.

No open match comes with a TV broadcast, so one is made up from each match (mtah/fakecast.py): a
camera that shows the pitch around the ball, close-ups and replays at stoppages, the half-time cut
out, its own tracking errors. The tactical side is the match degraded to look like the competition's
tracking, placed on a longer timeline with empty frames before kick-off and during half-time, as in
the real video. The aligner sees only the two sets of tracking and has to find how they line up.

This tests the lining up. It does not test how much a real director's cuts say about stoppages:
the made-up director follows rules I wrote.

    python3 experiments/align_check.py      # about 5 minutes
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import datasets, harness  # noqa: E402
from mtah.align import LIVE, NO_SHIFT, NOT_WIDE, REPLAY, NoAlignment, align, tactical_view  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.fakecast import fake_broadcast  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402

OUT = os.path.join("runs", "align_check.json")
PRE, HALF_TIME, POST = 3000, 6000, 2000
RUNS = (("J03WR9", 0, (1, 1), False), ("J03WR9", 1, (-1, -1), True), ("J03WN1", 0, (1, 1), False),
        ("METRICA2", 0, (-1, 1), False), ("METRICA3", 1, (1, 1), True), (datasets.SKILLCORNER_DEV[0], 0, (1, 1), False))


def on_video_timeline(match):
    """The tactical players with empty frames before kick-off, at half-time and after, and where each match frame went."""
    (a0, a1), (b0, b1) = match.halves
    n = PRE + (a1 - a0) + HALF_TIME + (b1 - b0) + POST
    players = np.full((n, match.players.shape[1], 2), np.nan, dtype=np.float32)
    players[PRE:PRE + a1 - a0] = match.players[a0:a1]
    second = PRE + (a1 - a0) + HALF_TIME
    players[second:second + b1 - b0] = match.players[b0:b1]
    moved = np.full(match.n_frames, -1, dtype=np.int64)
    moved[a0:a1] = np.arange(a0, a1) - a0 + PRE
    moved[b0:b1] = np.arange(b0, b1) - b0 + second
    return players, moved


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(pre=PRE, half_time=HALF_TIME, post=POST)
    stats = load_stats()
    rows = []
    print(f"{'match':<11}{'seed':>5}{'drawn':>9}{'pieces':>8}{'way round':>11}{'teams':>7}{'shift exact':>13}{'within 2':>10}"
          f"{'worst':>7}{'live found':>12}{'live right':>12}{'replays':>9}{'seconds':>9}")
    for match_id, seed, flip, swap in RUNS:
        clean = harness.harmonise(datasets.load_open(match_id))
        cast = fake_broadcast(clean, seed=seed, flip=flip, swap_teams=swap)
        copy = harness.prepared(match_id, DegradeConfig(), 3000 + seed, stats)
        players, moved = on_video_timeline(copy)
        truth = np.where(cast.live >= 0, moved[np.clip(cast.live, 0, None)] - np.arange(len(cast.live)), NO_SHIFT)
        drawn = {(1, 1): "same", (-1, -1): "turned", (-1, 1): "mirrored", (1, -1): "mirrored"}[flip]
        started = time.time()
        try:
            found = align(players, cast.players, cast.valid, copy.pitch, copy.team, cast.team)
        except NoAlignment as problem:
            print(f"{match_id:<11}{seed:>5}{drawn:>9}  not lined up: {problem}")
            rows.append({"match": match_id, "seed": seed, "failed": str(problem)})
            continue
        took = time.time() - started
        both = (truth != NO_SHIFT) & (found.shift != NO_SHIFT)
        error = np.abs(found.shift[both].astype(np.int64) - truth[both])
        live_truth, live_called = cast.state == LIVE, found.state == LIVE
        _, seen_from_tactical = tactical_view(found, len(players))
        row = {"match": match_id, "seed": seed, "drawn": drawn, "pieces": len(found.pieces),
               "way_round_right": found.flip == flip, "teams_right": found.same_teams == (not swap),
               "shift_exact": float((error == 0).mean()), "shift_within_2": float((error <= 2).mean()),
               "shift_worst": int(error.max()), "covered": float(both.sum() / (truth != NO_SHIFT).sum()),
               "live_found": float(live_called[live_truth].mean()), "live_right": float(live_truth[live_called].mean()),
               "replays_caught": float((found.state[cast.state == REPLAY] != LIVE).mean()),
               "not_wide_right": float((found.state[cast.state == NOT_WIDE] == NOT_WIDE).mean()),
               "fit_m": found.fit_m, "clips": found.clips, "agreed": found.agreed, "seconds": took,
               "valid_share": float(cast.valid.mean()),
               "tactical_frames_with_live_view": float((seen_from_tactical[moved[moved >= 0]] == LIVE).mean())}
        rows.append(row)
        print(f"{match_id:<11}{seed:>5}{drawn:>9}{row['pieces']:>8}{'right' if row['way_round_right'] else 'WRONG':>11}"
              f"{'right' if row['teams_right'] else 'WRONG':>7}{row['shift_exact']:>13.1%}{row['shift_within_2']:>10.1%}"
              f"{row['shift_worst']:>7}{row['live_found']:>12.1%}{row['live_right']:>12.1%}{row['replays_caught']:>9.1%}"
              f"{took:>9.0f}", flush=True)
    done = [r for r in rows if "failed" not in r]
    print(f"\nlined up: {len(done)} of {len(rows)}. Typical distance between matched players {np.mean([r['fit_m'] for r in done]):.2f} m, "
          f"clips that agreed {np.mean([r['agreed'] / r['clips'] for r in done]):.0%}, "
          f"wide shots {np.mean([r['valid_share'] for r in done]):.0%} of the made-up broadcasts.")
    record.update(rows=rows)
    save_manifest(OUT, record)
