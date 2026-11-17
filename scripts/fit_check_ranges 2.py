"""Reference values for scripts/check_real_match.py, from open matches only.

Writes mtah/check_ranges.json:
  ball    how much of the ball track is an exactly straight line in Metrica's sample games (stored at
          the same precision as the competition's tracking) and in the DFL matches (followed by cameras);
  inputs  for every model input, what the 30 practice copies cover, drawn exactly as for training;
  output  figures about the model's own output on the 10 open matches it had not seen (runs/oof/v5).

    python3 scripts/fit_check_ranges.py      # about 15 minutes
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from check_real_match import RANGES, STATS, ball_figures, input_figures, output_figures  # noqa: E402
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import TRAIN_COPIES  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.features import build_features  # noqa: E402
from sweep_decoder import load  # noqa: E402

FLOAT32_SLACK = 5e-5   # the caches hold metres as float32


def box(values):
    return [float(v) for v in np.nanpercentile(np.asarray(values, dtype=np.float64), (0, 25, 75, 100))]


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    stats = load_stats()
    ids = ds.available()

    ball = {}
    for match_id in ids:
        match = ds.load_open(match_id)
        x, y = match.ball[:, 0].astype(np.float64), match.ball[:, 1].astype(np.float64)
        fine = (1e-5 * match.pitch[0], 1e-5 * match.pitch[1])   # the precision of Metrica's EPTS files
        ball[match_id] = ball_figures(x, y, *fine, players=match.players, slack=FLOAT32_SLACK, blocks=match.halves)
        print(f"{match_id}: " + ", ".join(f"{k} {v:.3f}" for k, v in ball[match_id].items()), flush=True)
    metrica = [ball[i] for i in ids if i.startswith("METRICA")]
    dfl = [ball[i] for i in ids if not i.startswith("METRICA")]
    ball_ref = {"metrica_share": [min(r["share"] for r in metrica), max(r["share"] for r in metrica)],
                "dfl_share": [min(r["share"] for r in dfl), max(r["share"] for r in dfl)],
                "per_match": ball}

    figures, names = [], None
    for copy in harness.training_matches(ids, 0, TRAIN_COPIES, stats):
        values, names, _, _ = build_features(copy, extra=True)
        figures.append(input_figures(values))
        print(f"inputs measured on copy {len(figures)}", flush=True)
    figures = np.array(figures)
    inputs_ref = {"copies": len(figures), "names": names,
                  "ranges": {name: {stat: box(figures[:, i, j]) for j, stat in enumerate(STATS)}
                             for i, name in enumerate(names)}}

    rows = []
    for match_id, fold in sorted(load("v5").items()):
        test = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        if not np.array_equal(test.labels, fold["labels"]):
            raise ValueError(f"{match_id}: the rebuilt test copy differs from the saved fold")
        halves = [tuple(int(v) for v in r) for r in fold["halves"]]
        seen = np.isfinite(test.ball[fold["frames"], 0])
        rows.append(output_figures(fold["prob"], fold["frames"], fold["half"], seen, halves, len(fold["labels"])))
        print(f"output measured on {match_id}", flush=True)
    output_ref = {key: box([r[key] for r in rows]) for key in rows[0]}

    with open(RANGES, "w") as handle:
        json.dump({"made_from": "open matches only: 10 clean matches, 30 practice copies, runs/oof/v5",
                   "ball": ball_ref, "inputs": inputs_ref, "output": output_ref}, handle, indent=1)
    print(f"written {os.path.relpath(RANGES)}")
