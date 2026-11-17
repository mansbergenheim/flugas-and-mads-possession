"""Run a script with the earlier, narrower training range, for a like-for-like comparison.

Test data is unchanged, so scoring this against the current model shows what the wider range did.

    python3 experiments/with_old_ranges.py scripts/train_oof.py --tag v5_old_ranges
    python3 experiments/with_old_ranges.py scripts/stress_test.py --out runs/stress_old_ranges.json
"""

import os
import runpy
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import harness  # noqa: E402
from mtah.degrade import DegradeConfig  # noqa: E402


def narrow(rng):
    return DegradeConfig(
        ball_noise_m=float(rng.uniform(0.2, 3.0)),
        ball_false_rate=float(rng.uniform(0.0, 0.15)),
        player_noise_m=float(np.exp(rng.uniform(np.log(0.1), np.log(2.5)))),
        noise_rho=float(rng.uniform(0.5, 0.99)),
        tracklet_mean_s=float(rng.uniform(10.0, 60.0)),
        gap_mean_s=float(rng.uniform(2.0, 8.0)),
        wrong_team_rate=float(rng.uniform(0.0, 0.1)),
        gap_keep=float(rng.uniform(0.4, 1.0)),
        dead_hide_prob=float(rng.uniform(0.0, 1.0)),
    )


harness.random_config = narrow
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
