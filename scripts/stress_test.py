"""How much of the score survives if the real tracking is worse than the degrader assumes?

Trains once on normally degraded matches, then tests held-out matches under conditions the model
never trained on. Also trains ablations to show where the gain comes from. Open data only.

    python3 scripts/stress_test.py
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, degrade, load_stats  # noqa: E402
from mtah.model import LearnedModel, TreeConfig  # noqa: E402
from mtah.provenance import manifest  # noqa: E402

HELD_OUT = ("J03WR9", "J03WOH", "METRICA2")
COPIES = 3
BASE = DegradeConfig()
R = dataclasses.replace
CONDITIONS = {
    "as trained": BASE,
    "ball always visible": R(BASE, drop_ball=False),
    "ball noise 1.5 m": R(BASE, ball_noise_m=1.5),
    "ball noise 3 m": R(BASE, ball_noise_m=3.0),
    "false ball 5%": R(BASE, ball_false_rate=0.05),
    "false ball 15%": R(BASE, ball_false_rate=0.15),
    "player noise 1 m": R(BASE, player_noise_m=1.0),
    "player noise 2 m": R(BASE, player_noise_m=2.0),
    "tracklets 10 s": R(BASE, tracklet_mean_s=10.0),
    "objects down to 18": R(BASE, gap_mean_s=8.0),
    "wrong team 5%": R(BASE, wrong_team_rate=0.05),
    "wrong team 15%": R(BASE, wrong_team_rate=0.15),
    "more blackouts": R(BASE, gap_keep=1.0),
    "ball in most stoppages": R(BASE, dead_hide_prob=0.1),
    "ball in few stoppages": R(BASE, dead_hide_prob=0.9),
    "white noise 1 m": R(BASE, player_noise_m=1.0, noise_rho=0.5),
    "no ball at all": "no_ball",
    "all harsher": R(BASE, ball_noise_m=1.5, ball_false_rate=0.05, player_noise_m=1.0,
                     tracklet_mean_s=15.0, wrong_team_rate=0.05, gap_keep=0.7),
    "very harsh": R(BASE, ball_noise_m=3.0, ball_false_rate=0.15, player_noise_m=2.0,
                    tracklet_mean_s=10.0, gap_mean_s=5.0, wrong_team_rate=0.1, gap_keep=1.0,
                    dead_hide_prob=0.5),
}


def blank_ball(match):
    return match.with_tracking(np.full_like(match.ball, np.nan), match.players, match.team)


def score(model, match, transform=None):
    x = harness.without_labels(match if transform is None else transform(match))
    prob, frames, half = model.probabilities(x)
    pred = expand(decode_samples(prob, half, DECODER), frames, half, match.halves, match.n_frames)
    return m.evaluate(match.labels, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="runs/stress_test.json")
    args = parser.parse_args()

    record = manifest(held_out=list(HELD_OUT), copies=COPIES)
    stats = load_stats()
    ids = [i for i in ds.available() if i not in HELD_OUT]
    started = time.time()
    model = LearnedModel(decoder=DECODER)
    model.fit(harness.training_matches(ids, 0, COPIES, stats))
    print(f"trained on {len(ids)} matches x{COPIES} in {time.time() - started:.0f}s\n", flush=True)

    print(f"{'condition':<22} {'spellF1':>8} {'macroF1':>8} {'poss x':>7} {'team':>6} {'alive':>6}", flush=True)
    results = {}
    for name, cfg in CONDITIONS.items():
        rows = []
        for held in HELD_OUT:
            clean = harness.harmonise(ds.load_open(held))
            if cfg == "no_ball":
                rows.append(score(model, degrade(clean, BASE, 1000, stats), blank_ball))
            else:
                rows.append(score(model, degrade(clean, cfg, 1000, stats)))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("spell_f1", "macro_f1", "team_acc", "alive_dead_f1")}
        mean["poss"] = float(np.mean([r["n_pred_spells"] / r["n_gt_spells"] for r in rows]))
        results[name] = mean
        print(f"{name:<22} {mean['spell_f1']:>8.3f} {mean['macro_f1']:>8.3f} {mean['poss']:>7.2f}"
              f" {mean['team_acc']:>6.3f} {mean['alive_dead_f1']:>6.3f}", flush=True)
    os.makedirs("runs", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"manifest": record, "results": results}, handle, indent=1)
