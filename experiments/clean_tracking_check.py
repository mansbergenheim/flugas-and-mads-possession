"""Would cleaner tracking rescue the short possessions the model misses? Open matches only.

Trains one model, then predicts the same 3 held-out matches from four versions of the input: as
tested, clean players, clean ball, and both. About 8 minutes. Run short_possessions.py first, because
this reads the sorting it wrote. The model still trained on degraded copies, so this says what better
inputs give this model, not what a model trained on clean tracking could do.

    python3 experiments/clean_tracking_check.py --tag v5
"""

import argparse
import csv
import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, degrade, load_stats  # noqa: E402
from mtah.model import LearnedModel  # noqa: E402

HELD_OUT = ("J03WR9", "J03WOH", "METRICA2")
LOOSE = ("one touch, then ball in flight", "one touch, quick change", "no clear touch")
BASE = DegradeConfig()
BALL = dict(drop_ball=False, ball_noise_m=0.0)
PLAYERS = dict(player_noise_m=0.0, tracklet_mean_s=1e7, gap_mean_s=0.0, wrong_team_rate=0.0)
INPUTS = {"as tested": BASE, "clean players": dataclasses.replace(BASE, **PLAYERS),
          "clean ball": dataclasses.replace(BASE, **BALL), "both": dataclasses.replace(BASE, **BALL, **PLAYERS)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v5")
    args = parser.parse_args()

    os.chdir(ROOT)
    with open(os.path.join("runs", f"short_possessions_{args.tag}.csv")) as handle:
        kind = {(r["match"], int(r["start"])): r["kind"] for r in csv.DictReader(handle)}

    stats = load_stats()
    model = LearnedModel(decoder=DECODER)
    model.fit(harness.training_matches([i for i in ds.available() if i not in HELD_OUT], 0, TRAIN_COPIES, stats))

    scores = {name: [] for name in INPUTS}
    hits = {}
    for held in HELD_OUT:
        clean = harness.harmonise(ds.load_open(held))
        for name, cfg in INPUTS.items():
            match = degrade(clean, cfg, harness.TEST_SEED_OFFSET, stats)
            prob, frames, half = model.probabilities(harness.without_labels(match))
            pred = expand(decode_samples(prob, half, DECODER), frames, half, match.halves, match.n_frames)
            scores[name].append(m.evaluate(match.labels, pred)["spell_f1"])
            runs = m.valid_runs(match.labels)
            spells, predicted = m.possession_spells(match.labels, runs), m.possession_spells(pred, runs)
            matched = m._matched(m._overlaps(spells, predicted), len(spells), 0.5)
            for i, spell in enumerate(spells):
                if (held, spell[0]) in kind:
                    hits.setdefault((held, spell[0]), {})[name] = bool(matched[i])

    print("spellF1, one model, 3 matches")
    for name in INPUTS:
        print(f"  {name:<14}{np.mean(scores[name]):.3f}")
    print("\nshare found of possessions under 4 s, then share of the misses each input rescues")
    for title, test in (("signs of control", lambda k: k not in LOOSE), ("single touch or none", lambda k: k in LOOSE)):
        keys = [k for k in hits if test(kind[k])]
        missed = [k for k in keys if not hits[k]["as tested"]]
        print(f"  {title:<22}{len(keys):>4}  " + "  ".join(f"{n} {np.mean([hits[k][n] for k in keys]):.0%}" for n in INPUTS))
        print(f"  {'rescued':<22}{len(missed):>4}  " + "  ".join(f"{n} {np.mean([hits[k][n] for k in missed]):.0%}" for n in list(INPUTS)[1:]))
