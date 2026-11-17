"""Size the Smart Tagging weights using simulated tags on saved held-out probabilities.

Open matches have no Smart Tagging, so this can't say how much the real tags help. It answers a
narrower question: at what weight does the prior help with mediocre tags without hurting with bad ones?

    python3 scripts/sweep_tags.py --tag v3
"""

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.smart_tagging import TagWeights, simulate_tags, tag_prior  # noqa: E402
from sweep_decoder import load  # noqa: E402

QUALITY = {"good": dict(recall=0.85, team_correct=0.9, restart_recall=0.9, false_rate=0.05),
           "mediocre": dict(recall=0.7, team_correct=0.8, restart_recall=0.75, false_rate=0.15),
           "bad": dict(recall=0.5, team_correct=0.6, restart_recall=0.5, false_rate=0.4)}
SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)


def run(fold, tags, weights):
    halves = [tuple(int(v) for v in r) for r in fold["halves"]]
    prior = tag_prior(tags, fold["frames"], fold["half"], halves, weights)
    path = decode_samples(fold["prob"], fold["half"], DECODER, prior)
    pred = expand(path, fold["frames"], fold["half"], halves, len(fold["labels"]))
    return m.evaluate(fold["labels"], pred)["spell_f1"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v3")
    args = parser.parse_args()
    folds = load(args.tag)
    base = TagWeights()
    print(f"{len(folds)} folds, mean spellF1 with simulated tags\n")
    print(f"{'tag quality':<12}" + "".join(f"{'x' + str(s):>8}" for s in SCALES))
    for name, quality in QUALITY.items():
        row = []
        for scale in SCALES:
            w = dataclasses.replace(base, phase=base.phase * scale, alive=base.alive * scale,
                                    restart_dead=base.restart_dead * scale,
                                    restart_team=base.restart_team * scale)
            scores = []
            for match_id, fold in folds.items():
                tags = simulate_tags(fold["labels"], np.random.default_rng(1), **quality)
                scores.append(run(fold, tags, w))
            row.append(float(np.mean(scores)))
        print(f"{name:<12}" + "".join(f"{v:>8.3f}" for v in row))
