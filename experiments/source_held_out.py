"""Does training on another labelling style help on a style the model has never seen?

broadcast_check.py found that most of the learned model's lost lead on SkillCorner comes from its
labels (or league), not its tracking: the model has partly learned DFL's and Metrica's conventions
for where a possession starts and ends. The competition's labels are a style we can't see. So
here a whole source is held out: train on DFL, test on Metrica, and the reverse, each with and
without the 10 working SkillCorner matches added to training. Same test copies as v5; each model
gets its decoder from the other test matches. About 30 minutes.

Decided before any result was seen: SkillCorner helps on an unseen labelling style if adding it
gains at least +0.01 in both directions. That is evidence about robustness, not a change: whether
to train on it is the team's call, alongside skillcorner_check.py's own rule.

    python3 experiments/source_held_out.py
"""

import itertools
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import TRAIN_COPIES  # noqa: E402
from mtah.decoder import DecoderConfig  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import LearnedModel  # noqa: E402
from sweep_decoder import GRID, score  # noqa: E402

DIRECTIONS = (("DFL", ds.DFL_IDS, "Metrica", ds.METRICA_IDS), ("Metrica", ds.METRICA_IDS, "DFL", ds.DFL_IDS))
CONFIGS = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]


def held_out_scores(folds, ids):
    grid = np.array([[score(folds[i], c)["spell_f1"] for i in ids] for c in CONFIGS])
    return np.array([grid[int(np.delete(grid, j, axis=1).mean(1).argmax()), j] for j in range(len(ids))])


if __name__ == "__main__":
    os.chdir(ROOT)
    cfg, stats = DegradeConfig(), load_stats()
    for train_name, train_ids, test_name, test_ids in DIRECTIONS:
        tests = {i: harness.prepared(i, cfg, harness.TEST_SEED_OFFSET, stats) for i in test_ids}
        result = {}
        for label, extra in (("alone", ()), ("with SkillCorner", ds.SKILLCORNER_DEV)):
            model = LearnedModel()
            model.fit(harness.training_matches(list(train_ids) + list(extra), 0, TRAIN_COPIES, stats))
            folds = {}
            for i, test in tests.items():
                prob, frames, half = model.probabilities(harness.without_labels(test))
                folds[i] = {"prob": prob, "frames": frames, "half": half, "labels": test.labels,
                            "halves": np.array(test.halves)}
            result[label] = held_out_scores(folds, list(test_ids))
            print(f"train {train_name} {label:<17} test {test_name}: {result[label].mean():.3f}", flush=True)
        diff = result["with SkillCorner"] - result["alone"]
        print(f"   gain {diff.mean():+.3f}, {(diff > 0).sum()} up / {(diff < 0).sum()} down: "
              + " ".join(f"{i}:{d:+.3f}" for i, d in zip(test_ids, diff)), flush=True)
