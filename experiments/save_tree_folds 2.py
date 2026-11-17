"""Save the tree models' probabilities where the networks will be compared with them. Open data only.

Held-out predictions on the 10 open matches exist already (runs/oof/v5, runs/oof/v6_d0 to v6_d2), and
so do the stress ones (runs/stress_v6_parts). Two tests only kept their scores. This saves their
probabilities in the same form, so that a network can be averaged with the trees there too:

  runs/oof/v5_sc, runs/oof/trees_sc            SkillCorner's working half (models saved by skillcorner_v6.py)
  runs/oof/v5_unseen, runs/oof/trees_unseen    each open match predicted by models that never saw its source

"trees" is the final tree model: the extra inputs and three draws averaged, with no smoother.
The reserve half of SkillCorner is never loaded.

    python3 experiments/save_tree_folds.py      # about 40 minutes
"""

import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import LearnedModel, TreeConfig  # noqa: E402
from skillcorner_v6 import DEV, V5_MODEL, V6_MODEL  # noqa: E402
from sweep_decoder import score  # noqa: E402
from v6_common import DRAWS  # noqa: E402

DIRECTIONS = (("DFL", ds.DFL_IDS, "Metrica", ds.METRICA_IDS), ("Metrica", ds.METRICA_IDS, "DFL", ds.DFL_IDS))
TREES = "extra inputs, 3 draws averaged"


def keep(tag, match_id, prob, frames, half, match):
    folder = os.path.join("runs", "oof", tag)
    os.makedirs(folder, exist_ok=True)
    fold = {"prob": prob, "frames": frames, "half": half, "labels": match.labels, "halves": np.array(match.halves),
            "source": np.array(match.source)}
    np.savez_compressed(os.path.join(folder, f"{match_id}.npz"), **fold)
    return score(fold, DECODER)["spell_f1"]


def both(v5, models, match):
    """v5's probabilities and the three averaged models', on the same moments."""
    blind = harness.without_labels(match)
    p5, frames5, half5 = v5.probabilities(blind)
    values, swapped, frames, half = models[0].views(blind)
    if not (np.array_equal(frames, frames5) and np.array_equal(half, half5)):
        raise ValueError("the two sets of inputs are not sampled at the same moments")
    mean = sum(model.probabilities_from(values, swapped) for model in models) / float(len(models))
    return p5, mean, frames, half


if __name__ == "__main__":
    os.chdir(ROOT)
    stats = load_stats()

    with open(V5_MODEL, "rb") as handle:
        v5 = pickle.load(handle)
    with open(V6_MODEL, "rb") as handle:
        models = pickle.load(handle)["ensemble"].models
    a, b = [], []
    for match_id in DEV:
        match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        p5, mean, frames, half = both(v5, models, match)
        a.append(keep("v5_sc", match_id, p5, frames, half, match))
        b.append(keep("trees_sc", match_id, mean, frames, half, match))
    print(f"SkillCorner's working half: v5 {np.mean(a):.3f}, trees {np.mean(b):.3f}  (0.340 and 0.351 before)", flush=True)

    with open(os.path.join("runs", "source_held_out_v6.json")) as handle:
        earlier = json.load(handle)["results"]
    for train_name, train_ids, test_name, test_ids in DIRECTIONS:
        started = time.time()
        v5 = LearnedModel(decoder=DECODER)
        v5.fit(harness.training_matches(list(train_ids), 0, TRAIN_COPIES, stats))
        models = []
        for draw in range(len(DRAWS)):
            model = LearnedModel(trees=TreeConfig(seed=draw), extra=True)
            model.fit(harness.training_matches(list(train_ids), draw, TRAIN_COPIES, stats))
            models.append(model)
        was = earlier[f"{train_name} to {test_name}"]
        gap = 0.0
        for match_id in test_ids:
            match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
            p5, mean, frames, half = both(v5, models, match)
            gap = max(gap, abs(keep("v5_unseen", match_id, p5, frames, half, match) - was["v5"]["v5_setting"][match_id]),
                      abs(keep("trees_unseen", match_id, mean, frames, half, match) - was[TREES]["v5_setting"][match_id]))
        print(f"trained on {train_name} only, {len(test_ids)} {test_name} matches saved in {time.time() - started:.0f}s; "
              f"scores repeat runs/source_held_out_v6.json: {'yes' if gap < 1e-12 else 'NO'} (largest gap {gap:.2e})",
              flush=True)
