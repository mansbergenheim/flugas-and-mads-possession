"""Does training on labels closer to the written definition help? Both models on both sets of labels.

We don't know how strict the competition's labels are, so a model has to be judged on both. Each
model gets its own decoder for each set of labels, picked on the other matches' held-out predictions,
so a simple change of decoder gets a fair chance to do the same job as retraining.

Decided before the results were seen: adopt only if the gain on the definition labels is clearly
larger than the loss on the current labels, and the gain shows on most matches.

    python3 experiments/compare_definition_labels.py --new v5_def
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.decoder import DecoderConfig, decode_samples, expand  # noqa: E402
from short_possessions import follow_definition  # noqa: E402
from sweep_decoder import GRID, load  # noqa: E402

LABELS = {"current": None, "half way (under 2 s)": 2.0, "definition (under 4 s)": 4.0}


def label_sets(ids):
    out = {}
    for match_id in ids:
        match = harness.harmonise(ds.load_open(match_id))
        out[match_id] = {name: match.labels if limit is None else follow_definition(match, limit)
                         for name, limit in LABELS.items()}
    return out


def score_grid(folds, labels, configs):
    """score[label set][config, match] and possessions ratio, decoding each pair only once."""
    ids = sorted(folds)
    spell = {name: np.zeros((len(configs), len(ids))) for name in LABELS}
    ratio = {name: np.zeros((len(configs), len(ids))) for name in LABELS}
    for c, cfg in enumerate(configs):
        for j, match_id in enumerate(ids):
            fold = folds[match_id]
            halves = [tuple(int(v) for v in r) for r in fold["halves"]]
            pred = expand(decode_samples(fold["prob"], fold["half"], cfg), fold["frames"], fold["half"],
                          halves, len(fold["labels"]))
            for name in LABELS:
                result = m.evaluate(labels[match_id][name], pred)
                spell[name][c, j] = result["spell_f1"]
                ratio[name][c, j] = result["n_pred_spells"] / result["n_gt_spells"]
    return spell, ratio


def held_out_pick(grid):
    """Each match scored with the setting that did best on the other matches."""
    n = grid.shape[1]
    return np.array([grid[int(np.delete(grid, j, axis=1).mean(1).argmax()), j] for j in range(n)]), \
        [int(np.delete(grid, j, axis=1).mean(1).argmax()) for j in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", default="v5")
    parser.add_argument("--new", default="v5_def")
    args = parser.parse_args()

    os.chdir(ROOT)
    old, new = load(args.old), load(args.new)
    ids = sorted(new)
    configs = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
    labels = label_sets(ids)
    for match_id in ids:
        if not np.array_equal(new[match_id]["labels"], labels[match_id]["definition (under 4 s)"]):
            raise ValueError(f"{match_id}: saved labels aren't the definition labels")
    results = {tag: score_grid(folds, labels, configs) for tag, folds in (("current model", old), ("new model", new))}

    print(f"{'scored on':<26}{'current model':>16}{'new model':>12}{'gain':>9}  up/down")
    for name in LABELS:
        a, _ = held_out_pick(results["current model"][0][name])
        b, picks = held_out_pick(results["new model"][0][name])
        diff = b - a
        print(f"{name:<26}{a.mean():>16.3f}{b.mean():>12.3f}{diff.mean():>+9.3f}  {(diff > 0).sum()}/{(diff < 0).sum()}")
        print("   per match: " + " ".join(f"{k}:{d:+.3f}" for k, d in zip(ids, diff)))
        ratio = results["new model"][1][name]
        print(f"   new model predicts x{np.mean([ratio[p, j] for j, p in enumerate(picks)]):.2f} of the true count")

    print("\none decoder for everything, the best on the mean of current and definition labels")
    for tag, (spell, _) in results.items():
        both = (spell["current"] + spell["definition (under 4 s)"]) / 2
        best = int(both.mean(1).argmax())
        print(f"  {tag:<14}{configs[best]}  current {spell['current'][best].mean():.3f}"
              f"  definition {spell['definition (under 4 s)'][best].mean():.3f}")
