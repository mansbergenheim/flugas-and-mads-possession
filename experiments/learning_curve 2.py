"""Would more matches help? Train on fewer of the other matches and see how fast the score falls.

For each held-out match, trains on 3 and on 6 of the other 9 (the 3 inside the 6, drawn at random)
and compares with the current model, which trained on all 9 (tag v5). The test copy and the training
copies of each match are the same as v5's, so only the number of matches changes. Each size gets its
decoder from the other matches' held-out predictions. About 50 minutes. The probabilities are saved
under runs/oof/lc3 and runs/oof/lc6, so a second run only rescores.

Decided before the results were seen: learning curves are usually close to straight against the
log of the data, so the gain from doubling to 18 matches is estimated as the gain per doubling
between 6 and 9. If that's under 0.01, more matches of the same kind aren't worth the remaining days.

    python3 experiments/learning_curve.py
"""

import argparse
import itertools
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import TRAIN_COPIES  # noqa: E402
from mtah.decoder import DecoderConfig  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import LearnedModel  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from sweep_decoder import GRID, load, score  # noqa: E402

SIZES = (3, 6)


def subsets(ids):
    picks = {}
    for held in ids:
        order = np.random.default_rng(zlib.crc32(held.encode())).permutation([i for i in ids if i != held])
        picks[held] = {k: sorted(order[:k].tolist()) for k in SIZES}
    return picks


def path(k, held):
    return os.path.join("runs", "oof", f"lc{k}", f"{held}.npz")


def train(ids, picks, seed):
    cfg, stats = DegradeConfig(), load_stats()
    for k in SIZES:
        save_manifest(os.path.join("runs", "oof", f"lc{k}", "manifest.json"),
                      manifest(held_out=list(ids), seed=seed, trained_on={h: picks[h][k] for h in ids}))
    for held in ids:
        todo = [k for k in SIZES if not os.path.exists(path(k, held))]
        if not todo:
            continue
        started = time.time()
        test = harness.prepared(held, cfg, seed + harness.TEST_SEED_OFFSET, stats)
        blind = harness.without_labels(test)
        for k in todo:
            model = LearnedModel()
            model.fit(harness.training_matches(picks[held][k], seed, TRAIN_COPIES, stats))
            prob, frames, half = model.probabilities(blind)
            np.savez_compressed(path(k, held), prob=prob, frames=frames, half=half, labels=test.labels,
                                halves=np.array(test.halves), source=test.source)
        print(f"{held} done in {time.time() - started:.0f}s", flush=True)


def held_out_scores(folds, ids, configs):
    """Each match scored with the decoder that did best on the other matches."""
    grid = np.array([[score(folds[i], cfg)["spell_f1"] for i in ids] for cfg in configs])
    return np.array([grid[int(np.delete(grid, j, axis=1).mean(1).argmax()), j] for j in range(len(ids))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", default="v5", help="tag of the model trained on all 9 other matches")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.chdir(ROOT)
    ids = ds.available()
    picks = subsets(ids)
    train(ids, picks, args.seed)

    folds = {k: load(f"lc{k}") for k in SIZES}
    folds[9] = load(args.full)
    for held in ids:
        for k in SIZES:
            if not np.array_equal(folds[k][held]["labels"], folds[9][held]["labels"]):
                raise ValueError(f"{held}: the {k}-match test copy differs from {args.full}'s")
    configs = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
    scores = {k: held_out_scores(folds[k], ids, configs) for k in (*SIZES, 9)}

    dfl = np.array([i.startswith("J") for i in ids])
    print(f"{'matches in training':<22}{'spellF1':>8}{'DFL':>7}{'Metrica':>9}")
    for k, s in scores.items():
        print(f"{k:<22}{s.mean():>8.3f}{s[dfl].mean():>7.3f}{s[~dfl].mean():>9.3f}")
    for a, b in ((3, 6), (6, 9)):
        diff = scores[b] - scores[a]
        print(f"\n{a} to {b}: {diff.mean():+.3f}, {(diff > 0).sum()} up / {(diff < 0).sum()} down, "
              f"{diff.mean() / np.log2(b / a):+.3f} per doubling")
        print("   per match: " + " ".join(f"{i}:{d:+.3f}" for i, d in zip(ids, diff)))
    print(f"\nestimated gain from 9 to 18 matches of the same kind: "
          f"{(scores[9] - scores[6]).mean() / np.log2(9 / 6):+.3f}")
