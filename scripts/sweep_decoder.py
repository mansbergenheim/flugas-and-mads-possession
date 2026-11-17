"""Explore decoder settings on saved held-out probabilities. Takes seconds, no retraining.

Each match's reported score uses settings chosen on the other matches' held-out predictions. That
isn't strictly nested: the models behind those predictions trained on the scored match, so it can
still leak a little. It can't leak more than the whole grid spans, which the sweep prints, and a truly
nested version would mean training a model for every pair of matches.

    python3 scripts/sweep_decoder.py --tag v3
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import metrics as m  # noqa: E402
from mtah.decoder import DecoderConfig, decode_samples, expand  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402

# Only cost / temperature matters to the path, so temperature stays at 1.
GRID = {"temperature": (1.0,),
        "turnover": (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
        "to_dead": (1.5, 2.0, 3.0, 4.0, 6.0),
        "from_dead": (1.5, 2.0, 3.0, 4.0, 6.0)}


def load(tag):
    out = {}
    for path in sorted(glob.glob(os.path.join("runs", "oof", tag, "*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            out[os.path.basename(path)[:-4]] = {k: z[k] for k in z.files}
    return out


def score(fold, cfg, key="prob"):
    halves = [tuple(int(v) for v in r) for r in fold["halves"]]
    path = decode_samples(fold[key], fold["half"], cfg)
    pred = expand(path, fold["frames"], fold["half"], halves, len(fold["labels"]))
    return m.evaluate(fold["labels"], pred)


def table(rows, ids):
    dfl = [i for i, k in enumerate(ids) if k.startswith("J")]
    met = [i for i, k in enumerate(ids) if not k.startswith("J")]
    s = np.array([r["spell_f1"] for r in rows])
    return (f"spellF1 {s.mean():.3f}  (DFL {s[dfl].mean():.3f}, Metrica {s[met].mean():.3f})  "
            f"macroF1 {np.mean([r['macro_f1'] for r in rows]):.3f}  "
            f"possessions x{np.mean([r['n_pred_spells'] / r['n_gt_spells'] for r in rows]):.2f}  "
            f"team acc {np.mean([r['team_acc'] for r in rows]):.3f}  "
            f"alive/dead F1 {np.mean([r['alive_dead_f1'] for r in rows]):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v3")
    parser.add_argument("--key", default="prob", help="prob, or rule_prob for the rule model")
    args = parser.parse_args()

    record = manifest(tag=args.tag, key=args.key)
    folds = load(args.tag)
    ids = sorted(folds)
    print(f"{len(ids)} folds: {' '.join(ids)}")
    configs = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
    grid = np.array([[score(folds[i], c, args.key)["spell_f1"] for i in ids] for c in configs])

    means = grid.mean(1)
    best = int(means.argmax())
    print(f"\nbest overall: {configs[best]}  mean spellF1 {means[best]:.3f}")
    for rank in means.argsort()[::-1][1:4]:
        print(f"  next: {configs[rank]}  {means[rank]:.3f}")

    print(f"whole grid spans {means.min():.3f} to {means.max():.3f}, which bounds any selection leak")

    rows, picks = [], []
    for j, match_id in enumerate(ids):
        others = np.delete(grid, j, axis=1).mean(1)
        picks.append(configs[int(others.argmax())])
        rows.append(score(folds[match_id], picks[-1], args.key))
    print("\nsettings chosen on the other matches' held-out predictions:")
    print(" ", table(rows, ids))
    print("\nper match:", " ".join(f"{k}={r['spell_f1']:.2f}" for k, r in zip(ids, rows)))

    record.update(summary=table(rows, ids), best=str(configs[best]),
                  grid_span=[float(means.min()), float(means.max())],
                  per_match={k: {"spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"], "decoder": str(c)}
                             for k, r, c in zip(ids, rows, picks)})
    save_manifest(os.path.join("runs", "oof", args.tag, f"sweep_{args.key}.json"), record)
