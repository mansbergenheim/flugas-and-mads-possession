"""Does the candidate's gain survive if the hidden labels are stricter or looser than ours?

We don't know how the competition's labels treat brief touches. The same held-out predictions of v5
and of the candidate are scored against four versions of the truth: DFL's raw flags (looser than
ours; the 7 DFL matches only), our current labels, and two versions closer to the organisers'
written definition, where single-touch possessions under 2 s or under 4 s go back to the team that
had the ball (experiments/short_possessions.py). Nothing is retrained.

Each model is scored twice on each version: with v5's shipped decoder setting, and with the setting
that did best on the other matches for that version of the truth, so a model isn't marked down for
a decoder that suits other labels. Needs the smoothed predictions saved by
experiments/v6_same_decoder.py. Open data only.

The candidate's two unsmoothed pieces (one model with the extra inputs, and three averaged) are in
the table as well, added on 11 Nov after the stress table by piece showed the smoother to be the
part that breaks.

    python3 experiments/v6_label_styles.py      # about 15 minutes
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import DecoderConfig, decode_samples, expand  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from short_possessions import follow_definition  # noqa: E402
from sweep_decoder import GRID, load  # noqa: E402
from v6_common import DRAWS, averaged  # noqa: E402

RAW = "DFL's raw flags (looser)"
STYLES = {RAW: "raw", "our current labels": None, "single touches under 2 s folded": 2.0,
          "single touches under 4 s folded (the written definition)": 4.0}
# A name of a folder under runs/oof, or several of them to average.
TAGS = {"v5": "v5", "v5, smoothed": "v5_smoothed", "extra inputs": DRAWS[0],
        "extra inputs, 3 draws averaged": DRAWS, "candidate": "v6_avg_smoothed"}
# The control from experiments/v6_averaging_only.py, once its two extra draws have been run.
V5_DRAWS = ("v5", "v5_d1", "v5_d2")
if all(os.path.isdir(os.path.join(ROOT, "runs", "oof", tag)) for tag in V5_DRAWS):
    TAGS["v5 inputs, 3 draws averaged"] = V5_DRAWS
WIDTH = max(len(name) for name in TAGS) + 2
OUT = os.path.join("runs", "v6_label_styles.json")


def label_sets(ids):
    out = {}
    for match_id in ids:
        loaded = ds.load_open(match_id)
        match = harness.harmonise(loaded)
        out[match_id] = {}
        for name, limit in STYLES.items():
            if limit == "raw":
                if match_id in ds.DFL_IDS:
                    out[match_id][name] = loaded.labels
            else:
                out[match_id][name] = match.labels if limit is None else follow_definition(match, limit)
    return out


def score_grid(folds, labels, ids, configs):
    """spell[style][config, match] and the possession count ratio; NaN where a style has no labels."""
    spell = {name: np.full((len(configs), len(ids)), np.nan) for name in STYLES}
    ratio = {name: np.full((len(configs), len(ids)), np.nan) for name in STYLES}
    for c, cfg in enumerate(configs):
        for j, match_id in enumerate(ids):
            fold = folds[match_id]
            halves = [tuple(int(v) for v in r) for r in fold["halves"]]
            pred = expand(decode_samples(fold["prob"], fold["half"], cfg), fold["frames"], fold["half"],
                          halves, len(fold["labels"]))
            for name, truth in labels[match_id].items():
                result = m.evaluate(truth, pred)
                spell[name][c, j] = result["spell_f1"]
                ratio[name][c, j] = result["n_pred_spells"] / result["n_gt_spells"]
    return spell, ratio


def held_out_pick(grid):
    """Each match scored with the setting that did best on the other matches. Columns must be complete."""
    n = grid.shape[1]
    picks = [int(np.delete(grid, j, axis=1).mean(1).argmax()) for j in range(n)]
    return np.array([grid[p, j] for j, p in enumerate(picks)]), picks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="two decoder settings only, to test the script")
    args = parser.parse_args()
    os.chdir(ROOT)
    record = manifest(tags={name: tag if isinstance(tag, str) else list(tag) for name, tag in TAGS.items()},
                      quick=args.quick)

    folds = {name: load(tag) if isinstance(tag, str) else averaged(tag) for name, tag in TAGS.items()}
    ids = sorted(folds["v5"])
    configs = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
    if args.quick:
        configs = [DECODER, DecoderConfig(1.0, 1.5, 6.0, 6.0)]
    shipped = configs.index(DECODER)
    started = time.time()
    labels = label_sets(ids)
    for match_id in ids:
        for name in TAGS:
            if not np.array_equal(folds[name][match_id]["labels"], labels[match_id]["our current labels"]):
                raise ValueError(f"{name}, {match_id}: saved labels aren't the current labels")
    print(f"four versions of the truth for {len(ids)} matches ready in {time.time() - started:.0f}s", flush=True)
    grids = {}
    for name in TAGS:
        started = time.time()
        grids[name] = score_grid(folds[name], labels, ids, configs)
        print(f"{name}: {len(configs)} decoder settings scored in {time.time() - started:.0f}s", flush=True)

    results = {}
    for style in STYLES:
        have = np.array([style in labels[i] for i in ids])
        possessions = np.mean([len(m.possession_spells(labels[i][style], m.valid_runs(labels[i][style])))
                               for i in ids if style in labels[i]])
        print(f"\nscored on: {style}  ({have.sum()} matches, {possessions:.0f} possessions a match)")
        print(f"{'':{WIDTH}}{'decoder picked on the other matches':>40}{'':8}{'v5 shipped decoder setting':>34}")
        print(f"{'':{WIDTH}}{'spellF1':>9}{'gain':>9}{'up/down':>9}{'count':>8}{'':13}{'spellF1':>9}{'gain':>9}{'up/down':>9}{'count':>8}")
        results[style] = {}
        base_pick = base_fixed = None
        for name in TAGS:
            spell, ratio = grids[name]
            picked, picks = held_out_pick(spell[style][:, have])
            fixed = spell[style][shipped, have]
            count_pick = float(np.mean([ratio[style][:, have][p, j] for j, p in enumerate(picks)]))
            count_fixed = float(ratio[style][shipped, have].mean())
            if base_pick is None:
                base_pick, base_fixed = picked, fixed
            g_pick, g_fixed = picked - base_pick, fixed - base_fixed
            results[style][name] = {
                "picked": float(picked.mean()), "picked_gain": float(g_pick.mean()),
                "picked_up": int((g_pick > 0).sum()), "picked_down": int((g_pick < 0).sum()),
                "picked_per_match": dict(zip([i for i, h in zip(ids, have) if h], picked.tolist())),
                "picked_count_ratio": count_pick,
                "shipped": float(fixed.mean()), "shipped_gain": float(g_fixed.mean()),
                "shipped_up": int((g_fixed > 0).sum()), "shipped_down": int((g_fixed < 0).sum()),
                "shipped_per_match": dict(zip([i for i, h in zip(ids, have) if h], fixed.tolist())),
                "shipped_count_ratio": count_fixed,
                "picked_settings": sorted({str(configs[p]) for p in picks})}
            r = results[style][name]
            print(f"{name:<{WIDTH}}{r['picked']:>9.3f}{r['picked_gain']:>+9.3f}{r['picked_up']:>6}/{r['picked_down']:<2}"
                  f"{'x%.2f' % count_pick:>8}{'':13}{r['shipped']:>9.3f}{r['shipped_gain']:>+9.3f}"
                  f"{r['shipped_up']:>6}/{r['shipped_down']:<2}{'x%.2f' % count_fixed:>8}", flush=True)
    record.update(results=results, matches=ids)
    if not args.quick:
        save_manifest(OUT, record)
