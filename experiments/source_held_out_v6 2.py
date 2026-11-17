"""Does the candidate's gain survive on a source whose labels it has never seen?

SkillCorner says no: every piece of the candidate is level with v5 there
(experiments/skillcorner_v6_parts.py). DFL and Metrica are the two sources we train on, and they are
labelled in different ways (DFL's own flags; our labels built from Metrica's events). Here one of
them is held out entirely: v5 and the candidate are trained on DFL only and tested on Metrica, and
the reverse. Same test copies as everywhere else.

The smoother is fitted on the training source's held-out probabilities from the 10-match runs, so it
never sees a label from the held-out source. Those probabilities came from first models that had
seen the held-out source, the same small leak as in experiments/stress_v6.py.

Reading fixed before the run (10 Nov 2026): the gain carries to an unseen source if, over the 10
matches tested this way and with v5's decoder setting for both, the candidate is at least 0.015
above v5 on average with at least 8 up. It does not carry if the average is under 0.005. Anything
between is unclear. This is evidence for the team's decision on v6, not a change to anything.

    python3 experiments/source_held_out_v6.py      # about 35 minutes
"""

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
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.decoder import DecoderConfig  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import LearnedModel, TreeConfig  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from sweep_decoder import GRID, score  # noqa: E402
from v6_common import DRAWS, MIN_GAIN, MIN_UP, averaged, fit_smoother, load  # noqa: E402

DIRECTIONS = (("DFL", ds.DFL_IDS, "Metrica", ds.METRICA_IDS), ("Metrica", ds.METRICA_IDS, "DFL", ds.DFL_IDS))
CONFIGS = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
NO_CARRY = 0.005
OUT = os.path.join("runs", "source_held_out_v6.json")


def picked_on_the_others(folds, ids):
    """Each test match with the decoder setting that did best on the other test matches."""
    grid = np.array([[score(folds[i], c)["spell_f1"] for i in ids] for c in CONFIGS])
    return np.array([grid[int(np.delete(grid, j, axis=1).mean(1).argmax()), j] for j in range(len(ids))])


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(draws=list(DRAWS))
    with open(os.path.join("runs", "oof", "v6", "summary.json")) as handle:
        home = json.load(handle)["results"]
    candidate = list(home)[-1]
    averaged_name = candidate.replace(", smoothed", "")
    pieces = ["v5", "v5, smoothed", averaged_name, candidate]
    own = {name: DecoderConfig(**home[name]["best_overall_config"]) for name in pieces}
    own["v5"] = DECODER
    cfg, stats = DegradeConfig(), load_stats()
    saved5, saved6 = load("v5"), averaged(DRAWS)

    results, all_v5, all_candidate, all_ids = {}, [], [], []
    for train_name, train_ids, test_name, test_ids in DIRECTIONS:
        train_ids, test_ids = list(train_ids), list(test_ids)
        started = time.time()
        v5 = LearnedModel(decoder=DECODER)
        v5.fit(harness.training_matches(train_ids, 0, TRAIN_COPIES, stats))
        models = []
        for draw in range(len(DRAWS)):
            model = LearnedModel(trees=TreeConfig(seed=draw), extra=True)
            model.fit(harness.training_matches(train_ids, draw, TRAIN_COPIES, stats))
            models.append(model)
        smoother5 = fit_smoother(saved5, train_ids)
        smoother3 = fit_smoother(saved6, train_ids)
        print(f"\ntrained on {train_name} only ({len(train_ids)} matches) in {time.time() - started:.0f}s; "
              f"tested on {test_name} ({len(test_ids)} matches)", flush=True)

        folds = {piece: {} for piece in pieces}
        for i in test_ids:
            test = harness.prepared(i, cfg, harness.TEST_SEED_OFFSET, stats)
            blind = harness.without_labels(test)
            p5, frames5, half5 = v5.probabilities(blind)
            values, swapped, frames, half = models[0].views(blind)
            if not (np.array_equal(frames, frames5) and np.array_equal(half, half5)):
                raise ValueError("the two sets of inputs are not sampled at the same moments")
            mean = sum(model.probabilities_from(values, swapped) for model in models) / float(len(models))
            prob = {"v5": p5, "v5, smoothed": smoother5.probabilities(p5, half), averaged_name: mean,
                    candidate: smoother3.probabilities(mean, half)}
            for piece in pieces:
                folds[piece][i] = {"prob": prob[piece], "frames": frames, "half": half, "labels": test.labels,
                                   "halves": np.array(test.halves)}

        base = np.array([score(folds["v5"][i], DECODER)["spell_f1"] for i in test_ids])
        base_picked = picked_on_the_others(folds["v5"], test_ids)
        print(f"{'':52}{'v5 setting':>11}{'gain':>8} up/down{'own setting':>13}{'gain':>8}{'picked on test':>16}{'gain':>8}")
        results[f"{train_name} to {test_name}"] = {}
        for piece in pieces:
            same = np.array([score(folds[piece][i], DECODER)["spell_f1"] for i in test_ids])
            mine = np.array([score(folds[piece][i], own[piece])["spell_f1"] for i in test_ids])
            picked = picked_on_the_others(folds[piece], test_ids)
            gain = same - base
            results[f"{train_name} to {test_name}"][piece] = {
                "v5_setting": dict(zip(test_ids, same.tolist())), "own_setting": dict(zip(test_ids, mine.tolist())),
                "picked_on_test": dict(zip(test_ids, picked.tolist()))}
            print(f"{piece:<52}{same.mean():>11.3f}{gain.mean():>+8.3f}  {(gain > 0).sum()}/{(gain < 0).sum()}  "
                  f"{mine.mean():>11.3f}{(mine - base).mean():>+8.3f}{picked.mean():>16.3f}"
                  f"{(picked - base_picked).mean():>+8.3f}", flush=True)
            if piece == candidate:
                print("      per match, v5's setting: " + " ".join(f"{i}:{g:+.3f}" for i, g in zip(test_ids, gain)))
                all_v5.extend(base.tolist())
                all_candidate.extend(same.tolist())
                all_ids.extend(test_ids)

    gain = np.array(all_candidate) - np.array(all_v5)
    up = int((gain > 0).sum())
    if gain.mean() >= MIN_GAIN and up >= MIN_UP:
        reading = "the gain CARRIES to an unseen source"
    elif gain.mean() < NO_CARRY:
        reading = "the gain does NOT carry to an unseen source"
    else:
        reading = "UNCLEAR: part of the gain carries"
    print(f"\nall {len(gain)} matches, each tested by models that never saw its source, v5's decoder setting for both:")
    print(f"v5 {np.mean(all_v5):.3f}, candidate {np.mean(all_candidate):.3f}, gain {gain.mean():+.3f}, "
          f"{up} up / {int((gain < 0).sum())} down")
    print(f"reading fixed before the run: {reading}")
    record.update(results=results, matches=all_ids, gain=float(gain.mean()), up=up, reading=reading)
    save_manifest(OUT, record)
