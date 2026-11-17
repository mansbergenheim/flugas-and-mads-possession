"""Is it the extra inputs or the averaging? Three v5 models averaged, with no new inputs.

The half of the v6 candidate that held up changes two things at once: 69 extra inputs, and three
models from different random draws averaged. This is the control with only the second: three models
with v5's own 129 inputs. Draw 0 is v5 itself. v5's decoder setting throughout, so nothing is picked.

Four tests, the same as for the candidate's pieces:
  - the 10 open matches, held-out (needs scripts/train_oof.py --tag v5_d1 --draw 1, and v5_d2 --draw 2;
    experiments/v6_label_styles.py then adds the stricter and looser labels),
  - SkillCorner's working half (the reserve matches are never loaded here),
  - the stress table (three held-out matches, 19 conditions),
  - trained on DFL and tested on Metrica, and the reverse.
Beside each result is the candidate without its smoother, from the earlier runs, for comparison.
Open data only. Nothing here changes v5.

    python3 experiments/v6_averaging_only.py      # about 75 minutes
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
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, degrade, load_stats  # noqa: E402
from mtah.model import LearnedModel, TreeConfig  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from skillcorner_v6 import DEV, V5_MODEL, V6_MODEL  # noqa: E402
from stress_test import BASE, CONDITIONS, COPIES, HELD_OUT, blank_ball  # noqa: E402
from sweep_decoder import score  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

V5_DRAWS = ("v5", "v5_d1", "v5_d2")
ONE, THREE, CANDIDATE = "v5", "v5 inputs, 3 draws averaged", "extra inputs, 3 draws averaged"
DIRECTIONS = (("DFL", ds.DFL_IDS, "Metrica", ds.METRICA_IDS), ("Metrica", ds.METRICA_IDS, "DFL", ds.DFL_IDS))
OUT = os.path.join("runs", "v6_averaging_only.json")


def grade(match, prob, frames, half):
    pred = expand(decode_samples(prob, half, DECODER), frames, half, match.halves, match.n_frames)
    got = m.evaluate(match.labels, pred)
    return got["spell_f1"], got["n_pred_spells"] / got["n_gt_spells"]


def three(ids, copies, stats, first=None):
    """Models with v5's inputs from draws 0, 1 and 2 on the same matches. Draw 0 is v5's own recipe."""
    models = [] if first is None else [first]
    for draw in range(len(models), 3):
        model = LearnedModel(decoder=DECODER, trees=TreeConfig(seed=draw))
        model.fit(harness.training_matches(ids, draw, copies, stats))
        models.append(model)
    return models


def both(models, blind):
    """The first model's probabilities, and the average of all, on the same samples."""
    values, swapped, frames, half = models[0].views(blind)
    each = [model.probabilities_from(values, swapped) for model in models]
    return each[0], sum(each) / float(len(each)), frames, half


def line(name, scores, base, counts=None):
    gain = np.array(scores) - np.array(base)
    count = "" if counts is None else f"  x{np.mean(counts):.2f}"
    print(f"  {name:<34}{np.mean(scores):.3f}{count}  {gain.mean():+.3f}  {(gain > 0).sum()} up / "
          f"{(gain < 0).sum()} down", flush=True)
    return {"spell_f1": float(np.mean(scores)), "gain": float(gain.mean()), "up": int((gain > 0).sum()),
            "down": int((gain < 0).sum()), "scores": [float(s) for s in scores]}


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(matches=list(DEV), held_out=list(HELD_OUT), draws=list(V5_DRAWS))
    results = {}
    stats = load_stats()

    # 1. The 10 open matches, from saved held-out predictions.
    if all(os.path.isdir(os.path.join("runs", "oof", tag)) for tag in V5_DRAWS):
        sets = {ONE: load(V5_DRAWS[0]), THREE: averaged(V5_DRAWS), CANDIDATE: averaged(DRAWS)}
        ids = sorted(sets[ONE])
        print(f"the {len(ids)} open matches, held-out")
        results["open matches"], base = {}, None
        for name, folds in sets.items():
            got = [score(folds[i], DECODER) for i in ids]
            scores = [g["spell_f1"] for g in got]
            base = scores if base is None else base
            results["open matches"][name] = line(name, scores, base, [g["n_pred_spells"] / g["n_gt_spells"] for g in got])
    else:
        print("the 10 open matches: skipped, no held-out predictions under runs/oof/v5_d1 and v5_d2 yet")

    # 2. SkillCorner's working half.
    with open(V5_MODEL, "rb") as handle:
        v5 = pickle.load(handle)
    with open(V6_MODEL, "rb") as handle:
        candidate = pickle.load(handle)["ensemble"]
    started = time.time()
    models = three(ds.available(), TRAIN_COPIES, stats, first=v5)
    print(f"\nSkillCorner's working half (two more models trained on {len(ds.available())} open matches in "
          f"{time.time() - started:.0f}s)", flush=True)
    rows = {name: ([], []) for name in (ONE, THREE, CANDIDATE)}
    for match_id in DEV:
        match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        blind = harness.without_labels(match)
        one, mean, frames, half = both(models, blind)
        prob6, frames6, half6 = candidate.probabilities(blind)
        for name, (p, f, h) in zip(rows, ((one, frames, half), (mean, frames, half), (prob6, frames6, half6))):
            spell, count = grade(match, p, f, h)
            rows[name][0].append(spell)
            rows[name][1].append(count)
        print(f"  {match_id}  " + "  ".join(f"{rows[name][0][-1]:.3f}" for name in rows), flush=True)
    results["SkillCorner"] = {name: line(name, s, rows[ONE][0], c) for name, (s, c) in rows.items()}

    # 3. The stress table.
    ids = [i for i in ds.available() if i not in HELD_OUT]
    started = time.time()
    models = three(ids, COPIES, stats)
    with open(os.path.join("runs", "stress_v6_parts.json")) as handle:
        earlier = json.load(handle)["results"]
    print(f"\nstress table (three models trained on {len(ids)} matches in {time.time() - started:.0f}s)")
    print(f"  {'condition':<24}{'v5':>8}{'v5 x3':>10}{'gain':>9}{'candidate':>12}{'gain':>9}", flush=True)
    results["stress"], gains3, gains6, gap = {}, [], [], 0.0
    for name, cfg in CONDITIONS.items():
        alone, together = [], []
        for held in HELD_OUT:
            clean = harness.harmonise(ds.load_open(held))
            test = degrade(clean, BASE, 1000, stats) if cfg == "no_ball" else degrade(clean, cfg, 1000, stats)
            blind = harness.without_labels(blank_ball(test) if cfg == "no_ball" else test)
            one, mean, frames, half = both(models, blind)
            alone.append(grade(test, one, frames, half)[0])
            together.append(grade(test, mean, frames, half)[0])
        base, mine = float(np.mean(alone)), float(np.mean(together))
        theirs = float(np.mean(earlier[name][CANDIDATE]["v5 setting"])) if name in earlier else float("nan")
        if name in earlier:
            gap = max(gap, abs(base - float(np.mean(earlier[name]["v5"]["own"]))))
        gains3.append(mine - base)
        gains6.append(theirs - base)
        results["stress"][name] = {ONE: [float(s) for s in alone], THREE: [float(s) for s in together],
                                  CANDIDATE: theirs}
        print(f"  {name:<24}{base:>8.3f}{mine:>10.3f}{mine - base:>+9.3f}{theirs:>12.3f}{theirs - base:>+9.3f}", flush=True)
    for name, gains in ((THREE, np.array(gains3)), (CANDIDATE, np.array(gains6))):
        print(f"  {name}: mean {np.nanmean(gains):+.4f}, rows up {(gains > 0).sum()}, down {(gains < 0).sum()}, "
              f"worst {np.nanmin(gains):+.4f}")
    print(f"  v5 column repeats runs/stress_v6_parts.json: {'yes' if gap < 1e-12 else 'NO'} (largest gap {gap:.2e})",
          flush=True)

    # 4. A label source the models never saw.
    with open(os.path.join("runs", "source_held_out_v6.json")) as handle:
        earlier = json.load(handle)["results"]
    results["unseen source"] = {}
    everything = {name: [] for name in (ONE, THREE, CANDIDATE)}
    for train_name, train_ids, test_name, test_ids in DIRECTIONS:
        started = time.time()
        models = three(list(train_ids), TRAIN_COPIES, stats)
        was = earlier.get(f"{train_name} to {test_name}", {})
        scores = {name: [] for name in everything}
        for i in test_ids:
            test = harness.prepared(i, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
            one, mean, frames, half = both(models, harness.without_labels(test))
            scores[ONE].append(grade(test, one, frames, half)[0])
            scores[THREE].append(grade(test, mean, frames, half)[0])
            scores[CANDIDATE].append(was.get(CANDIDATE, {}).get("v5_setting", {}).get(i, float("nan")))
        print(f"\ntrained on {train_name} only, tested on {test_name} ({time.time() - started:.0f}s)")
        before = was.get(ONE, {}).get("v5_setting", {})
        if before:
            gap = max(abs(s - before[i]) for s, i in zip(scores[ONE], test_ids))
            print(f"  v5 repeats runs/source_held_out_v6.json: {'yes' if gap < 1e-12 else 'NO'} (largest gap {gap:.2e})")
        results["unseen source"][f"{train_name} to {test_name}"] = {
            name: line(name, scores[name], scores[ONE]) for name in scores}
        for name in scores:
            everything[name].extend(scores[name])
    print("\nall matches, each tested by models that never saw its source")
    results["unseen source"]["all"] = {name: line(name, everything[name], everything[ONE]) for name in everything}

    record.update(results=results)
    save_manifest(OUT, record)
