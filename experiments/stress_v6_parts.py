"""Why the candidate fell below v5 in the harshest stress row: its models, or its decoder setting?

experiments/stress_v6.py scores the candidate with the decoder setting evaluate_v6.py chose for it on
the 10 open matches, against v5 with the shipped setting, so the two differ in both respects. This
repeats the table with every piece of the candidate decoded twice: once with its own setting and
once with v5's. It also keeps every piece's probabilities under runs/stress_v6_parts, so a later
question about the decoder needs no new training.

v5 is trained here exactly as scripts/stress_test.py trains it and the candidate as stress_v6.py
does, so two columns must repeat runs/stress_test.json and runs/stress_test_v6.json. The script says
whether they do. Open data only.

    python3 experiments/stress_v6_parts.py      # about 40 minutes
"""

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
from mtah.degrade import degrade, load_stats  # noqa: E402
from mtah.model import LearnedModel, TreeConfig  # noqa: E402
from mtah.provenance import manifest  # noqa: E402
from stress_test import BASE, CONDITIONS, COPIES, HELD_OUT, blank_ball  # noqa: E402
from v6_common import DRAWS, MAX_STRESS_LOSS, averaged, fit_smoother, load  # noqa: E402

KEEP = os.path.join("runs", "stress_v6_parts")
OUT = os.path.join("runs", "stress_v6_parts.json")


def grade(match, prob, frames, half, cfg):
    pred = expand(decode_samples(prob, half, cfg), frames, half, match.halves, match.n_frames)
    got = m.evaluate(match.labels, pred)
    return got["spell_f1"], got["n_pred_spells"] / got["n_gt_spells"]


def file_name(condition, held):
    return os.path.join(KEEP, condition.replace(" ", "_").replace("%", "pct") + "__" + held + ".npz")


if __name__ == "__main__":
    os.chdir(ROOT)
    os.makedirs(KEEP, exist_ok=True)
    record = manifest(held_out=list(HELD_OUT), copies=COPIES, draws=list(DRAWS))
    with open(os.path.join("runs", "oof", "v6", "summary.json")) as handle:
        home = json.load(handle)["results"]
    with open(os.path.join("runs", "stress_test.json")) as handle:
        earlier_v5 = {name: row["spell_f1"] for name, row in json.load(handle)["results"].items()}
    with open(os.path.join("runs", "stress_test_v6.json")) as handle:
        earlier_v6 = {name: row["candidate"] for name, row in json.load(handle)["results"].items()}
    candidate = list(home)[-1]
    averaged_name = candidate.replace(", smoothed", "")
    pieces = ["v5", "v5, smoothed", "extra inputs", "extra inputs, smoothed", averaged_name, candidate]
    own = {name: DecoderConfig(**home[name]["best_overall_config"]) for name in pieces}
    own["v5"] = DECODER

    stats = load_stats()
    ids = [i for i in ds.available() if i not in HELD_OUT]
    started = time.time()
    v5 = LearnedModel(decoder=DECODER)
    v5.fit(harness.training_matches(ids, 0, COPIES, stats))
    models = []
    for draw in range(len(DRAWS)):
        model = LearnedModel(trees=TreeConfig(seed=draw), extra=True)
        model.fit(harness.training_matches(ids, draw, COPIES, stats))
        models.append(model)
    smoother5 = fit_smoother(load("v5"), ids)
    smoother1 = fit_smoother(load(DRAWS[0]), ids)
    smoother3 = fit_smoother(averaged(DRAWS), ids)
    print(f"trained 4 models on {len(ids)} matches and 3 smoothers in {time.time() - started:.0f}s\n", flush=True)

    results = {}
    for name, cfg in CONDITIONS.items():
        rows = {piece: {"own": [], "v5 setting": [], "count own": [], "count v5 setting": []} for piece in pieces}
        for held in HELD_OUT:
            clean = harness.harmonise(ds.load_open(held))
            test = degrade(clean, BASE, 1000, stats) if cfg == "no_ball" else degrade(clean, cfg, 1000, stats)
            blind = harness.without_labels(blank_ball(test) if cfg == "no_ball" else test)
            p5, frames5, half5 = v5.probabilities(blind)
            values, swapped, frames, half = models[0].views(blind)
            if not (np.array_equal(frames, frames5) and np.array_equal(half, half5)):
                raise ValueError("the two sets of inputs are not sampled at the same moments")
            each = [model.probabilities_from(values, swapped) for model in models]
            mean = sum(each) / float(len(each))
            prob = {"v5": p5, "v5, smoothed": smoother5.probabilities(p5, half),
                    "extra inputs": each[0], "extra inputs, smoothed": smoother1.probabilities(each[0], half),
                    averaged_name: mean, candidate: smoother3.probabilities(mean, half)}
            np.savez_compressed(file_name(name, held), labels=test.labels, halves=np.array(test.halves),
                                frames=frames, half=half, pieces=np.array(pieces),
                                **{f"prob{i}": prob[piece] for i, piece in enumerate(pieces)})
            for piece in pieces:
                score, count = grade(test, prob[piece], frames, half, own[piece])
                rows[piece]["own"].append(score)
                rows[piece]["count own"].append(count)
                score, count = grade(test, prob[piece], frames, half, DECODER)
                rows[piece]["v5 setting"].append(score)
                rows[piece]["count v5 setting"].append(count)
        results[name] = rows
        base = float(np.mean(rows["v5"]["own"]))
        print(f"{name:<24}v5 {base:.3f} (x{np.mean(rows['v5']['count own']):.2f})   candidate, own setting "
              f"{np.mean(rows[candidate]['own']):.3f} ({np.mean(rows[candidate]['own']) - base:+.3f}, "
              f"x{np.mean(rows[candidate]['count own']):.2f})   candidate, v5's setting "
              f"{np.mean(rows[candidate]['v5 setting']):.3f} ({np.mean(rows[candidate]['v5 setting']) - base:+.3f}, "
              f"x{np.mean(rows[candidate]['count v5 setting']):.2f})", flush=True)

    def table(key, title):
        print(f"\n{title}\n{'condition':<24}" + "".join(f"{p[:22]:>24}" for p in pieces) + f"{'candidate - v5':>16}")
        worst = 0.0
        for name, rows in results.items():
            base = float(np.mean(rows["v5"]["own"]))
            cells = [float(np.mean(rows[p]["own" if p == "v5" else key])) for p in pieces]
            gain = cells[-1] - base
            worst = min(worst, gain)
            flag = "   <-- more than 0.01 below v5" if gain < -MAX_STRESS_LOSS else ""
            print(f"{name:<24}" + "".join(f"{c:>24.3f}" for c in cells) + f"{gain:>+16.3f}{flag}")
        print(f"worst row for the candidate: {worst:+.3f}")
        return worst

    worst_own = table("own", "Each piece with the decoder setting chosen for it on the 10 open matches")
    worst_same = table("v5 setting", "Each piece with v5's decoder setting")

    same_v5 = max(abs(float(np.mean(results[n]["v5"]["own"])) - earlier_v5[n]) for n in results)
    same_v6 = max(abs(float(np.mean(results[n][candidate]["own"])) - earlier_v6[n]) for n in results)
    print(f"\nv5 column repeats runs/stress_test.json: {'yes' if same_v5 < 1e-12 else 'NO'} (largest gap {same_v5:.2e})")
    print(f"candidate column repeats runs/stress_test_v6.json: {'yes' if same_v6 < 1e-12 else 'NO'} "
          f"(largest gap {same_v6:.2e})")
    with open(OUT, "w") as handle:
        json.dump({"manifest": record, "pieces": pieces, "own_settings": {p: str(own[p]) for p in pieces},
                   "results": results, "worst_own": worst_own, "worst_v5_setting": worst_same},
                  handle, indent=1, default=float)
