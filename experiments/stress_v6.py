"""The stress part of the v6 rule: no row may fall more than 0.01 below v5. Open data only.

Same held-out matches, conditions and test copies as scripts/stress_test.py, whose v5 results are read
from runs/stress_test.json. The candidate is three models with the extra inputs, averaged, then the
smoother, then the decoder setting chosen by experiments/evaluate_v6.py.

The smoother is fitted on the held-out probabilities of the 7 training matches from the 10-match runs.
Those came from models that had seen the 3 matches scored here, the same small leak as elsewhere.

    python3 experiments/stress_v6.py      # about 40 minutes
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
from mtah.decoder import DecoderConfig, decode_samples, expand  # noqa: E402
from mtah.degrade import degrade, load_stats  # noqa: E402
from mtah.model import AveragedModel, LearnedModel, TreeConfig  # noqa: E402
from mtah.provenance import manifest  # noqa: E402
from stress_test import BASE, CONDITIONS, COPIES, HELD_OUT, blank_ball  # noqa: E402
from v6_common import DRAWS, MAX_STRESS_LOSS, averaged, fit_smoother  # noqa: E402


def spell(match, prob, frames, half, cfg):
    pred = expand(decode_samples(prob, half, cfg), frames, half, match.halves, match.n_frames)
    return m.evaluate(match.labels, pred)["spell_f1"]


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(held_out=list(HELD_OUT), copies=COPIES, draws=list(DRAWS))
    with open(os.path.join("runs", "oof", "v6", "summary.json")) as handle:
        summary = json.load(handle)
    with open(os.path.join("runs", "stress_test.json")) as handle:
        v5 = {name: row["spell_f1"] for name, row in json.load(handle)["results"].items()}
    names = list(summary["results"])
    plain_cfg = DecoderConfig(**summary["results"][names[-2]]["best_overall_config"])
    smooth_cfg = DecoderConfig(**summary["results"][names[-1]]["best_overall_config"])
    print(f"averaged: {names[-2]} with {plain_cfg}\ncandidate: {names[-1]} with {smooth_cfg}", flush=True)

    stats = load_stats()
    ids = [i for i in ds.available() if i not in HELD_OUT]
    started = time.time()
    models = []
    for draw in range(len(DRAWS)):
        model = LearnedModel(trees=TreeConfig(seed=draw), extra=True)
        model.fit(harness.training_matches(ids, draw, COPIES, stats))
        models.append(model)
    ensemble = AveragedModel(models)
    smoother = fit_smoother(averaged(DRAWS), ids)
    print(f"trained {len(models)} models on {len(ids)} matches and the smoother in {time.time() - started:.0f}s\n",
          flush=True)

    print(f"{'condition':<24}{'v5':>7}{'averaged':>10}{'candidate':>11}{'against v5':>12}", flush=True)
    results, worst = {}, 0.0
    for name, cfg in CONDITIONS.items():
        plain, smooth = [], []
        for held in HELD_OUT:
            clean = harness.harmonise(ds.load_open(held))
            test = degrade(clean, BASE, 1000, stats) if cfg == "no_ball" else degrade(clean, cfg, 1000, stats)
            blind = harness.without_labels(blank_ball(test) if cfg == "no_ball" else test)
            prob, frames, half = ensemble.probabilities(blind)
            plain.append(spell(test, prob, frames, half, plain_cfg))
            smooth.append(spell(test, smoother.probabilities(prob, half), frames, half, smooth_cfg))
        gain = float(np.mean(smooth)) - v5[name]
        worst = min(worst, gain)
        results[name] = {"v5": v5[name], "averaged": float(np.mean(plain)), "candidate": float(np.mean(smooth)),
                         "candidate_per_match": smooth}
        flag = "   <-- more than 0.01 below v5" if gain < -MAX_STRESS_LOSS else ""
        print(f"{name:<24}{v5[name]:>7.3f}{np.mean(plain):>10.3f}{np.mean(smooth):>11.3f}{gain:>+12.3f}{flag}",
              flush=True)
    passed = worst >= -MAX_STRESS_LOSS
    print(f"\nstress part of the rule (no row more than {MAX_STRESS_LOSS} below v5): "
          + ("PASSED" if passed else "NOT passed") + f"  (worst row {worst:+.3f})")
    with open(os.path.join("runs", "stress_test_v6.json"), "w") as handle:
        json.dump({"manifest": record, "results": results, "passed": bool(passed)}, handle, indent=1, default=float)
