"""Quick screen of a model change: train once, score three held-out matches in a few conditions.

Screening only. A change that looks good here still has to win scripts/train_oof.py on all 10
matches and the stress test. Open data only.

    python3 scripts/dev_split.py --weight 0.5 0.75
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER, SPELL_WEIGHT, TRAIN_COPIES  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import LearnedModel  # noqa: E402
from mtah.provenance import manifest  # noqa: E402

HELD_OUT = ("J03WR9", "J03WOH", "METRICA2")
TURNOVERS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
R = dataclasses.replace
BASE = DegradeConfig()
CONDITIONS = {"as trained": BASE, "ball noise 3 m": R(BASE, ball_noise_m=3.0),
              "white ball 1.5 m": R(BASE, ball_noise_m=1.5, noise_rho=0.5),
              "false ball 15%": R(BASE, ball_false_rate=0.15), "clean": None}

def found_short(gt, pred, limit_s=4.0):
    runs = m.valid_runs(gt)
    gs, ps = m.possession_spells(gt, runs), m.possession_spells(pred, runs)
    hit = m._matched(m._overlaps(gs, ps), len(gs), 0.5)
    short = np.array([(g[1] - g[0]) / 25 < limit_s for g in gs])
    return float(hit[short].mean())


def decode(prob, frames, half, match, cfg):
    return expand(decode_samples(prob, half, cfg), frames, half, match.halves, match.n_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weight", nargs="+", type=float, default=[SPELL_WEIGHT])
    parser.add_argument("--held", nargs="*", default=list(HELD_OUT))
    args = parser.parse_args()

    record = manifest(held_out=args.held)
    stats = load_stats()
    ids = [i for i in ds.available() if i not in args.held]
    tests = {name: [(harness.harmonise(ds.load_open(h)) if cfg is None else
                     harness.prepared(h, cfg, harness.TEST_SEED_OFFSET, stats)) for h in args.held]
             for name, cfg in CONDITIONS.items()}
    os.makedirs("runs/dev", exist_ok=True)
    for weight in args.weight:
        started = time.time()
        model = LearnedModel(decoder=DECODER, spell_weight=weight)
        model.fit(harness.training_matches(ids, 0, TRAIN_COPIES, stats))
        print(f"\n=== weight {weight}  ({len(model.names)} features, trained in {time.time() - started:.0f}s)", flush=True)
        results = {}
        for name, matches in tests.items():
            probs = [model.probabilities(harness.without_labels(t)) for t in matches]
            sweep = {}
            for turnover in TURNOVERS:
                cfg = R(DECODER, turnover=turnover)
                preds = [decode(*p, t, cfg) for p, t in zip(probs, matches)]
                rows = [m.evaluate(t.labels, q) for q, t in zip(preds, matches)]
                sweep[turnover] = (float(np.mean([r["spell_f1"] for r in rows])), rows, preds)
            best = max(sweep, key=lambda k: sweep[k][0])
            s, rows, preds = sweep[DECODER.turnover]
            short = np.mean([found_short(t.labels, q) for q, t in zip(preds, matches)])
            print(f"{name:<18} spell {s:.3f} [{' '.join(f'{r['spell_f1']:.3f}' for r in rows)}]  macro "
                  f"{np.mean([r['macro_f1'] for r in rows]):.3f}  poss x"
                  f"{np.mean([r['n_pred_spells'] / r['n_gt_spells'] for r in rows]):.2f}  short {short:.3f}"
                  f"  best turnover {best}: {sweep[best][0]:.3f}", flush=True)
            results[name] = {"spell": s, "per_match": [r["spell_f1"] for r in rows], "short": float(short),
                             "best_turnover": best, "best": sweep[best][0]}
        with open(os.path.join("runs/dev", f"weight_{weight}.json"), "w") as handle:
            json.dump({"manifest": record, "results": results}, handle, indent=1)
