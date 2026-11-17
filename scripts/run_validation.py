"""Score predictors with leave-one-match-out validation on degraded open matches.

    python3 scripts/run_validation.py nearest:50 nearest:125
    python3 scripts/run_validation.py nearest:50 --clean --bootstrap 1000

The first predictor named is the reference for comparisons. Add your own to PREDICTORS. For the
learned model use scripts/train_oof.py and scripts/sweep_decoder.py, which train on randomised noise.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import datasets as ds, harness  # noqa: E402
from mtah.baselines import NearestPlayer  # noqa: E402
from mtah.degrade import DegradeConfig  # noqa: E402
from mtah.model import RuleModel  # noqa: E402

PREDICTORS = {
    # name: (factory taking the text after the colon, needs_training)
    "nearest": (lambda arg: NearestPlayer(int(arg or 50)), False),
    "rule": (lambda arg: RuleModel(), False),
}


def build(spec):
    name, _, arg = spec.partition(":")
    factory, needs_training = PREDICTORS[name]
    return (lambda: factory(arg)), needs_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictors", nargs="+")
    parser.add_argument("--matches", nargs="*", help="default: every open match on disk")
    parser.add_argument("--clean", action="store_true", help="skip the degrader")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    ids = args.matches or ds.available()
    cfg = None if args.clean else DegradeConfig()
    print(f"{len(ids)} matches, {'clean' if args.clean else 'degraded'}, seed {args.seed}\n")
    print(f"{'predictor':<20} {'spellF1':>8} {'sd':>6} {'macroF1':>8} {'poss x':>7} {'transF1':>8}")
    results = []
    for spec in args.predictors:
        make, needs_training = build(spec)
        rows = harness.run_lomo(make, ids, cfg, args.seed, args.bootstrap, needs_training)
        s = harness.summarise(rows)
        print(f"{s['name']:<20} {s['spell_f1']:>8.3f} {s['spell_f1_sd']:>6.3f} {s['macro_f1']:>8.3f}"
              f" {s['poss_ratio']:>7.2f} {s['transition_f1_25']:>8.3f}")
        results.append(rows)
        harness.save_run(rows, s["name"])
    for rows in results[1:]:
        c = harness.compare(results[0], rows)
        print(f"\n{rows[0]['name']} vs {results[0][0]['name']}: {c['mean_gain']:+.3f} mean spellF1, "
              f"{c['wins']} wins, {c['losses']} losses of {c['matches']}: {c['verdict']}")
