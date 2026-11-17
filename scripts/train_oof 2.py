"""Train one model per held-out match and save its held-out probabilities.

Decoder settings can then be explored in seconds with scripts/sweep_decoder.py, with no retraining.

    python3 scripts/train_oof.py --tag v3
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import TRAIN_COPIES  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import LearnedModel, RuleModel, TreeConfig  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v3")
    parser.add_argument("--matches", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra", action="store_true", help="add the v6 candidate's extra inputs")
    parser.add_argument("--draw", type=int, help="random draw of the training copies and trees, if it "
                        "should differ from --seed; the test copies follow --seed only")
    parser.add_argument("--resume", action="store_true", help="keep folds already saved under this tag")
    args = parser.parse_args()
    draw = args.seed if args.draw is None else args.draw

    ids = args.matches or ds.available()
    cfg, stats = DegradeConfig(), load_stats()
    folder = os.path.join("runs", "oof", args.tag)
    os.makedirs(folder, exist_ok=True)
    record = manifest(held_out=list(ids), seed=args.seed, draw=draw, extra=args.extra, prob_sha256={})
    for held_out in ids:
        started = time.time()
        saved = os.path.join(folder, f"{held_out}.npz")
        if args.resume and os.path.exists(saved):
            with np.load(saved, allow_pickle=False) as z:
                record["prob_sha256"][held_out] = hashlib.sha256(z["prob"].tobytes()).hexdigest()
            save_manifest(os.path.join(folder, "manifest.json"), record)
            print(f"{held_out} kept from an earlier run", flush=True)
            continue
        test = harness.prepared(held_out, cfg, args.seed + harness.TEST_SEED_OFFSET, stats)
        blind = harness.without_labels(test)

        rule_prob, frames, half = RuleModel().probabilities(blind)
        model = LearnedModel(trees=TreeConfig(seed=draw), extra=args.extra)
        model.fit(harness.training_matches([i for i in ids if i != held_out], draw,
                                           TRAIN_COPIES, stats))
        prob, _, _ = model.probabilities(blind)
        np.savez_compressed(saved, prob=prob, rule_prob=rule_prob,
                            frames=frames, half=half, labels=test.labels,
                            halves=np.array(test.halves), source=test.source)
        record["prob_sha256"][held_out] = hashlib.sha256(prob.tobytes()).hexdigest()
        save_manifest(os.path.join(folder, "manifest.json"), record)
        print(f"{held_out} done in {time.time() - started:.0f}s", flush=True)
    print("all folds done", flush=True)
