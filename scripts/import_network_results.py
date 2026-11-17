"""Bring a network's predictions back from Colab into the form the scoring scripts read. Open data only.

The notebooks in colab/ write one folder per round under results/<model>/ on Drive. Download that
folder into runs/network_results/<model>/ and run this. It writes the predictions in the same form
as scripts/train_oof.py, so every existing comparison works on them:

  runs/oof/net_<model>/           the 10 open matches, each predicted by networks that never saw it
  runs/oof/net_<model>_unseen/    the same matches, predicted by networks that never saw their source
  runs/oof/net_<model>_sc/        SkillCorner's working half, predicted by the networks trained on all 10
  runs/stress_networks/<model>/   the stress copies, in the layout of runs/stress_v6_parts

Probabilities are rounded to four decimals, as the prediction code on our own computers rounds them
(mtah/network.py). The reserve half of SkillCorner is never loaded.

    python3 scripts/import_network_results.py --model tcn
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.network import DECIMALS  # noqa: E402
from stress_test import CONDITIONS, HELD_OUT  # noqa: E402
from sweep_decoder import score  # noqa: E402

PACK = os.path.join("runs", "network_pack")
TREES_STRESS = os.path.join("runs", "stress_v6_parts")


def prob_of(path, key="prob"):
    with np.load(path, allow_pickle=False) as z:
        return np.round(z[key].astype(np.float64), DECIMALS)


def like_the_trees(match_id):
    """The frames, labels and halves of a match's standard test copy, from the trees' held-out file."""
    with np.load(os.path.join("runs", "oof", "v5", f"{match_id}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in ("frames", "half", "labels", "halves", "source")}


def write(folder, match_id, prob, shared):
    if len(prob) != len(shared["frames"]):
        raise ValueError(f"{match_id}: {len(prob)} predictions for {len(shared['frames'])} moments")
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, f"{match_id}.npz"), prob=prob, **shared)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="tcn or bilstm: the folder name under the results")
    parser.add_argument("--source", default=os.path.join("runs", "network_results"), help="where the results were unzipped")
    args = parser.parse_args()
    os.chdir(ROOT)
    source = os.path.join(args.source, args.model)
    if not os.path.isdir(source):
        sys.exit(f"no folder {source}: download results/{args.model} from Drive into {args.source} first")
    tag = f"net_{args.model}"
    done, absent = {}, []

    def have(*parts):
        path = os.path.join(source, *parts)
        if not os.path.exists(path):
            absent.append(os.path.join(*parts))
            return None
        return path

    # The 10 open matches, held out one at a time.
    for match_id in ds.available():
        path = have(f"fold_{match_id}", f"test_{match_id}.npz")
        if path:
            write(os.path.join("runs", "oof", tag), match_id, prob_of(path), like_the_trees(match_id))
            done.setdefault("held-out open matches", []).append(match_id)

    # The same matches, predicted by networks that never saw their source.
    for round_name, targets in (("dfl_only", ds.METRICA_IDS), ("metrica_only", ds.DFL_IDS)):
        for match_id in targets:
            path = have(round_name, f"test_{match_id}.npz")
            if path:
                write(os.path.join("runs", "oof", tag + "_unseen"), match_id, prob_of(path), like_the_trees(match_id))
                done.setdefault("unseen source", []).append(match_id)

    # SkillCorner's working half.
    stats = None
    for match_id in ds.SKILLCORNER_DEV:
        path = have("all", f"sc_{match_id}.npz")
        if path:
            stats = stats or load_stats()
            match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
            with np.load(os.path.join(PACK, f"sc_{match_id}.npz"), allow_pickle=False) as z:
                shared = {"frames": z["frames"], "half": z["half"], "labels": match.labels,
                          "halves": np.array(match.halves), "source": np.array(match.source)}
            write(os.path.join("runs", "oof", tag + "_sc"), match_id, prob_of(path), shared)
            done.setdefault("SkillCorner working half", []).append(match_id)

    # The stress copies.
    for held in HELD_OUT:
        path = have("stress", f"stress_{held}.npz")
        if not path:
            continue
        for index, condition in enumerate(CONDITIONS):
            name = condition.replace(" ", "_").replace("%", "pct") + f"__{held}.npz"
            with np.load(os.path.join(TREES_STRESS, name), allow_pickle=False) as z:
                shared = {k: z[k] for k in ("frames", "half", "labels", "halves")}
            prob = prob_of(path, f"prob{index}")
            if len(prob) != len(shared["frames"]):
                raise ValueError(f"{held}, {condition}: {len(prob)} predictions for {len(shared['frames'])} moments")
            folder = os.path.join("runs", "stress_networks", args.model)
            os.makedirs(folder, exist_ok=True)
            np.savez_compressed(os.path.join(folder, name), prob=prob, **shared)
        done.setdefault("stress matches", []).append(held)

    for what, matches in done.items():
        print(f"{what}: {len(matches)} written")
    if absent:
        print(f"not in {source} (round not run yet?): " + ", ".join(absent[:8]) + (" ..." if len(absent) > 8 else ""))

    # A first look: the network alone on the held-out open matches, with v5's decoder setting.
    matches = done.get("held-out open matches", [])
    if matches:
        def mean_score(folder):
            scores = []
            for match_id in matches:
                with np.load(os.path.join("runs", "oof", folder, f"{match_id}.npz"), allow_pickle=False) as z:
                    scores.append(score({k: z[k] for k in z.files}, DECODER)["spell_f1"])
            return float(np.mean(scores)), scores

        mine, each = mean_score(tag)
        base, base_each = mean_score("v5")
        up = sum(a > b for a, b in zip(each, base_each))
        print(f"\nfirst look, {len(matches)} held-out open matches, v5's decoder setting: v5 {base:.3f}, "
              f"{args.model} alone {mine:.3f} ({mine - base:+.3f}, {up} up / {len(matches) - up} not up)")
        print("the full comparison, alone and averaged with the trees, is experiments/compare_networks.py")
    rounds = [json.load(open(p)) for p in (os.path.join(source, r, "info.json") for r in sorted(os.listdir(source)))
              if os.path.exists(p)]
    if rounds:
        print(f"\n{len(rounds)} rounds done on Colab; steps {min(r['steps'] for r in rounds)} to "
              f"{max(r['steps'] for r in rounds)}, temperature {min(r['temperature'] for r in rounds)} to "
              f"{max(r['temperature'] for r in rounds)}, {sum(r['minutes'] for r in rounds):.0f} minutes in all")
