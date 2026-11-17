"""The 10-match part of the v6 rule: v5 against each piece of the candidate, on the same test copies.

Reads held-out probabilities saved by scripts/train_oof.py. Each match's score uses decoder settings
chosen on the other matches, as for v5. Open data only.

    python3 scripts/train_oof.py --tag v6_d0 --extra --draw 0      # about an hour each
    python3 scripts/train_oof.py --tag v6_d1 --extra --draw 1
    python3 scripts/train_oof.py --tag v6_d2 --extra --draw 2
    python3 experiments/evaluate_v6.py

With only some draws finished it reports what it can. --quick sweeps the turnover cost alone.
"""

import argparse
import dataclasses
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
from mtah import metrics as m  # noqa: E402
from mtah.decoder import DecoderConfig  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from sweep_decoder import GRID, score  # noqa: E402
from v6_common import DRAWS, MIN_GAIN, MIN_UP, averaged, load, smoothed  # noqa: E402

QUICK = {"temperature": (1.0,), "turnover": (1.5, 2.0, 2.5, 3.0, 4.0, 5.0), "to_dead": (6.0,), "from_dead": (4.0,)}


def nested(folds, ids, configs):
    """Per-match rows with the decoder picked on the other matches, and the best single setting."""
    grid = np.array([[score(folds[i], c)["spell_f1"] for i in ids] for c in configs])
    rows = []
    for j, match_id in enumerate(ids):
        pick = configs[int(np.delete(grid, j, axis=1).mean(1).argmax())]
        rows.append(score(folds[match_id], pick))
    best = int(grid.mean(1).argmax())
    return rows, configs[best], float(grid.mean(1)[best])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="v5")
    parser.add_argument("--draws", nargs="*", default=list(DRAWS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=os.path.join("runs", "oof", "v6", "summary.json"))
    args = parser.parse_args()
    os.chdir(ROOT)

    record = manifest(base=args.base, draws=args.draws, quick=args.quick)
    space = QUICK if args.quick else GRID
    configs = [DecoderConfig(*c) for c in itertools.product(*space.values())]
    base = load(args.base)
    ids = sorted(base)
    have = [t for t in args.draws if os.path.isdir(os.path.join("runs", "oof", t))
            and len(os.listdir(os.path.join("runs", "oof", t))) > len(ids)]
    variants = {"v5": base, "v5, smoothed": None}
    if have:
        first = load(have[0])
        variants["extra inputs"] = first
        variants["extra inputs, smoothed"] = None
    if len(have) > 1:
        variants[f"extra inputs, {len(have)} draws averaged"] = averaged(have)
        variants[f"extra inputs, {len(have)} draws averaged, smoothed"] = None
    for name in list(variants):
        if variants[name] is None:
            started = time.time()
            variants[name] = smoothed(variants[name.replace(", smoothed", "")])
            print(f"smoother fitted for '{name}' in {time.time() - started:.0f}s", flush=True)
    for name, folds in variants.items():
        for i in ids:
            if not np.array_equal(folds[i]["labels"], base[i]["labels"]):
                raise ValueError(f"{name}, {i}: not the same test copy as {args.base}")

    dfl = np.array([i.startswith("J") for i in ids])
    results, reference = {}, None
    print(f"\n{'':52}spellF1    DFL  Metrica  macroF1  count    gain  up/down")
    for name, folds in variants.items():
        rows, best, best_mean = nested(folds, ids, configs)
        s = np.array([r["spell_f1"] for r in rows])
        reference = s if reference is None else reference
        gain = s - reference
        results[name] = {"spell_f1": float(s.mean()), "per_match": dict(zip(ids, s.tolist())),
                         "gain": float(gain.mean()), "up": int((gain > 0).sum()), "down": int((gain < 0).sum()),
                         "best_overall": str(best), "best_overall_mean": best_mean,
                         "best_overall_config": dataclasses.asdict(best),
                         "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                         "count_ratio": float(np.mean([r["n_pred_spells"] / r["n_gt_spells"] for r in rows]))}
        r = results[name]
        print(f"{name:<52}{s.mean():.3f}  {s[dfl].mean():.3f}    {s[~dfl].mean():.3f}    {r['macro_f1']:.3f}"
              f"  x{r['count_ratio']:.2f}  {gain.mean():+.3f}  {r['up']}/{r['down']}", flush=True)
        if name != "v5":
            print("      per match: " + " ".join(f"{i}:{g:+.3f}" for i, g in zip(ids, gain)))
        print(f"      best single setting: {best}  {best_mean:.3f}")

    candidate = list(results)[-1]
    c = results[candidate]
    passed = c["gain"] >= MIN_GAIN and c["up"] >= MIN_UP
    complete = len(have) == len(args.draws)
    print(f"\ncandidate: {candidate}")
    print(f"10-match part of the rule (gain at least {MIN_GAIN}, at least {MIN_UP} of {len(ids)} up): "
          + ("PASSED" if passed else "NOT passed") + ("" if complete else "  (not all draws are in yet)"))
    record.update(results=results, candidate=candidate, passed_10_match_part=bool(passed), complete=complete)
    save_manifest(args.out, record)
