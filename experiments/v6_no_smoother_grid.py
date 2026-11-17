"""Is the candidate without its smoother ahead of v5 whatever the decoder setting?

On 11 Nov the stress table by piece showed the smoother to be the part of the v6 candidate that
breaks. The rest of it (extra inputs, three models averaged) looked level with v5 on SkillCorner's
working half at the decoder setting picked for it on the 10 open matches, and ahead on all 10
matches at v5's own setting. A result that flips with the setting needs a second look.

Here v5 and the candidate without its smoother are decoded with the same setting, for every setting
of the grid, on three sets of matches: the 10 open matches (saved held-out probabilities), SkillCorner's
working half (models saved by experiments/skillcorner_v6.py), and three rows of the stress table
(probabilities saved by experiments/stress_v6_parts.py). Nothing is chosen from these tables. They
only say whether the gap depends on the setting.

Only the 10 working matches of SkillCorner. The reserve matches are never loaded here. Open data only.

    python3 experiments/v6_no_smoother_grid.py      # about 15 minutes
"""

import itertools
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah import harness  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import DecoderConfig  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from skillcorner_v6 import DEV, V5_MODEL, V6_MODEL  # noqa: E402
from stress_test import HELD_OUT  # noqa: E402
from stress_v6_parts import file_name  # noqa: E402
from sweep_decoder import GRID, score  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

CONFIGS = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
SHIPPED = CONFIGS.index(DECODER)
STRESS_ROWS = ("as trained", "all harsher", "very harsh")
V5_COLUMN, AVERAGED_COLUMN = "prob0", "prob4"      # the order of pieces in stress_v6_parts.py
OUT = os.path.join("runs", "v6_no_smoother_grid.json")


def grid(folds, ids):
    """spellF1 for every decoder setting (rows) and match (columns)."""
    return np.array([[score(folds[i], cfg)["spell_f1"] for i in ids] for cfg in CONFIGS])


def report(title, base, other):
    gain = other - base
    mean = gain.mean(1)
    at = gain[SHIPPED]
    by_turnover = {t: float(np.mean([mean[k] for k, c in enumerate(CONFIGS) if c.turnover == t]))
                   for t in GRID["turnover"]}
    print(f"\n{title}  ({base.shape[1]} matches)")
    print(f"  v5's shipped setting for both: v5 {base[SHIPPED].mean():.3f}, without the smoother "
          f"{other[SHIPPED].mean():.3f}, gain {at.mean():+.3f}, {(at > 0).sum()} up / {(at < 0).sum()} down")
    print(f"  the same setting for both, each of the {len(CONFIGS)} settings: ahead on average at "
          f"{(mean > 0).sum()} of them; gain {mean.min():+.3f} at worst, {np.median(mean):+.3f} in the middle, "
          f"{mean.max():+.3f} at best")
    print("  by turnover cost: " + "  ".join(f"{t:g}: {g:+.3f}" for t, g in by_turnover.items()))
    print(f"  each at its own best setting on these matches (a description, nothing is chosen from it): "
          f"v5 {base.mean(1).max():.3f}, without the smoother {other.mean(1).max():.3f}", flush=True)
    return {"shipped_v5": float(base[SHIPPED].mean()), "shipped_other": float(other[SHIPPED].mean()),
            "shipped_gain": float(at.mean()), "shipped_up": int((at > 0).sum()), "shipped_down": int((at < 0).sum()),
            "settings_ahead": int((mean > 0).sum()), "settings": len(CONFIGS), "worst_gain": float(mean.min()),
            "median_gain": float(np.median(mean)), "best_gain": float(mean.max()), "by_turnover": by_turnover,
            "best_v5": float(base.mean(1).max()), "best_other": float(other.mean(1).max())}


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(matches=list(DEV), stress_rows=list(STRESS_ROWS), draws=list(DRAWS))
    results = {}

    started = time.time()
    folds5, folds6 = load("v5"), averaged(DRAWS)
    ids = sorted(folds5)
    results["the 10 open matches"] = report("the 10 open matches, held-out", grid(folds5, ids), grid(folds6, ids))
    print(f"  ({time.time() - started:.0f}s)", flush=True)

    started = time.time()
    with open(V5_MODEL, "rb") as handle:
        v5 = pickle.load(handle)
    with open(V6_MODEL, "rb") as handle:
        models = pickle.load(handle)["ensemble"].models
    stats = load_stats()
    folds5, folds6 = {}, {}
    for match_id in DEV:
        match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        blind = harness.without_labels(match)
        p5, frames5, half5 = v5.probabilities(blind)
        values, swapped, frames, half = models[0].views(blind)
        mean = sum(model.probabilities_from(values, swapped) for model in models) / float(len(models))
        truth = {"labels": match.labels, "halves": np.array(match.halves)}
        folds5[match_id] = dict(truth, prob=p5, frames=frames5, half=half5)
        folds6[match_id] = dict(truth, prob=mean, frames=frames, half=half)
    results["SkillCorner's working half"] = report("SkillCorner's working half", grid(folds5, DEV), grid(folds6, DEV))
    print(f"  ({time.time() - started:.0f}s)", flush=True)

    for row in STRESS_ROWS:
        started = time.time()
        folds5, folds6 = {}, {}
        for held in HELD_OUT:
            with np.load(file_name(row, held), allow_pickle=False) as z:
                names = [str(p) for p in z["pieces"]]
                if names[0] != "v5" or not names[4].endswith("averaged"):
                    raise ValueError(f"unexpected order of pieces in {file_name(row, held)}: {names}")
                shared = {k: z[k] for k in ("labels", "halves", "frames", "half")}
                folds5[held] = dict(shared, prob=z[V5_COLUMN])
                folds6[held] = dict(shared, prob=z[AVERAGED_COLUMN])
        results[f"stress row '{row}'"] = report(f"stress row '{row}'", grid(folds5, HELD_OUT), grid(folds6, HELD_OUT))
        print(f"  ({time.time() - started:.0f}s)", flush=True)

    record.update(results=results)
    save_manifest(OUT, record)
