"""Is trees + TCN ahead of the trees whatever the decoder setting? And would a setting picked for it at home carry?

The decoder setting in use was chosen for v5. The candidate (the trees and the TCN, averaged) gives
different probabilities, so two questions:

1. Does its lead over the trees depend on the setting? Both are decoded with the same setting, for each
   of the 175 settings of the grid, on three sets of matches the models had not seen: the 10 open
   matches, the unseen-source test, and SkillCorner's working half.
2. If the setting were picked again for the candidate on the 10 open matches, what would that setting
   do on the other two sets? On 11 Nov a setting picked at home for the v6 candidate cost it its lead on
   SkillCorner, so this is a what-if. Nothing is chosen from these tables.

Works from saved probabilities. The reserve half of SkillCorner is never loaded. Open data only.

    python3 experiments/candidate_decoder_grid.py      # about 15 minutes
"""

import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import DecoderConfig  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from sweep_decoder import GRID, score  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

CONFIGS = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
SHIPPED = CONFIGS.index(DECODER)
SETS = (("the 10 open matches, held out", None, "net_tcn"),
        ("an unseen label source", "trees_unseen", "net_tcn_unseen"),
        ("SkillCorner's working half", "trees_sc", "net_tcn_sc"))
OUT = os.path.join("runs", "candidate_decoder_grid.json")


def grid(folds, ids):
    """spellF1 for every decoder setting (rows) and match (columns)."""
    return np.array([[score(folds[i], cfg)["spell_f1"] for i in ids] for cfg in CONFIGS])


def words(cfg):
    return f"turnover {cfg.turnover:g}, to dead {cfg.to_dead:g}, from dead {cfg.from_dead:g}, temperature {cfg.temperature:g}"


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(draws=list(DRAWS))
    tables, results = {}, {}
    for title, tree_tag, net_tag in SETS:
        started = time.time()
        trees = averaged(DRAWS) if tree_tag is None else load(tree_tag)
        net = load(net_tag)
        ids = sorted(trees)
        both = {}
        for i in ids:
            if not np.array_equal(trees[i]["frames"], net[i]["frames"]):
                raise ValueError(f"{i}: the trees and the TCN were not predicted at the same moments")
            both[i] = dict(trees[i], prob=(trees[i]["prob"] + net[i]["prob"]) / 2.0)
        base, other = grid(trees, ids), grid(both, ids)
        tables[title] = (base, other)
        gain = other - base
        mean, at = gain.mean(1), gain[SHIPPED]
        by_turnover = {t: float(np.mean([mean[k] for k, c in enumerate(CONFIGS) if c.turnover == t]))
                       for t in GRID["turnover"]}
        print(f"\n{title}  ({len(ids)} matches)")
        print(f"  v5's setting for both: trees {base[SHIPPED].mean():.3f}, trees + TCN {other[SHIPPED].mean():.3f}, "
              f"gain {at.mean():+.3f}, {(at > 0).sum()} up / {(at < 0).sum()} down")
        print(f"  the same setting for both, each of the {len(CONFIGS)} settings: ahead on average at "
              f"{(mean > 0).sum()} of them; gain {mean.min():+.3f} at worst, {np.median(mean):+.3f} in the middle, "
              f"{mean.max():+.3f} at best")
        print("  by turnover cost: " + "  ".join(f"{t:g}: {g:+.3f}" for t, g in by_turnover.items()))
        print(f"  each at its own best setting on these matches (a description, nothing is chosen from it): "
              f"trees {base.mean(1).max():.3f}, trees + TCN {other.mean(1).max():.3f}")
        print(f"  ({time.time() - started:.0f}s)", flush=True)
        results[title] = {"shipped_trees": float(base[SHIPPED].mean()), "shipped_candidate": float(other[SHIPPED].mean()),
                          "shipped_gain": float(at.mean()), "shipped_up": int((at > 0).sum()),
                          "shipped_down": int((at < 0).sum()), "settings_ahead": int((mean > 0).sum()),
                          "settings": len(CONFIGS), "worst_gain": float(mean.min()),
                          "median_gain": float(np.median(mean)), "best_gain": float(mean.max()),
                          "by_turnover": by_turnover, "best_trees": float(base.mean(1).max()),
                          "best_candidate": float(other.mean(1).max()),
                          "candidate_by_setting": [float(v) for v in other.mean(1)]}

    home = tables[SETS[0][0]][1]
    picked = int(home.mean(1).argmax())
    print(f"\nwhat if the setting were picked again for trees + TCN on the 10 open matches?")
    print(f"  v5's setting: {words(CONFIGS[SHIPPED])}")
    print(f"  the pick:     {words(CONFIGS[picked])}")
    carry = {}
    for title, _, _ in SETS:
        other = tables[title][1]
        move = other[picked] - other[SHIPPED]
        best = int(other.mean(1).argmax())
        print(f"  {title}: {other[SHIPPED].mean():.3f} at v5's setting, {other[picked].mean():.3f} at the pick "
              f"({move.mean():+.3f}, {(move > 0).sum()} up / {(move < 0).sum()} down); "
              f"best on these matches {other[best].mean():.3f} at {words(CONFIGS[best])}")
        carry[title] = {"shipped": float(other[SHIPPED].mean()), "picked": float(other[picked].mean()),
                        "move": float(move.mean()), "up": int((move > 0).sum()), "down": int((move < 0).sum()),
                        "best": float(other[best].mean()), "best_setting": words(CONFIGS[best])}
    record.update(results=results, picked_at_home=words(CONFIGS[picked]), carry=carry)
    save_manifest(OUT, record)
