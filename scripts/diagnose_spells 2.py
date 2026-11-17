"""Where do possessions get lost? Reads saved held-out probabilities, so it takes seconds.

For every true possession, reports how long it was, whether it was matched, and if not, why not.

    python3 scripts/diagnose_spells.py --tag v3
"""

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.match import D  # noqa: E402
from sweep_decoder import load  # noqa: E402

BUCKETS = (0, 1, 2, 4, 8, 16, 32, 1e9)
FPS = 25.0


def bucket(seconds):
    return int(np.searchsorted(BUCKETS, seconds, side="right")) - 1


def bucket_name(i):
    top = "+" if BUCKETS[i + 1] > 1e6 else f"-{BUCKETS[i + 1]:g}"
    return f"{BUCKETS[i]:g}{top} s"


def why_missed(g, pred_spells, pred):
    """Reason a true possession was not matched at IoU 0.5."""
    g0, g1, team = g
    same = [p for p in pred_spells if p[2] == team and p[0] < g1 and p[1] > g0]
    if not same:
        window = pred[g0:g1]
        return "shown as stoppage" if (window == D).mean() > 0.5 else "shown as other team"
    longest = max(same, key=lambda p: p[1] - p[0])
    length = longest[1] - longest[0]
    if len(same) > 1 and sum(min(p[1], g1) - max(p[0], g0) for p in same) > 0.5 * (g1 - g0):
        return "split in pieces"
    if length > 2 * (g1 - g0):
        return "merged into longer"
    if length < 0.5 * (g1 - g0):
        return "only part found"
    return "boundaries off"


def analyse(folds, key="prob", cfg=DECODER):
    table = collections.defaultdict(lambda: collections.Counter())
    reasons = collections.defaultdict(lambda: collections.Counter())
    false_pos = collections.Counter()
    for fold in folds.values():
        gt = fold["labels"]
        halves = [tuple(int(v) for v in r) for r in fold["halves"]]
        path = decode_samples(fold[key], fold["half"], cfg)
        pred = expand(path, fold["frames"], fold["half"], halves, len(gt))
        runs = m.valid_runs(gt)
        gs, ps = m.possession_spells(gt, runs), m.possession_spells(pred, runs)
        ov = m._overlaps(gs, ps)
        flags = {t: m._matched(ov, len(gs), t) for t in m.SPELL_IOUS}
        for i, g in enumerate(gs):
            b = bucket((g[1] - g[0]) / FPS)
            table[b]["n"] += 1
            table[b]["frames"] += g[1] - g[0]
            table[b]["hit"] += np.mean([flags[t][i] for t in m.SPELL_IOUS])
            if not flags[0.5][i]:
                reasons[b][why_missed(g, ps, pred)] += 1
        taken_gt, matched = set(), set()
        for iou, i, j in ov:  # the metric's one-to-one matching, keeping the predicted side
            if iou >= 0.5 and i not in taken_gt and j not in matched:
                taken_gt.add(i)
                matched.add(j)
        for j, p in enumerate(ps):
            b = bucket((p[1] - p[0]) / FPS)
            false_pos["all", b] += 1
            if j not in matched:
                false_pos["false", b] += 1
    return table, reasons, false_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v3")
    args = parser.parse_args()
    folds = load(args.tag)
    table, reasons, false_pos = analyse(folds)

    total = sum(t["n"] for t in table.values())
    print(f"{len(folds)} matches, {total} true possessions\n")
    print(f"{'length':<10}{'true':>6}{'share':>7}{'found':>7}{'pred':>6}{'false':>7}")
    for b in sorted(table):
        t = table[b]
        print(f"{bucket_name(b):<10}{t['n']:>6}{t['n'] / total:>7.1%}{t['hit'] / t['n']:>7.1%}"
              f"{false_pos['all', b]:>6}{false_pos['false', b]:>7}")
    print("\nwhy true possessions were missed at IoU 0.5")
    names = sorted({k for r in reasons.values() for k in r})
    print(f"{'length':<10}" + "".join(f"{n[:18]:>20}" for n in names))
    for b in sorted(reasons):
        print(f"{bucket_name(b):<10}" + "".join(f"{reasons[b][n]:>20}" for n in names))
